"""
model_service.py
================
KNN 回归模型,预测 NYC 房产 SALE PRICE。

模型流水线 (Pipeline):

    ColumnTransformer
        ├── 数值列: StandardScaler  (GROSS SQUARE FEET / YEAR BUILT / TOTAL UNITS)
        └── 分类列: OneHotEncoder   (BOROUGH)
        │
        ▼
    KNeighborsRegressor(n_neighbors=10, weights='distance')

关键设计:
    1. 训练目标用 log10(SALE PRICE),预测时 10**y_hat 还原 —— 解决长尾分布问题
    2. 训练前裁掉极端值 (top/bottom 0.5%) —— 避免 outliers 污染邻居搜索
    3. 模型用 joblib 缓存到 models/knn_pipeline.pkl —— 重启不用重训
    4. 预测方法包 try/except —— 输入异常时不让 Flask 崩
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelService:
    """KNN 模型的封装:训练、缓存、加载、预测。"""

    # 特征列定义(类属性,所有实例共享)
    NUMERIC_FEATURES = ["GROSS SQUARE FEET", "YEAR BUILT", "TOTAL UNITS"]
    CATEGORICAL_FEATURES = ["BOROUGH"]
    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    TARGET = "SALE PRICE"

    # KNN 超参数
    N_NEIGHBORS = 10
    RANDOM_STATE = 42

    # 训练集裁剪上下分位
    LOWER_PCT = 0.005   # 裁掉最低 0.5%
    UPPER_PCT = 0.995   # 裁掉最高 0.5%

    def __init__(self, model_path):
        """
        参数:
            model_path: .pkl 缓存文件路径 (str 或 Path)

        实例状态:
            self.model_path: 缓存路径
            self.pipeline:   sklearn Pipeline,初始 None,fit_or_load() 之后才有
            self.metrics:    评估指标 dict {'r2', 'mae', 'n_train', 'n_test'}
        """
        self.model_path = model_path
        self.pipeline = None
        self.metrics = None

    # ============================================================
    # 公开方法:训练 / 加载 / 预测
    # ============================================================

    def fit_or_load(self, df):
        """
        智能加载:有 .pkl 缓存就 load,没有就训练并保存。

        参数:
            df: DataService.df,清洗后的完整 DataFrame
        """
        if self._cache_exists():
            self._load()
        else:
            self._train(df)
            self._save()

    def predict(self, gross_sqft, year_built, total_units, borough):
        """
        预测单条房产的 SALE PRICE。

        参数:
            gross_sqft:  毛建筑面积(平方英尺)
            year_built:  建造年份
            total_units: 总单元数
            borough:     行政区编码 (1-5)

        返回:
            预测价格(美元)。失败时抛 RuntimeError(由调用方 catch)。
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call fit_or_load() first.")

        try:
            # 构造输入 DataFrame —— 列名必须和训练时一致
            X = pd.DataFrame([{
                "GROSS SQUARE FEET": float(gross_sqft),
                "YEAR BUILT": int(year_built),
                "TOTAL UNITS": int(total_units),
                "BOROUGH": int(borough),
            }])

            # Pipeline 内部:ColumnTransformer 处理 → KNN 预测
            # 注意预测出来的是 log10(price),需要 10**y 还原
            y_log = self.pipeline.predict(X)[0]
            return float(10 ** y_log)

        except Exception as exc:
            raise RuntimeError(f"Prediction failed: {exc}") from exc

    # ============================================================
    # 内部方法:训练 / 保存 / 加载 (下划线开头 = 私有)
    # ============================================================

    def _train(self, df):
        """训练 KNN 并算评估指标。"""
        print("🏋️  Training KNN model...")

        # Step 1: 裁掉极端值(top/bottom 0.5%),避免 outliers 污染邻居搜索
        df_trimmed = self._trim_outliers(df)
        print(f"   trimmed: {len(df):,} → {len(df_trimmed):,} rows")

        # Step 2: 准备 X (4 列原始特征) 和 y (log10 价格)
        X = df_trimmed[self.FEATURES]
        y = np.log10(df_trimmed[self.TARGET].values)

        # Step 3: 切训练 / 测试集 (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.RANDOM_STATE
        )

        # Step 4: 构造 Pipeline 并训练
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Step 5: 评估 —— 在测试集上预测,转回原 scale 计算指标
        y_pred_log = self.pipeline.predict(X_test)
        y_pred = np.power(10, y_pred_log)
        y_true = np.power(10, y_test)

        self.metrics = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

        print(
            f"✅ Trained | n_train={self.metrics['n_train']:,} "
            f"n_test={self.metrics['n_test']:,} "
            f"MAE=${self.metrics['mae']:,.0f} "
            f"R²={self.metrics['r2']:.3f}"
        )

    def _build_pipeline(self):
        """构造预处理 + KNN 的 Pipeline。"""
        # OneHotEncoder 的参数名在不同 sklearn 版本里不一样,兼容写法
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # 老版本
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preproc = ColumnTransformer([
            ("num", StandardScaler(), self.NUMERIC_FEATURES),
            ("cat", ohe, self.CATEGORICAL_FEATURES),
        ])

        knn = KNeighborsRegressor(
            n_neighbors=self.N_NEIGHBORS,
            weights="distance",   # 越近的邻居权重越大
            n_jobs=-1,            # 并行,加速预测
        )

        return Pipeline([("preproc", preproc), ("knn", knn)])

    def _trim_outliers(self, df):
        """裁掉 SALE PRICE 和 GROSS SQUARE FEET 上下 0.5% 的极端值。"""
        lo_p, hi_p = df[self.TARGET].quantile([self.LOWER_PCT, self.UPPER_PCT])
        lo_s, hi_s = df["GROSS SQUARE FEET"].quantile([self.LOWER_PCT, self.UPPER_PCT])
        return df[
            df[self.TARGET].between(lo_p, hi_p)
            & df["GROSS SQUARE FEET"].between(lo_s, hi_s)
        ].copy()

    def _cache_exists(self):
        """检查 .pkl 缓存文件是否存在。"""
        from pathlib import Path
        return Path(self.model_path).exists()

    def _save(self):
        """把训练好的 pipeline 保存到磁盘。"""
        from pathlib import Path
        # 确保 models/ 目录存在
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"💾 Saved model to {self.model_path}")

    def _load(self):
        """从磁盘加载已训练的 pipeline。"""
        self.pipeline = joblib.load(self.model_path)
        print(f"📦 Loaded cached model from {self.model_path}")


# ============================================================
# 自测代码:python app/services/model_service.py
# ============================================================
if __name__ == "__main__":
    from pathlib import Path

    # 复用 DataService 加载数据
    from app.services.data_service import DataService

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "nyc_real_estate.csv"
    model_path = project_root / "models" / "knn_pipeline.pkl"

    # 1. 加载数据
    data_svc = DataService(csv_path)
    if not data_svc.load_data():
        print("数据加载失败")
        exit(1)

    # 2. 训练或加载模型
    model_svc = ModelService(model_path)
    model_svc.fit_or_load(data_svc.df)

    # 3. 测试预测 —— 4 个典型场景
    test_cases = [
        # (sqft, year, units, borough, 备注)
        (1500, 1998, 2,  3, "Brooklyn 中产 2-family"),
        (5000, 2010, 1,  1, "Manhattan 高端 single"),
        (2500, 1965, 1,  4, "Queens 普通 single"),
        ( 900, 1920, 1,  2, "Bronx 老式 small"),
    ]

    print("\n========== 预测测试 ==========")
    for sqft, year, units, borough, note in test_cases:
        try:
            price = model_svc.predict(
                gross_sqft=sqft,
                year_built=year,
                total_units=units,
                borough=borough,
            )
            print(f"  {note:30s} → ${price:>12,.2f}")
        except RuntimeError as e:
            print(f"  {note:30s} → ❌ {e}")