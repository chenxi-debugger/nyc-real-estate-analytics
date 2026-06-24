"""
model_service.py
================
模型服务层 —— 运行时【加载 + 预测】。

职责 (单一):
    1. 加载 train_and_evaluate.py 产出的冠军模型 (joblib 缓存)
    2. 对外提供 predict():输入房产信息 → 输出预测价格
    3. 持有特征工程 / 预处理的"定义",供训练脚本复用

重要设计 —— 训练逻辑不在这里:
    本类【不自己训练模型】。所有"多模型对比 + 调参 + 选优"的逻辑
    集中在 train_and_evaluate.py 一处,避免重复。
    fit_or_load() 的行为:
        - 有缓存 → 直接加载
        - 无缓存 → 回调 train_and_evaluate.train_and_select() 训练一个
                  (训练代码仍然只有那一份,这里只是"借用")

特征 (用户填 5 个 + 后端衍生 3 个):
    用户填:  GROSS SQUARE FEET / YEAR BUILT / TOTAL UNITS / BOROUGH / NEIGHBORHOOD
    后端算:  SQFT_PER_UNIT / BUILDING_AGE / SALE_MONTH

关键约定 (训练与预测必须一致):
    1. 目标用 log10(SALE PRICE),预测时 10**y_hat 还原
    2. 类别特征用 OneHot,无 target encoding,无泄漏风险
    3. 数值特征:线性回归用 StandardScaler,树模型 passthrough
    4. predict 包 try/except,输入异常不让 Flask 崩
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelService:
    """模型服务:加载冠军模型 + 预测。训练委托给 train_and_evaluate.py。"""

    # ---- 特征列定义 (训练脚本也复用这些,作为单一数据源) ----
    # 用户输入的数值特征
    USER_NUMERIC = ["GROSS SQUARE FEET", "YEAR BUILT", "TOTAL UNITS"]
    # 后端从用户输入衍生的数值特征 (预测时仍随输入变化,不是常量)
    DERIVED_NUMERIC = ["SQFT_PER_UNIT", "BUILDING_AGE"]
    NUMERIC_FEATURES = USER_NUMERIC + DERIVED_NUMERIC
    # 类别特征 —— 全部由用户提供 (区/小区/建筑类型),没有常量兜底
    CATEGORICAL_FEATURES = ["BOROUGH", "NEIGHBORHOOD", "BUILDING CLASS CATEGORY"]
    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    TARGET = "SALE PRICE"

    # 建筑类型若意外缺失时的回退值 (正常情况用户会从下拉框选)
    DEFAULT_BUILDING_CLASS = "01 ONE FAMILY DWELLINGS"

    RANDOM_STATE = 42
    CV_FOLDS = 5
    LOWER_PCT = 0.005
    UPPER_PCT = 0.995

    def __init__(self, model_path):
        self.model_path = model_path
        self.pipeline = None
        self.champion = None
        self.metrics = None
        self.leaderboard = None

    # ============================================================
    # 公开方法:加载(或触发训练) / 预测
    # ============================================================

    def fit_or_load(self, df):
        """
        加载出厂自带的冠军模型。

        正常情况下,训练好的 champion_pipeline.pkl 随项目一起提交,
        app 启动只需加载它 —— 用户/招聘官 clone 下来即开即用,
        不需要训练。

        训练逻辑【不在本文件】,集中在 train_and_evaluate.py,作为
        "模型选优"的工作留痕,供查看与复现,但不在产品运行路径上。

        仅当模型文件意外缺失时 (例如有人手滑删了 models/),才回退到
        现场训练一个,保证服务不中断 —— 这是兜底,不是常规路径。
        """
        if self._cache_exists():
            self._load()
            return

        # —— 兜底:模型文件缺失才会走到这里 ——
        print("   ⚠️  champion_pipeline.pkl not found — this file ships with "
              "the project and should normally exist.")
        print("   Falling back to training one now (one-off). "
              "To regenerate deliberately, run: python train_and_evaluate.py")
        from train_and_evaluate import train_and_select
        bundle = train_and_select(df, save=True, make_plots=True, verbose=True)
        self._apply_bundle(bundle)

    def predict(self, gross_sqft, year_built, total_units, borough,
                neighborhood, building_class=None):
        """预测单条房产的 SALE PRICE (美元)。失败抛 RuntimeError。"""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call fit_or_load() first.")

        try:
            year_built = int(year_built)
            total_units = int(total_units)
            gross_sqft = float(gross_sqft)
            if building_class is None:
                building_class = self.DEFAULT_BUILDING_CLASS

            sqft_per_unit = gross_sqft / total_units if total_units else gross_sqft
            current_year = pd.Timestamp.now().year
            building_age = max(current_year - year_built, 0)

            X = pd.DataFrame([{
                "GROSS SQUARE FEET": gross_sqft,
                "YEAR BUILT": year_built,
                "TOTAL UNITS": total_units,
                "SQFT_PER_UNIT": sqft_per_unit,
                "BUILDING_AGE": building_age,
                "BOROUGH": int(borough),
                "NEIGHBORHOOD": str(neighborhood),
                "BUILDING CLASS CATEGORY": str(building_class),
            }])
            X = self._coerce_categoricals(X)

            y_log = self.pipeline.predict(X)[0]
            return float(10 ** y_log)

        except Exception as exc:
            raise RuntimeError(f"Prediction failed: {exc}") from exc

    # ============================================================
    # 预处理构造器 (训练脚本 train_and_evaluate 复用这些)
    # ============================================================

    def _build_linear_pre(self):
        """线性模型的预处理:数值列标准化 + 类别列 OneHot。"""
        ct = ColumnTransformer([
            ("num", StandardScaler(), self.NUMERIC_FEATURES),
            ("cat", self._make_ohe(), self.CATEGORICAL_FEATURES),
        ])
        # 输出保留为带列名的 DataFrame —— 让下游模型见到一致的特征名,
        # 消除 "X does not have valid feature names" 警告
        ct.set_output(transform="pandas")
        return ct

    def _fresh_tree_pre(self):
        """树模型的预处理:数值列原样 + 类别列 OneHot (每次新建一份)。"""
        ct = ColumnTransformer([
            ("num", "passthrough", self.NUMERIC_FEATURES),
            ("cat", self._make_ohe(), self.CATEGORICAL_FEATURES),
        ])
        ct.set_output(transform="pandas")
        return ct

    def _make_ohe(self):
        """跨 sklearn 版本兼容的 OneHotEncoder 工厂。"""
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    # ============================================================
    # 特征工程 + 清洗 (训练脚本复用)
    # ============================================================

    def _engineer_features(self, df):
        """从原始列衍生 SQFT_PER_UNIT / BUILDING_AGE。"""
        df = df.copy()

        # 房龄需要成交年份;SALE DATE 仅用于此,不再单独作为月份特征
        if "SALE DATE" in df.columns:
            sale_year = pd.to_datetime(df["SALE DATE"], errors="coerce").dt.year
        else:
            sale_year = pd.Series(pd.Timestamp.now().year, index=df.index)

        df["BUILDING_AGE"] = (sale_year - df["YEAR BUILT"]).clip(lower=0).fillna(0)

        units = df["TOTAL UNITS"].replace(0, np.nan)
        df["SQFT_PER_UNIT"] = (df["GROSS SQUARE FEET"] / units).fillna(
            df["GROSS SQUARE FEET"]
        )

        if "BUILDING CLASS CATEGORY" not in df.columns:
            df["BUILDING CLASS CATEGORY"] = self.DEFAULT_BUILDING_CLASS
        else:
            df["BUILDING CLASS CATEGORY"] = (
                df["BUILDING CLASS CATEGORY"].astype(str).str.strip()
            )
        return df

    def _coerce_categoricals(self, X):
        """类别列统一成 str (OneHot 才能稳定工作)。"""
        X = X.copy()
        for col in self.CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip()
        return X

    def _trim_outliers(self, df):
        """裁掉 SALE PRICE 和 GROSS SQUARE FEET 上下 0.5% 极端值。"""
        lo_p, hi_p = df[self.TARGET].quantile([self.LOWER_PCT, self.UPPER_PCT])
        lo_s, hi_s = df["GROSS SQUARE FEET"].quantile([self.LOWER_PCT, self.UPPER_PCT])
        return df[
            df[self.TARGET].between(lo_p, hi_p)
            & df["GROSS SQUARE FEET"].between(lo_s, hi_s)
        ].copy()

    # ============================================================
    # 缓存
    # ============================================================

    def _cache_exists(self):
        from pathlib import Path
        return Path(self.model_path).exists()

    def _apply_bundle(self, bundle):
        """把训练产出的 bundle 装进 self。"""
        self.pipeline = bundle["pipeline"]
        self.champion = bundle.get("champion")
        self.metrics = bundle.get("metrics")
        self.leaderboard = bundle.get("leaderboard")

    def _load(self):
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict) and "pipeline" in bundle:
            self._apply_bundle(bundle)
        else:
            self.pipeline = bundle  # 兼容老格式 (直接 dump pipeline)
        print(f"Loaded cached model ({self.champion}) from {self.model_path}")


# ============================================================
# 自测:python -m app.services.model_service
# ============================================================
if __name__ == "__main__":
    from pathlib import Path
    from app.services.data_service import DataService

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "nyc_real_estate.csv"
    model_path = project_root / "models" / "champion_pipeline.pkl"

    data_svc = DataService(csv_path)
    if not data_svc.load_data():
        print("数据加载失败")
        exit(1)

    model_svc = ModelService(model_path)
    model_svc.fit_or_load(data_svc.df)

    if model_svc.leaderboard:
        print("\n========== Leaderboard ==========")
        for row in model_svc.leaderboard:
            print(f"  {row['model']:18s} CV R2={row['cv_r2_mean']:.3f}")

    print("\n========== 预测测试 ==========")
    for sqft, year, units, boro, nbhd, note in [
        (1500, 1998, 2, 3, "PARK SLOPE", "Brooklyn 2-family"),
        (5000, 2010, 1, 1, "UPPER EAST SIDE (79-96)", "Manhattan high-end"),
        (900, 1920, 1, 2, "RIVERDALE", "Bronx small"),
    ]:
        try:
            price = model_svc.predict(
                gross_sqft=sqft, year_built=year, total_units=units,
                borough=boro, neighborhood=nbhd,
            )
            print(f"  {note:22s} -> ${price:>12,.2f}")
        except RuntimeError as e:
            print(f"  {note:22s} -> {e}")
