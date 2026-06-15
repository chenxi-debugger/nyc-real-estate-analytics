"""
data_service.py
================
负责 NYC 房产数据的加载和清洗。

设计思路:
    把"持有数据 + 操作数据"的逻辑打包成一个 class,
    其他模块(model_service, plot_helper, app)拿到 DataService 实例后,
    通过 self.df 访问清洗后的 DataFrame。

清洗步骤(按需求 1.3):
    1. 去重 (drop_duplicates)
    2. 数值列强制转 numeric (errors='coerce' 把脏数据标 NaN)
    3. 过滤明显无效值:
         - SALE PRICE > 10000          (排除 $1 名义转让)
         - GROSS SQUARE FEET > 0       (面积必须为正)
         - YEAR BUILT 在 1800 ~ 2030   (排除 0 / 未来年份)
         - TOTAL UNITS 在 1 ~ 1000      (排除 0 / 异常大值)
    4. 派生列:
         - BOROUGH_NAME       (1→Manhattan ... 5→Staten Island)
         - Price_Per_SqFt     (单位面积价格,给箱线图用)
         - SALE DATE 转 datetime (给折线图按月聚合用)
"""

import pandas as pd


class DataService:
    """加载 + 清洗 NYC 房产数据。"""

    # 类属性:Borough 编码 → 名字。所有 DataService 实例共享。
    BOROUGH_MAP = {
        1: "Manhattan",
        2: "Bronx",
        3: "Brooklyn",
        4: "Queens",
        5: "Staten Island",
    }

    # 必须转成数值类型的列
    NUMERIC_COLS = ["GROSS SQUARE FEET", "SALE PRICE", "TOTAL UNITS", "YEAR BUILT"]

    def __init__(self, file_path):
        """
        构造器。只记下文件路径,真正读文件留给 load_data()。

        参数:
            file_path: CSV 文件的路径 (str 或 Path 对象都行)

        实例状态:
            self.file_path: 文件路径
            self.df: 清洗后的 DataFrame,初始 None,load_data() 之后才有内容
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        读 CSV → 清洗 → 写到 self.df。

        返回:
            True  : 加载 + 清洗成功
            False : 文件找不到或其他异常
        """
        try:
            print(f"📂 Loading data from {self.file_path} ...")
            df = pd.read_csv(self.file_path)
            print(f"   raw rows: {len(df):,}")

            # ===== Step 1: 去重 =====
            df = df.drop_duplicates().copy()

            # ===== Step 2: 数值列强制转 numeric =====
            # errors='coerce' 的意思:转不了的值标记成 NaN(而不是抛异常)
            # 这样后面 dropna 可以一并清掉
            for col in self.NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # ===== Step 3: 过滤明显无效值 =====
            df = df[
                (df["SALE PRICE"] > 10_000)
                & (df["GROSS SQUARE FEET"] > 0)
                & (df["YEAR BUILT"] > 1800)
                & (df["YEAR BUILT"] <= 2030)
                & (df["TOTAL UNITS"] >= 1)
                & (df["TOTAL UNITS"] <= 1000)
                & (df["BOROUGH"].isin(self.BOROUGH_MAP.keys()))
            ]

            # 上一步可能留下 NaN(因为 to_numeric coerce 出来的),整行清掉
            df = df.dropna(subset=self.NUMERIC_COLS + ["SALE DATE"])

            # ===== Step 4: 派生列 =====
            df["SALE DATE"] = pd.to_datetime(df["SALE DATE"])
            df["BOROUGH_NAME"] = df["BOROUGH"].astype(int).map(self.BOROUGH_MAP)
            df["Price_Per_SqFt"] = df["SALE PRICE"] / df["GROSS SQUARE FEET"]

            # 重置索引(过滤后索引会有空洞)
            df = df.reset_index(drop=True)

            self.df = df
            print(f"✅ Cleaned rows: {len(self.df):,} (dropped {self._dropped_count():,})")
            return True

        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return False

    def _dropped_count(self):
        """
        辅助方法:返回被清洗掉的行数。
        以下划线开头的方法是 Python 惯例,表示"这是内部用的,外部别调"。
        """
        try:
            raw = pd.read_csv(self.file_path, usecols=[0])
            return len(raw) - len(self.df)
        except Exception:
            return 0


# ============================================================
# 当这个文件被直接运行时(python app/services/data_service.py),
# 跑下面这段做自测;被 import 时不会跑。
# demo 里的标准做法,每个模块独立可测。
# ============================================================
if __name__ == "__main__":
    from pathlib import Path

    # __file__       是当前文件的路径
    # .resolve()     转成绝对路径
    # .parents[2]    往上跳 2 层(services → app → 项目根)
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "nyc_real_estate.csv"

    service = DataService(csv_path)
    if service.load_data():
        print("\n========== 数据总览 ==========")
        print(f"行数: {len(service.df):,}")
        print(f"列数: {len(service.df.columns)}")
        print(f"\n前 3 行:")
        print(service.df.head(3))
        print(f"\nBorough 分布:")
        print(service.df["BOROUGH_NAME"].value_counts())
    else:
        print("数据加载失败")