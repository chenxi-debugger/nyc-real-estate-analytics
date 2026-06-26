"""
plot_helper.py
==============
负责生成 7 个 Plotly 图表的 JSON。

设计模式:
    每个图 = 一个 @staticmethod,接收 DataFrame,返回 JSON 字符串。
    generate_all_charts(df) 是总入口,一次性返回所有 7 个图的 dict。

为什么是 @staticmethod ?
    画图函数没有任何状态需要存(不像 DataService 要存 self.df),
    本质是"输入 df → 输出 JSON"的纯函数。
    但包装到 PlotService 类里有两个好处:
        1. 命名空间组织清晰
        2. 调用时可以 PlotService.xxx() 一目了然

7 个图(无重复类型,符合需求 1.2):
    1. Bar       —— Average Sale Price by Borough
    2. Pie       —— Building Class Share (top 6 + Other)
    3. Scatter   —— Sale Price vs Gross SqFt (log-log, sampled)
    4. Line      —— Monthly Average Sale Price Trend
    5. Histogram —— Sale Price Distribution (log10)
    6. Box       —— Price per SqFt by Borough
    7. Heatmap   —— Feature Correlation
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class PlotService:
    """生成 7 个 NYC 房产分析图表。所有方法都是 staticmethod。"""

    # ============================================================
    # 1. Bar —— 各区平均成交价
    # ============================================================
    @staticmethod
    def make_borough_bar(df):
        """各 borough 平均 SALE PRICE 柱状图。"""
        data = (
            df.groupby("BOROUGH_NAME")["SALE PRICE"]
            .mean()
            .sort_values(ascending=False)
        )

        fig = px.bar(
            data,
            labels={"value": "Average Sale Price ($)", "BOROUGH_NAME": "Borough"},
            color=data.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            showlegend=False,
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            margin=dict(t=20, b=40, l=60, r=20),
        )
        return fig.to_json()

    # ============================================================
    # 2. Pie —— 建筑类别占比(Top 6 + Other)
    # ============================================================
    @staticmethod
    def make_building_pie(df):
        """Building Class 占比饼图。聚合 Top 6,其余合并为 Other。"""
        all_counts = df["BUILDING CLASS CATEGORY"].value_counts()
        top = all_counts.head(6)
        other_sum = all_counts.iloc[6:].sum()
        data = pd.concat([top, pd.Series({"Other": other_sum})])

        fig = px.pie(
            values=data.values,
            names=data.index,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4,                   # 环形图
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            textfont_size=13,
            insidetextorientation="horizontal",
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.05,
                xanchor="center", x=0.5,
                font=dict(size=11),
            ),
        )
        return fig.to_json()

    # ============================================================
    # 3. Scatter —— 面积 vs 价格 (log-log + 按 borough 上色)
    # ============================================================
    @staticmethod
    def make_scatter(df, n_sample=2000, random_state=42):
        """
        面积 vs 价格散点图。

        关键设计:
            - 抽样 2000 点(防止前端卡死)
            - log-log 轴(房价是长尾分布,线性轴上点会全挤左下角)
        """
        data = df.sample(n=min(n_sample, len(df)), random_state=random_state)

        fig = px.scatter(
            data,
            x="GROSS SQUARE FEET",
            y="SALE PRICE",
            color="BOROUGH_NAME",
            log_x=True,
            log_y=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.5,
            hover_data={
                "GROSS SQUARE FEET": ":,.0f",
                "SALE PRICE": ":$,.0f",
            },
            labels={
                "GROSS SQUARE FEET": "Gross Square Feet",
                "SALE PRICE": "Sale Price ($)",
                "BOROUGH_NAME": "Borough",
            },
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(
            legend=dict(
                title="Borough",
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            margin=dict(t=50, b=40, l=60, r=20),
        )
        return fig.to_json()

    # ============================================================
    # 4. Line —— 月度平均成交价趋势
    # ============================================================
    @staticmethod
    def make_trend_line(df):
        """按月聚合的平均 SALE PRICE 折线图。"""
        data = (
            df.set_index("SALE DATE")
            .resample("ME")["SALE PRICE"]   # 'ME' = month end
            .mean()
        )

        fig = px.line(
            data,
            labels={"value": "Average Sale Price ($)", "SALE DATE": "Month"},
            markers=True,
        )
        fig.update_traces(
            line=dict(color="#2E86AB", width=2),
            marker=dict(size=6),
        )
        fig.update_layout(
            showlegend=False,
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            hovermode="x unified",
            margin=dict(t=20, b=40, l=60, r=20),
        )
        return fig.to_json()

    # ============================================================
    # 5. Histogram —— 价格分布 (log10) + 中位数标注
    # ============================================================
    @staticmethod
    def make_price_hist(df):
        """log10 价格分布直方图,加中位数虚线标注。"""
        log_prices = np.log10(df["SALE PRICE"])
        median_log = log_prices.median()
        median_actual = 10 ** median_log

        fig = px.histogram(
            log_prices,
            nbins=50,
            labels={"value": "log10(Sale Price)", "count": "Frequency"},
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.add_vline(
            x=median_log,
            line_dash="dash",
            line_color="navy",
            annotation_text=f"Median ≈ ${median_actual:,.0f}",
            annotation_position="top right",
        )
        fig.update_layout(
            showlegend=False,
            bargap=0.05,
            margin=dict(t=20, b=40, l=60, r=20),
        )
        return fig.to_json()

    # ============================================================
    # 6. Box —— 各区单位面积价格分布
    # ============================================================
    @staticmethod
    def make_pps_box(df, ppsf_cap=4000):
        """
        Price per SqFt 箱线图,按 borough 分组。

        关键设计:
            - 裁掉 Price_Per_SqFt > $4000 的极端值(防止箱体被压成一条线)
            - 按 borough median 排序(视觉对比更强)
        """
        data = df[df["Price_Per_SqFt"] <= ppsf_cap][["BOROUGH_NAME", "Price_Per_SqFt"]]

        borough_order = (
            data.groupby("BOROUGH_NAME")["Price_Per_SqFt"]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        fig = px.box(
            data,
            x="BOROUGH_NAME",
            y="Price_Per_SqFt",
            color="BOROUGH_NAME",
            category_orders={"BOROUGH_NAME": borough_order},
            labels={"BOROUGH_NAME": "Borough", "Price_Per_SqFt": "Price per SqFt ($)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            points="outliers",
        )
        fig.update_layout(
            showlegend=False,
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            margin=dict(t=20, b=40, l=60, r=20),
        )
        return fig.to_json()

    # ============================================================
    # 7. Heatmap —— 特征相关性
    # ============================================================
    @staticmethod
    def make_corr_heatmap(df):
        """
        关键特征的 Pearson 相关系数热图。

        关键设计:
            - 用 RdBu_r 发散色(正相关蓝,负相关红,白色 = 0)
            - zmin=-1, zmax=1 固定色阶,不同数据集可比
        """
        features = ["SALE PRICE", "GROSS SQUARE FEET", "YEAR BUILT",
                    "TOTAL UNITS", "Price_Per_SqFt"]
        corr = df[features].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            labels=dict(color="Correlation"),
        )
        fig.update_layout(
            xaxis=dict(side="bottom"),
            margin=dict(t=20, b=40, l=120, r=20),
        )
        return fig.to_json()

    # ============================================================
    # 总入口:一次生成所有 7 个图
    # ============================================================
    @staticmethod
    def generate_all_charts(df):
        """
        生成所有 7 个图,返回 dict。

        Key 名要和 templates/index.html 里的 Jinja 变量对得上:
            plot_borough    → Bar
            plot_building   → Pie
            plot_scatter    → Scatter
            plot_trend      → Line
            plot_price_dist → Histogram
            plot_pps_box    → Box
            plot_corr       → Heatmap
        """
        print("🎨 Generating 7 charts...")
        charts = {
            "plot_borough":    PlotService.make_borough_bar(df),
            "plot_building":   PlotService.make_building_pie(df),
            "plot_scatter":    PlotService.make_scatter(df),
            "plot_trend":      PlotService.make_trend_line(df),
            "plot_price_dist": PlotService.make_price_hist(df),
            "plot_pps_box":    PlotService.make_pps_box(df),
            "plot_corr":       PlotService.make_corr_heatmap(df),
        }
        print(f"✅ Generated {len(charts)} charts")
        return charts


# ============================================================
# 自测:python -m app.utils.plot_helper
# ============================================================
if __name__ == "__main__":
    from pathlib import Path
    from app.services.data_service import DataService

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "nyc_real_estate.csv"

    # 加载数据
    data_svc = DataService(csv_path)
    if not data_svc.load_data():
        print("数据加载失败")
        exit(1)

    # 生成所有图表
    charts = PlotService.generate_all_charts(data_svc.df)

    # 验证每个图都是合法 JSON
    import json
    print("\n========== 图表 JSON 验证 ==========")
    for name, json_str in charts.items():
        try:
            obj = json.loads(json_str)
            n_traces = len(obj.get("data", []))
            title = (
                obj.get("layout", {}).get("title", {}).get("text", "(no title)")
                if isinstance(obj.get("layout", {}).get("title"), dict)
                else obj.get("layout", {}).get("title", "(no title)")
            )
            print(f"  ✅ {name:20s} | traces={n_traces} | size={len(json_str):>7,} bytes")
        except json.JSONDecodeError as e:
            print(f"  ❌ {name:20s} | INVALID JSON: {e}")