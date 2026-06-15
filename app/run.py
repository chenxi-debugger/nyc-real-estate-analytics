"""
run.py
======
Flask 应用入口 (Application Factory Pattern)。

启动流程:
    1. create_app() 构造 Flask app
    2. 加载 + 清洗 NYC 房产数据         (DataService)
    3. 加载或训练 KNN 模型              (ModelService)
    4. 生成 7 个 Plotly 图表的 JSON    (PlotService)
    5. 把 ModelService、charts 存进 app.config
    6. 注册 Blueprint (routes/dashboard.py)
    7. 返回 app 给调用方

为什么用工厂模式 (create_app)?
    - 测试友好: 测试时可以用 create_app() 建独立实例
    - 可配置: 未来可以接受参数 create_app(config='testing')
    - 是 Flask 官方推荐的工业级架构

运行方式:
    python -m app.run
    → 启动 dev server,访问 http://127.0.0.1:5001
"""

from pathlib import Path

from flask import Flask

from app.routes.dashboard import dashboard_bp
from app.services.data_service import DataService
from app.services.model_service import ModelService
from app.utils.plot_helper import PlotService


# ============================================================
# 路径常量 (在 create_app 外定义,启动一次即可)
# ============================================================
# __file__              = .../app/run.py
# .resolve()            = 绝对路径
# .parents[1]           = .../app/ 的父目录,即【项目根】
BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "nyc_real_estate.csv"
MODEL_PATH = BASE_DIR / "models" / "knn_pipeline.pkl"
TEMPLATE_DIR = BASE_DIR / "templates"


# ============================================================
# Application Factory
# ============================================================
def create_app():
    """
    构造 Flask app 并完成所有初始化。

    返回:
        flask.Flask 实例,已加载数据 + 模型 + 图表,Blueprint 已注册。
    """
    print("=" * 60)
    print("🚀 Starting NYC Real Estate Analytics")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. 构造 Flask app (显式指定 templates/ 路径)
    # ------------------------------------------------------------
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
    )

    # ------------------------------------------------------------
    # 2. 加载 + 清洗数据
    # ------------------------------------------------------------
    print("\n[1/4] Loading data...")
    data_svc = DataService(CSV_PATH)
    if not data_svc.load_data():
        raise RuntimeError(
            f"Failed to load data from {CSV_PATH}. "
            "Make sure the CSV file exists."
        )

    # ------------------------------------------------------------
    # 3. 加载或训练 KNN 模型
    # ------------------------------------------------------------
    print("\n[2/4] Preparing KNN model...")
    model_svc = ModelService(MODEL_PATH)
    model_svc.fit_or_load(data_svc.df)

    # ------------------------------------------------------------
    # 4. 生成 7 个图表的 JSON (启动时算一次,后续请求直接用)
    # ------------------------------------------------------------
    print("\n[3/4] Generating charts...")
    charts = PlotService.generate_all_charts(data_svc.df)

    # ------------------------------------------------------------
    # 5. 把 services 和 charts 存进 app.config,供路由通过 current_app 取
    # ------------------------------------------------------------
    print("\n[4/4] Registering services + routes...")
    app.config["DATA_SERVICE"] = data_svc
    app.config["MODEL_SERVICE"] = model_svc
    app.config["CHARTS"] = charts

    # ------------------------------------------------------------
    # 6. 注册 Blueprint
    # ------------------------------------------------------------
    app.register_blueprint(dashboard_bp)

    print("\n" + "=" * 60)
    print("✅ Application ready")
    print("=" * 60)

    return app


# ============================================================
# 启动 server (只在 python -m app.run 直接运行时执行)
# ============================================================
if __name__ == "__main__":
    app = create_app()

    print("\n🌐 Server: http://127.0.0.1:5001")
    print("   Dashboard:   http://127.0.0.1:5001/")
    print("   Predictor:   http://127.0.0.1:5001/predict")
    print("\n(Press Ctrl+C to stop)\n")

    # debug=False 因为:
    #   1. 我们的应用启动会训练/加载模型,reloader 会触发两次启动,慢且乱
    #   2. 这是给老师/招聘官 demo 用,不是日常开发
    app.run(host="127.0.0.1", port=5001, debug=False)