"""
dashboard.py
============
Flask Blueprint:NYC 房产分析的 Web 路由。

设计模式 (Flask Blueprint):
    把路由独立到 Blueprint 而非直接挂在 app 上。好处:
    1. 路由按业务模块分组(以后可加 routes/api.py、routes/admin.py)
    2. 避免 app.py 越来越臃肿
    3. 便于多人协作 + 单元测试

资源共享 (current_app.config):
    DataService、ModelService、charts 都存在 app.config 里,
    路由用 current_app.config[KEY] 在请求处理时动态取。
    这避免了路由文件 import app 造成的循环依赖。

路由清单:
    GET  /          —— Dashboard,渲染 7 个图表
    GET  /predict   —— 显示空白预测表单
    POST /predict   —— 处理表单提交,返回 KNN 预测或验证错误
"""

from flask import Blueprint, current_app, render_template, request

from app.utils.validators import validate_prediction_form


# ============================================================
# 建 Blueprint
# ============================================================
# 第 1 个参数 "dashboard" 是 Blueprint 的名字(内部标识用)
# 第 2 个参数 __name__ 是 Python 惯例,Flask 用它定位资源
dashboard_bp = Blueprint("dashboard", __name__)


# ============================================================
# 模块级常量(不需要按请求重建)
# ============================================================
# 预测表单 borough 下拉选项 (value, label)
BOROUGH_OPTIONS = [
    (1, "Manhattan"),
    (2, "Bronx"),
    (3, "Brooklyn"),
    (4, "Queens"),
    (5, "Staten Island"),
]

# /predict 路由的初始表单状态 (空 form,无 error,无 result)
EMPTY_FORM = {
    "gross_sqft": "",
    "year_built": "",
    "total_units": "",
    "borough": "",
}


# ============================================================
# 路由 1:Dashboard 首页 —— 显示 7 个图表
# ============================================================
@dashboard_bp.route("/")
def index():
    """
    Dashboard 首页。从 app.config 取出预先生成好的 7 个图表 JSON,
    渲染到 templates/index.html。
    """
    charts = current_app.config["CHARTS"]

    # **charts 把 dict 展开成 plot_borough=..., plot_building=...
    # 这要求 charts 的 key 和 templates/index.html 里的 {{ plot_xxx }} 一致
    return render_template("index.html", **charts)


# ============================================================
# 路由 2:KNN 预测 —— GET 显示表单 / POST 处理预测
# ============================================================
@dashboard_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET:  返回空白预测表单
    POST: 验证输入 → 调用 KNN → 返回预测结果(或错误)

    模板需要 4 个变量:
        form:            dict,4 个字段的当前值(出错时回填到表单)
        error:           str 或 None,验证失败时的错误消息
        result:          float 或 None,预测的 SALE PRICE
        borough_options: borough 下拉选项
    """
    # 默认值:GET 请求或 POST 失败时的初始状态
    form = dict(EMPTY_FORM)   # 用 dict() 复制一份,避免修改全局常量
    error = None
    result = None

    if request.method == "POST":
        # ===== 第 1 步:把用户输入回写到 form =====
        # 出错时不清空输入框,让用户能修改而不是重输
        for key in form:
            form[key] = request.form.get(key, "").strip()

        # ===== 第 2 步:验证表单 =====
        cleaned, error = validate_prediction_form(request.form)

        # ===== 第 3 步:验证通过 → 调 KNN 模型 =====
        if cleaned is not None:
            model_svc = current_app.config["MODEL_SERVICE"]
            try:
                result = model_svc.predict(**cleaned)
                # **cleaned 把 dict 展开成关键字参数:
                #   gross_sqft=1500.0, year_built=1998, total_units=2, borough=3
            except RuntimeError as exc:
                # ModelService.predict() 已经把异常包装成 RuntimeError
                error = str(exc)
                result = None

    # GET 和 POST 都渲染同一个模板,只是 form/error/result 内容不同
    return render_template(
        "predict.html",
        form=form,
        error=error,
        result=result,
        borough_options=BOROUGH_OPTIONS,
    )