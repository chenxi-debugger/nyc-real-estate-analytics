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
    POST /predict   —— 处理表单提交,返回模型预测或验证错误
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

# 预测表单的初始状态 (空 form,无 error,无 result)
EMPTY_FORM = {
    "gross_sqft": "",
    "year_built": "",
    "total_units": "",
    "borough": "",
    "neighborhood": "",
    "building_class": "",
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
    return render_template("index.html", **charts)


# ============================================================
# 路由 2:价格预测 —— GET 显示表单 / POST 处理预测
# ============================================================
@dashboard_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET:  返回空白预测表单
    POST: 验证输入 → 调用冠军模型 → 返回预测结果(或错误)

    模板需要的变量:
        form:               dict,各字段当前值(出错时回填)
        error:              str 或 None
        result:             float 或 None,预测的 SALE PRICE
        borough_options:    borough 下拉选项
        neighborhoods_json: {borough: [小区,...]} 的 JSON,前端联动用
    """
    import json as _json

    form = dict(EMPTY_FORM)
    error = None
    result = None

    # 下拉框数据 (GET / POST 都要传给模板)
    neighborhoods = current_app.config.get("NEIGHBORHOODS", {})
    building_classes = current_app.config.get("BUILDING_CLASSES", [])

    if request.method == "POST":
        # ===== 第 1 步:把用户输入回写到 form =====
        for key in form:
            form[key] = request.form.get(key, "").strip()

        # ===== 第 2 步:验证表单 (带小区 + 建筑类型合法性校验) =====
        boro = form.get("borough", "")
        valid_nbhd = set(neighborhoods.get(boro, [])) if boro else None
        valid_bc = set(building_classes) if building_classes else None
        cleaned, error = validate_prediction_form(
            request.form, valid_nbhd, valid_bc
        )

        # ===== 第 3 步:验证通过 → 调冠军模型 =====
        if cleaned is not None:
            model_svc = current_app.config["MODEL_SERVICE"]
            try:
                result = model_svc.predict(**cleaned)
            except RuntimeError as exc:
                error = str(exc)
                result = None

    return render_template(
        "predict.html",
        form=form,
        error=error,
        result=result,
        borough_options=BOROUGH_OPTIONS,
        neighborhoods_json=_json.dumps(neighborhoods),
        building_classes=building_classes,
    )