"""
validators.py
=============
表单输入验证。

设计模式 (Result Pattern):
    所有 validator 函数返回 (cleaned_value_or_None, error_message_or_None)
    - 成功: (cleaned_dict, None)
    - 失败: (None, "human-readable error message")

为什么不用异常?
    用户输错不是"意外",是预期内事件。异常应该留给真正的意外
    (数据库断连、磁盘满之类)。Result Pattern 让控制流更清晰。

预测表单的 4 个字段(对应 KNN 模型的 4 个特征):
    GROSS SQUARE FEET   > 0
    YEAR BUILT          1800 ~ 2030
    TOTAL UNITS         1 ~ 1000
    BOROUGH             1 ~ 5
"""


# Borough 编码到名称的映射(和 DataService.BOROUGH_MAP 保持一致)
VALID_BOROUGHS = {1, 2, 3, 4, 5}


def validate_prediction_form(form):
    """
    验证 KNN 预测表单。

    参数:
        form: dict,通常来自 Flask request.form,所有 value 都是 str

    返回:
        (cleaned, error):
            cleaned: dict,4 个清洗后的字段(int/float 类型),验证失败时为 None
            error:   str,人类可读的错误消息,验证成功时为 None

    范围规则(和 DataService 清洗规则保持一致):
        gross_sqft  > 0
        year_built  1800 <= y <= 2030
        total_units 1 <= u <= 1000
        borough     1, 2, 3, 4 或 5
    """
    # ===== 第 1 层:类型转换 =====
    # 任何一个转不了 int/float,直接返回错误
    try:
        gross_sqft = float(form.get("gross_sqft", "").strip())
        year_built = int(form.get("year_built", "").strip())
        total_units = int(form.get("total_units", "").strip())
        borough = int(form.get("borough", "").strip())
    except (TypeError, ValueError):
        return None, "All inputs must be valid numbers."

    # ===== 第 2 层:业务范围检查 =====
    # 每个字段单独检查,出错就返回针对性的错误消息
    if gross_sqft <= 0:
        return None, "GROSS SQUARE FEET must be greater than 0."

    if not (1800 <= year_built <= 2030):
        return None, "YEAR BUILT must be between 1800 and 2030."

    if not (1 <= total_units <= 1000):
        return None, "TOTAL UNITS must be between 1 and 1000."

    if borough not in VALID_BOROUGHS:
        return None, "BOROUGH must be one of 1, 2, 3, 4, or 5."

    # ===== 全部通过 =====
    cleaned = {
        "gross_sqft": gross_sqft,
        "year_built": year_built,
        "total_units": total_units,
        "borough": borough,
    }
    return cleaned, None


# ============================================================
# 自测:python -m app.utils.validators
# ============================================================
if __name__ == "__main__":
    # 测试用例:(输入, 期望成功?, 描述)
    test_cases = [
        # ----- 应该成功 -----
        ({"gross_sqft": "1500", "year_built": "1998",
          "total_units": "2", "borough": "3"},
         True, "正常输入(Brooklyn)"),

        ({"gross_sqft": "1500.5", "year_built": "1998",
          "total_units": "2", "borough": "3"},
         True, "面积带小数"),

        # ----- 类型错误 -----
        ({"gross_sqft": "abc", "year_built": "1998",
          "total_units": "2", "borough": "3"},
         False, "面积非数字"),

        ({"gross_sqft": "", "year_built": "1998",
          "total_units": "2", "borough": "3"},
         False, "面积为空"),

        # ----- 范围错误 -----
        ({"gross_sqft": "0", "year_built": "1998",
          "total_units": "2", "borough": "3"},
         False, "面积为 0"),

        ({"gross_sqft": "-100", "year_built": "1998",
          "total_units": "2", "borough": "3"},
         False, "面积负数"),

        ({"gross_sqft": "1500", "year_built": "1700",
          "total_units": "2", "borough": "3"},
         False, "年份过早"),

        ({"gross_sqft": "1500", "year_built": "2050",
          "total_units": "2", "borough": "3"},
         False, "年份未来"),

        ({"gross_sqft": "1500", "year_built": "1998",
          "total_units": "0", "borough": "3"},
         False, "单位数为 0"),

        ({"gross_sqft": "1500", "year_built": "1998",
          "total_units": "9999", "borough": "3"},
         False, "单位数过大"),

        ({"gross_sqft": "1500", "year_built": "1998",
          "total_units": "2", "borough": "9"},
         False, "borough 非法"),

        ({"gross_sqft": "1500", "year_built": "1998",
          "total_units": "2", "borough": "0"},
         False, "borough 为 0"),
    ]

    print("========== Validator 测试 ==========")
    all_passed = True
    for form, should_succeed, description in test_cases:
        cleaned, error = validate_prediction_form(form)
        actual_success = cleaned is not None

        if actual_success == should_succeed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False

        if actual_success:
            detail = f"cleaned = {cleaned}"
        else:
            detail = f"error = '{error}'"

        print(f"  {status} | {description:20s} | {detail}")

    print("\n" + ("✅ All tests passed!" if all_passed else "❌ SOME TESTS FAILED"))