"""
train_and_evaluate.py
=====================
离线训练脚本 —— 项目唯一的"训练 + 选优"入口。

职责 (单一):
    1. 加载 + 清洗数据 (复用 DataService)
    2. 特征工程 (复用 ModelService 的方法,避免重复造轮子)
    3. 每个候选模型带自己的超参数网格,循环里统一用 GridSearchCV 调参
    4. 用 MLflow 追踪每个模型的参数与指标
    5. 按交叉验证 R² 选出冠军
    6. 在测试集上做最终评估
    7. 生成 4 张评估图
    8. 保存冠军模型 + 元数据 (供 Flask / ModelService 加载)

设计要点:
    - 训练逻辑【只在这一个文件里】。model_service.py 不再自己训练,
      它运行时只加载这里产出的模型;若缓存不存在,它会回调本文件的
      train_and_select() 函数。这样训练代码零重复。
    - 把训练封装成 train_and_select() 函数,既能被命令行直接运行
      (python train_and_evaluate.py),也能被 model_service 安全 import。

用法:
    python train_and_evaluate.py
产出:
    models/champion_pipeline.pkl       冠军模型 (供 Flask 加载)
    static/eval/*.png                  4 张评估图
    mlruns/                            MLflow 实验记录
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")  # 无界面后端,服务器上也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from app.services.data_service import DataService
from app.services.model_service import ModelService

# MLflow 是可选依赖:装了就追踪实验,没装也能正常训练
try:
    import mlflow
    import mlflow.sklearn
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

# 紫色/薰衣草配色 (跟 dashboard 风格一致)
PURPLE = "#7c5cbf"
LAVENDER = "#b9a3e3"
ACCENT_BLUE = "#4a7fd0"
GRID = "#e6e1f0"

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "nyc_real_estate.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "champion_pipeline.pkl"
EVAL_DIR = PROJECT_ROOT / "static" / "eval"
CV_FOLDS = 5
RANDOM_STATE = 42

# ============================================================
# 候选模型 + 各自的超参数网格
#   —— 每个模型带自己的 params,
#      循环里统一 GridSearchCV。这样每个模型都用各自最优参数公平比较。
#   —— 注意键名前缀 "model__":因为模型被包在 Pipeline 的 "model" 步骤里,
#      Pipeline 要求用 "步骤名__参数名" 来定位参数。
# ============================================================
def _build_model_configs(svc):
    """返回 {模型名: {"pipeline": Pipeline, "params": 网格}}。"""
    return {
        "LinearRegression": {
            "pipeline": Pipeline([
                ("pre", svc._build_linear_pre()),
                ("model", LinearRegression()),
            ]),
            "params": {},  # 线性回归无需调参,空网格 = 只跑一次默认
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("pre", svc._fresh_tree_pre()),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            "params": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [12, 20, None],
            },
        },
        "GBDT": {
            "pipeline": Pipeline([
                ("pre", svc._fresh_tree_pre()),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]),
            "params": {
                "model__n_estimators": [200, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        },
        "LightGBM": {
            "pipeline": Pipeline([
                ("pre", svc._fresh_tree_pre()),
                ("model", LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
            ]),
            "params": {
                "model__n_estimators": [300, 400],
                "model__learning_rate": [0.03, 0.05],
                "model__num_leaves": [31, 63],
                "model__reg_lambda": [0.0, 1.0],   # L2 正则,防过拟合
            },
        },
    }

# ============================================================
# 核心:训练 + 选优 (可被命令行运行,也可被 model_service import)
# ============================================================
def train_and_select(df, save=True, make_plots=True, verbose=True):
    """
    训练所有候选模型、调参、选冠军、(可选)出图并保存。

    参数:
        df:         DataService 清洗后的原始 DataFrame
        save:       是否把冠军模型存到 MODEL_PATH
        make_plots: 是否生成 4 张评估图
        verbose:    是否打印进度

    返回:
        bundle: dict —— 与 ModelService 期望的格式一致:
            {"pipeline", "champion", "metrics", "leaderboard"}
    """
    svc = ModelService(MODEL_PATH)

    # ---------- 1. 特征工程 (复用 ModelService 方法,零重复) ----------
    df_feat = svc._engineer_features(svc._trim_outliers(df))
    X = svc._coerce_categoricals(df_feat[svc.FEATURES].copy())
    y = np.log10(df_feat[svc.TARGET].values)   # log 目标:房价长尾
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    if verbose:
        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    if _HAS_MLFLOW:
        mlflow.set_experiment("nyc_real_estate_price")

    # ---------- 2. 每个模型带网格,循环统一 GridSearch ----------
    model_configs = _build_model_configs(svc)
    leaderboard = []
    best = {"name": None, "cv_r2": -np.inf, "pipeline": None, "params": None}

    for name, config in model_configs.items():
        if verbose:
            print(f"\nTraining {name} ...")

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        cv_r2 = grid.best_score_           # 交叉验证最优分 (选模型用这个)
        best_estimator = grid.best_estimator_
        best_params = grid.best_params_

        leaderboard.append({
            "model": name,
            "cv_r2_mean": float(cv_r2),
            "cv_r2_std": float(grid.cv_results_["std_test_score"][grid.best_index_]),
        })  # 用冠军编号,取出冠军那一个的标准差

        if verbose:
            print(f"  Best params: {best_params}")
            print(f"  CV R2 = {cv_r2:.4f}")

        # ---- MLflow 追踪 (装了才记录) ----
        if _HAS_MLFLOW:
            with mlflow.start_run(run_name=f"tune_{name}"):
                mlflow.log_params(best_params if best_params else {"default": True})
                mlflow.log_metric("cv_r2", cv_r2)
                # log_model 在某些环境 (skops 校验) 会对 LightGBM 报错,
                # 包一层保护:追踪到参数/指标即可,模型本身由 joblib 另存
                try:
                    mlflow.sklearn.log_model(best_estimator, name)
                except Exception as e:
                    if verbose:
                        print(f"  (mlflow log_model skipped: {type(e).__name__})")

        # ---- 记录冠军 (按交叉验证 R²) ----
        if cv_r2 > best["cv_r2"]:
            best.update(name=name, cv_r2=cv_r2,
                        pipeline=best_estimator, params=best_params)

    leaderboard.sort(key=lambda d: d["cv_r2_mean"], reverse=True)
    if verbose:
        print(f"\nBest model: {best['name']} (CV R2={best['cv_r2']:.4f})")

    # ---------- 3. 冠军在测试集上最终评估 ----------
    champion = best["pipeline"]
    y_pred = np.power(10, champion.predict(X_test))   # 还原成真实美元
    y_true = np.power(10, y_test)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if verbose:
        print(f"Test: R2={r2:.3f} MAE=${mae:,.0f} RMSE=${rmse:,.0f} MAPE={mape:.1f}%")

    metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "r2": float(r2), "mae": float(mae),
        "rmse": float(rmse), "mape": float(mape),
    }
    bundle = {
        "pipeline": champion,
        "champion": f"{best['name']} (tuned)",
        "metrics": metrics,
        "leaderboard": leaderboard,
    }

    # ---------- 4. 保存 ----------
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        if verbose:
            print(f"Saved -> {MODEL_PATH}")

    # ---------- 5. 出图 ----------
    if make_plots:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        _plot_leaderboard(leaderboard, EVAL_DIR / "leaderboard.png")
        _plot_pred_vs_actual(y_true, y_pred, r2, EVAL_DIR / "pred_vs_actual.png")
        _plot_residuals(y_true, y_pred, EVAL_DIR / "residuals.png")
        _plot_importance(champion, svc, EVAL_DIR / "feature_importance.png")
        if verbose:
            print(f"Plots saved to {EVAL_DIR}/")

    return bundle


def main():
    """命令行入口:加载数据 → 训练选优 → 保存 + 出图。"""
    data_svc = DataService(CSV_PATH)
    if not data_svc.load_data():
        raise SystemExit("数据加载失败")
    train_and_select(data_svc.df, save=True, make_plots=True, verbose=True)
    print("\nDone.")

# ============================================================
# 画图函数 (原样保留)
# ============================================================
def _plot_leaderboard(leaderboard, path):
    names = [d["model"] for d in leaderboard]
    means = [d["cv_r2_mean"] for d in leaderboard]
    stds = [d["cv_r2_std"] for d in leaderboard]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names[::-1], means[::-1], xerr=stds[::-1],
                   color=LAVENDER, edgecolor=PURPLE, capsize=4)
    bars[-1].set_color(PURPLE)  # 冠军高亮
    ax.set_xlabel("Cross-validated R² (higher is better)")
    ax.set_title("Model Comparison — 5-fold CV", color=PURPLE, fontweight="bold")
    for i, m in enumerate(means[::-1]):
        ax.text(m + 0.005, i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(means) * 1.18)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pred_vs_actual(y_true, y_pred, r2, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.25, color=PURPLE, edgecolors="none")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "--", color=ACCENT_BLUE, lw=1.5, label="Perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title(f"Predicted vs Actual  (R²={r2:.3f})",
                 color=PURPLE, fontweight="bold")
    ax.legend(); ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_residuals(y_true, y_pred, path):
    resid = np.log10(y_pred) - np.log10(y_true)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, resid, s=8, alpha=0.25, color=PURPLE, edgecolors="none")
    ax.axhline(0, color=ACCENT_BLUE, ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Predicted Sale Price ($)")
    ax.set_ylabel("Residual  (log10 pred − log10 actual)")
    ax.set_title("Residual Plot", color=PURPLE, fontweight="bold")
    ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_importance(best_model, svc, path):
    model = best_model.named_steps["model"]
    pre = best_model.named_steps["pre"]
    if not hasattr(model, "feature_importances_"):
        return  # 冠军不是树模型 (理论上不会发生),跳过此图
    feat_names = list(svc.NUMERIC_FEATURES)
    ohe = pre.named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(svc.CATEGORICAL_FEATURES)
    feat_names += list(ohe_names)

    importances = model.feature_importances_
    agg = {}
    for fname, imp in zip(feat_names, importances):
        base = fname
        for cat in svc.CATEGORICAL_FEATURES:
            if fname.startswith(cat):
                base = cat
                break
        agg[base] = agg.get(base, 0) + imp
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels[::-1], vals[::-1], color=LAVENDER, edgecolor=PURPLE)
    ax.set_xlabel("LightGBM Feature Importance (gain-based, aggregated)")
    ax.set_title("What Drives the Prediction", color=PURPLE, fontweight="bold")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()