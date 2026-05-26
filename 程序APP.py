import os
os.environ["OMP_NUM_THREADS"] = "1"

import traceback
import numpy as np

# ========================
# 兼容性补丁
# ========================
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import streamlit as st
import joblib
import pandas as pd
import shap
import plotly.graph_objects as go
import textwrap


# ========================
# 页面基础设置
# ========================
CONTAINER_W = 980

st.set_page_config(
    page_title="Non-Invasive PCOS Risk Prediction",
    layout="centered"
)

st.markdown(f"""
<style>
.main .block-container {{
  max-width: {CONTAINER_W}px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

div[data-testid="stPlotlyChart"] {{
  padding: 12px 10px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}

.badge {{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  font-size:14px;
  font-weight:600;
  color:#fff;
  margin-left:10px;
  vertical-align:middle;
}}

.stButton > button {{
  width: 100%;
  border-radius: 10px;
  font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


# ========================
# 路径设置
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = "Stacking_RF_LightGBM.pkl"

# 局部解释图只展示前 N 个贡献最大的变量，其余变量合并为一项
TOP_N_CONTRIBUTIONS = 6

BACKGROUND_CANDIDATES = [
    "shap_background.csv",
    "shap_background.xlsx",
    "shap_background.xls",
    "shap_background(1).csv"
]


# ========================
# 分类变量映射
# 编码必须和训练数据一致
# ========================
OPTION_MAP = {
    "Age at menarche": {
        "<11 years": 1,
        "11–15 years": 2,
        ">15 years": 3
    },

    "Menstrual cycle regularity": {
        "<21 days": 1,
        "<21 days, occasionally >35 days; >35 days, occasionally <21 days": 2,
        "21–35 days": 3,
        ">35 days": 4
    },

    "Hair loss": {
        "No": 0,
        "Yes": 1
    }
}

VALUE_TO_TEXT = {
    feat: {v: k for k, v in mapping.items()}
    for feat, mapping in OPTION_MAP.items()
}


# ========================
# 变量显示名和默认值
# ========================
FEATURE_META = {
    "Age": {
        "label": "Age, years",
        "type": "numerical",
        "default": 29.50,
        "step": 0.01
    },

    "Neck circumference": {
        "label": "Neck circumference, cm",
        "type": "numerical",
        "default": 31.38,
        "step": 0.01
    },

    "Waist circumference": {
        "label": "Waist circumference, cm",
        "type": "numerical",
        "default": 74.72,
        "step": 0.01
    },

    "Hip circumference": {
        "label": "Hip circumference, cm",
        "type": "numerical",
        "default": 94.29,
        "step": 0.01
    },

    "Systolic blood pressure": {
        "label": "Systolic blood pressure, mmHg",
        "type": "numerical",
        "default": 111.14,
        "step": 0.01
    },

    "Diastolic blood pressure": {
        "label": "Diastolic blood pressure, mmHg",
        "type": "numerical",
        "default": 71.98,
        "step": 0.01
    },

    "Skeletal muscle mass": {
        "label": "Skeletal muscle mass, kg",
        "type": "numerical",
        "default": 21.43,
        "step": 0.01
    },

    "Percent body fat": {
        "label": "Percent body fat, %",
        "type": "numerical",
        "default": 35.09,
        "step": 0.01
    },

    "Body mass index": {
        "label": "Body mass index, kg/m²",
        "type": "numerical",
        "default": 23.54,
        "step": 0.01
    },

    "Age at menarche": {
        "label": "Age at menarche",
        "type": "categorical",
        "default": "11–15 years"
    },

    "Menstrual cycle regularity": {
        "label": "Menstrual cycle regularity",
        "type": "categorical",
        "default": "21–35 days"
    },

    "Hirsutism score": {
        "label": "Hirsutism score",
        "type": "numerical",
        "default": 3.15,
        "step": 1.00
    },

    "Hair loss": {
        "label": "Hair loss",
        "type": "categorical",
        "default": "No"
    },

    "Acne score": {
        "label": "Acne score",
        "type": "numerical",
        "default": 3.32,
        "step": 1.00
    }
}


# ========================
# 模型加载工具函数
# ========================
def load_joblib_with_clear_error(path, model_label):
    try:
        return joblib.load(path)

    except ModuleNotFoundError as e:
        st.error(f"加载模型失败：{model_label}")
        st.code(
            f"Model file: {os.path.basename(str(path))}\n"
            f"Missing module: {e.name}\n"
            f"Error: {repr(e)}"
        )
        st.stop()

    except Exception:
        st.error(f"加载模型失败：{model_label}")
        st.code(traceback.format_exc())
        st.stop()


def extract_predictor(obj):
    if hasattr(obj, "predict_proba"):
        return obj

    if isinstance(obj, dict):
        preferred_keys = [
            "final_model",
            "final_pipeline",
            "model",
            "best_model",
            "estimator",
            "classifier",
            "clf"
        ]

        for key in preferred_keys:
            if key in obj and hasattr(obj[key], "predict_proba"):
                return obj[key]

        for value in obj.values():
            if hasattr(value, "predict_proba"):
                return value

    raise ValueError("未能从 pkl 文件中提取支持 predict_proba 的模型。")


def resolve_model_path(path_like):
    path_like = str(path_like)

    if os.path.isabs(path_like) and os.path.exists(path_like):
        return path_like

    candidate_1 = os.path.join(BASE_DIR, path_like)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(BASE_DIR, os.path.basename(path_like))
    if os.path.exists(candidate_2):
        return candidate_2

    raise FileNotFoundError(f"未找到基模型文件: {path_like}")


@st.cache_resource
def load_model_bundle(model_file):
    model_path = os.path.join(BASE_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到集成模型文件: {model_path}")

    obj = load_joblib_with_clear_error(model_path, model_file)

    if hasattr(obj, "predict_proba"):
        feature_cols = getattr(obj, "feature_names_in_", None)

        if feature_cols is None:
            raise ValueError("单模型中没有 feature_names_in_，请手动提供 MODEL_FEATURES。")

        return {
            "mode": "single",
            "model_name": os.path.splitext(model_file)[0],
            "model": obj,
            "feature_cols": list(feature_cols),
            "threshold": 0.5
        }

    if not isinstance(obj, dict):
        raise ValueError("模型既不是单模型，也不是集成模型字典。")

    members = obj.get("members", None)
    feature_cols = obj.get("feature_cols", None)
    ensemble_type = obj.get("ensemble_type", "")
    model_name = obj.get("model_name", os.path.splitext(model_file)[0])
    threshold = float(obj.get("threshold", 0.5))

    if members is None:
        raise ValueError("集成模型字典中缺少 members。")

    if feature_cols is None:
        raise ValueError("集成模型字典中缺少 feature_cols。")

    base_model_files = obj.get("base_model_files", {})
    base_models = {}

    for member in members:
        if member not in base_model_files:
            raise ValueError(f"base_model_files 中缺少成员模型 {member} 的文件路径。")

        base_path = resolve_model_path(base_model_files[member])
        base_obj = load_joblib_with_clear_error(base_path, f"base model: {member}")
        base_models[member] = extract_predictor(base_obj)

    if "Stacking" in str(ensemble_type) or str(model_name).startswith("Stacking"):
        meta_model = obj.get("meta_model", None)

        if meta_model is None:
            raise ValueError("Stacking 模型字典中缺少 meta_model。")

        return {
            "mode": "stacking",
            "model_name": model_name,
            "members": members,
            "base_models": base_models,
            "meta_model": meta_model,
            "feature_cols": list(feature_cols),
            "threshold": threshold
        }

    if "Weighted" in str(ensemble_type) or str(model_name).startswith("WeightedVoting"):
        weights = obj.get("weights", None)

        if weights is None:
            raise ValueError("WeightedVoting 模型字典中缺少 weights。")

        return {
            "mode": "weighted",
            "model_name": model_name,
            "members": members,
            "base_models": base_models,
            "weights": weights,
            "feature_cols": list(feature_cols),
            "threshold": threshold
        }

    raise ValueError(f"暂不支持的集成模型类型: {ensemble_type}")


# ========================
# 背景数据读取
# ========================
def find_background_path():
    for fname in BACKGROUND_CANDIDATES:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            return path
    return None


def read_table(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"不支持的背景数据格式: {ext}")


# ========================
# 加载模型
# ========================
bundle = load_model_bundle(MODEL_FILE)

MODEL_FEATURES = bundle["feature_cols"]
FIXED_THRESHOLD = bundle["threshold"]

CATEGORICAL_COLS = [c for c in MODEL_FEATURES if c in OPTION_MAP]
NUMERICAL_COLS = [c for c in MODEL_FEATURES if c not in CATEGORICAL_COLS]


# ========================
# 输入整理
# ========================
def prepare_input_df(data_like):
    if isinstance(data_like, pd.DataFrame):
        df = data_like.copy()
    else:
        arr = np.asarray(data_like)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        df = pd.DataFrame(arr, columns=MODEL_FEATURES)

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少变量: {missing}")

    df = df[MODEL_FEATURES].copy()

    for col in CATEGORICAL_COLS:
        df[col] = np.rint(pd.to_numeric(df[col], errors="raise")).astype(int)

    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def predict_positive_proba(data_like):
    df = prepare_input_df(data_like)

    if bundle["mode"] == "single":
        return bundle["model"].predict_proba(df)[:, 1]

    base_probs = []

    for member in bundle["members"]:
        p = bundle["base_models"][member].predict_proba(df)[:, 1]
        base_probs.append(p)

    base_probs = np.column_stack(base_probs)

    if bundle["mode"] == "stacking":
        return bundle["meta_model"].predict_proba(base_probs)[:, 1]

    if bundle["mode"] == "weighted":
        weights = np.array([float(bundle["weights"][m]) for m in bundle["members"]])
        weights = weights / weights.sum()
        return np.dot(base_probs, weights)

    raise ValueError("未知模型模式。")


# ========================
# 加载 SHAP 背景数据
# ========================
@st.cache_data
def load_background_data():
    bg_path = find_background_path()

    if bg_path is None:
        return None

    bg = read_table(bg_path)

    missing = [c for c in MODEL_FEATURES if c not in bg.columns]
    if missing:
        raise ValueError(f"SHAP 背景数据缺少以下列: {missing}")

    bg = prepare_input_df(bg[MODEL_FEATURES])

    if len(bg) > 100:
        bg = bg.sample(n=100, random_state=42)

    return bg


background_df = load_background_data()


# ========================
# 标签与格式
# ========================
def classify_prediction(p: float, threshold: float):
    if p >= threshold:
        return "Higher predicted likelihood", "#C62828"
    return "Lower predicted likelihood", "#2E7D32"


def get_display_name(feature):
    return FEATURE_META.get(feature, {}).get("label", feature)


def format_feature_value(feature, value):
    if feature in VALUE_TO_TEXT:
        try:
            value_int = int(round(float(value)))
        except Exception:
            return str(value)

        return VALUE_TO_TEXT[feature].get(value_int, str(value_int))

    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def format_contribution(x):
    if abs(x) < 0.005:
        return "0.00"
    return f"{x:+.2f}"


# ========================
# SHAP explainer
# ========================
def get_shap_explainer():
    if background_df is None:
        return None

    if "kernel_shap_explainer" not in st.session_state:
        st.session_state["kernel_shap_explainer"] = shap.KernelExplainer(
            predict_positive_proba,
            background_df,
            link="identity"
        )

    return st.session_state["kernel_shap_explainer"]


def compute_kernel_shap_probability(X_one_row):
    explainer = get_shap_explainer()

    if explainer is None:
        return None, None

    shap_values = explainer.shap_values(X_one_row, nsamples=200)

    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[0])
    else:
        shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        local_vals = shap_values[0]
    else:
        local_vals = shap_values.reshape(-1)

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]

    return local_vals, float(base_value)


# ========================
# 局部贡献图
# ========================
def summarize_contributions_for_plot(df_sorted, top_n=TOP_N_CONTRIBUTIONS):
    """
    只在图中展示贡献绝对值最大的 top_n 个变量，其余变量合并为 Other features。
    完整贡献仍保留在下方 expander 表格中。
    """
    df_sorted = df_sorted.copy().reset_index(drop=True)

    if len(df_sorted) <= top_n:
        return df_sorted

    top_df = df_sorted.iloc[:top_n].copy()
    other_df = df_sorted.iloc[top_n:].copy()

    other_contribution = float(other_df["dpp"].sum())

    other_row = pd.DataFrame({
        "feature": ["Other features"],
        "value": [np.nan],
        "value_text": ["combined"],
        "dpp": [other_contribution]
    })

    return pd.concat([top_df, other_row], ignore_index=True)


def plot_pp_bar(df_plot):
    df_plot = df_plot.copy()

    labels = []
    for f, vtxt in zip(df_plot["feature"], df_plot["value_text"]):
        if f == "Other features":
            label = "Other features (combined)"
        else:
            label = f"{get_display_name(f)} = {vtxt}"
        labels.append(textwrap.fill(label, width=30))

    x_vals = df_plot["dpp"].to_numpy()
    colors = np.where(x_vals >= 0, "#E45756", "#4C78A8")

    texts, textpos, textcolor = [], [], []

    for x in x_vals:
        texts.append(format_contribution(x))

        if x < 0:
            textpos.append("inside")
            textcolor.append("white")
        else:
            if abs(x) >= 1:
                textpos.append("inside")
                textcolor.append("white")
            else:
                textpos.append("outside")
                textcolor.append("black")

    fig = go.Figure(go.Bar(
        y=labels[::-1],
        x=x_vals[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=texts[::-1],
        texttemplate="%{text}",
        textposition=textpos[::-1],
        insidetextanchor="end",
        textfont=dict(color=textcolor[::-1], size=14),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.2f} percentage points<extra></extra>",
    ))

    fig.update_layout(
        height=max(260, 38 * len(labels) + 80),
        margin=dict(l=300, r=80, t=12, b=12),
        font=dict(family="Times New Roman", size=17),
        yaxis=dict(
            title="",
            type="category",
            tickfont=dict(size=15),
            automargin=True
        ),
        xaxis=dict(
            title="Approximate contribution to model-predicted probability (percentage points)",
            zeroline=True,
            zerolinewidth=1.2,
            zerolinecolor="#B0BEC5",
            showgrid=True,
            gridcolor="#EFEFEF",
            automargin=True
        ),
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        uniformtext_minsize=12,
        uniformtext_mode="hide"
    )

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="#B0BEC5",
        line_width=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


# ========================
# 页面标题
# ========================
st.title("Non-Invasive PCOS Risk Prediction")
st.caption("Estimate individualized PCOS risk using non-invasive clinical features.")

# ========================
# 构建表单顺序
# ========================
PREFERRED_ORDER = [
    "Age at menarche",
    "Menstrual cycle regularity",
    "Hair loss",
    "Age",
    "Body mass index",
    "Percent body fat",
    "Skeletal muscle mass",
    "Neck circumference",
    "Waist circumference",
    "Hip circumference",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Acne score",
    "Hirsutism score"
]

FORM_FEATURES = [f for f in PREFERRED_ORDER if f in MODEL_FEATURES]
FORM_FEATURES += [f for f in MODEL_FEATURES if f not in FORM_FEATURES]


# ========================
# 初始化默认值
# ========================
for feature in FORM_FEATURES:
    meta = FEATURE_META.get(feature, {
        "label": feature,
        "type": "numerical",
        "default": 0.00,
        "step": 0.01
    })

    key = f"{feature}_input"

    if key not in st.session_state:
        st.session_state[key] = meta["default"]


# ========================
# 输入表单
# ========================
with st.form("prediction_form", clear_on_submit=False):

    col_left, col_right = st.columns(2, gap="medium")

    for i, feature in enumerate(FORM_FEATURES):
        meta = FEATURE_META.get(feature, {
            "label": feature,
            "type": "numerical",
            "default": 0.00,
            "step": 0.01
        })

        label = meta["label"]
        ftype = meta["type"]
        key = f"{feature}_input"
        target_col = col_left if i % 2 == 0 else col_right

        with target_col:
            if ftype == "categorical":
                options = list(OPTION_MAP[feature].keys())

                current_value = st.session_state[key]

                if current_value not in options:
                    current_value = options[0]
                    st.session_state[key] = current_value

                st.selectbox(
                    label,
                    options=options,
                    index=options.index(current_value),
                    key=key
                )

            else:
                st.number_input(
                    label,
                    value=float(st.session_state[key]),
                    step=float(meta.get("step", 0.01)),
                    format="%.2f",
                    key=key
                )

    submitted = st.form_submit_button("Predict", type="primary")


# ========================
# 预测与解释
# ========================
if submitted:
    form_values = {}

    for feature in FORM_FEATURES:
        meta = FEATURE_META.get(feature, {
            "label": feature,
            "type": "numerical",
            "default": 0.00,
            "step": 0.01
        })

        key = f"{feature}_input"

        if meta["type"] == "categorical":
            form_values[feature] = int(OPTION_MAP[feature][st.session_state[key]])
        else:
            form_values[feature] = float(st.session_state[key])

    X = pd.DataFrame(
        [[form_values[col] for col in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    X = prepare_input_df(X)

    p1 = float(predict_positive_proba(X)[0])
    pred_label, pred_color = classify_prediction(p1, FIXED_THRESHOLD)

    st.markdown(
        f"""
        <div style='font-family:Times New Roman; font-size:20px;'>
          <b>Model-predicted probability of PCOS: {p1 * 100:.2f}%.</b>
          <span class="badge" style="background:{pred_color};">{pred_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if background_df is None:
        st.info(
            "Prediction is available. To display local feature contributions, "
            "please add shap_background.csv, shap_background.xlsx, or shap_background.xls "
            "to the same folder as this app."
        )

    else:
        with st.spinner("Calculating local feature contributions..."):
            shap_vals, _ = compute_kernel_shap_probability(X)

        if shap_vals is None:
            st.info("Prediction is available, but local feature contributions could not be calculated.")

        else:
            feat_vals = X.iloc[0].to_numpy()
            dpp = shap_vals * 100.0

            order = np.argsort(-np.abs(dpp), kind="mergesort")

            ordered_features = X.columns.to_numpy()[order]
            ordered_values = feat_vals[order]

            df_sorted = pd.DataFrame({
                "feature": ordered_features,
                "value": ordered_values,
                "value_text": [
                    format_feature_value(f, v)
                    for f, v in zip(ordered_features, ordered_values)
                ],
                "dpp": dpp[order],
            })

            df_for_plot = summarize_contributions_for_plot(df_sorted, top_n=TOP_N_CONTRIBUTIONS)
            plot_pp_bar(df_for_plot)

            with st.expander("Show full contributions"):
                table = df_sorted[["feature", "value_text", "dpp"]].copy()
                table["feature"] = table["feature"].map(get_display_name)
                table.columns = [
                    "Feature",
                    "Value",
                    "Contribution (percentage points)"
                ]
                table.insert(0, "Rank", np.arange(1, len(table) + 1))

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True
                )