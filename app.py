import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "pca_failure_model.joblib"
RESULTS_PATH = BASE_DIR / "artifacts" / "model_results.json"

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SmartPCA | Engineering Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f8ff;
    color: #17365d;
}

h1, h2, h3 {
    color: #17365d;
}

[data-testid="stMetric"] {
    background-color: #e3efff;
    border: 1px solid #c4dafa;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #17365d !important;
}

[data-testid="stSidebar"] {
    background-color: #e8f1ff;
}

.stButton button {
    border-radius: 8px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #d5e3f5;
    border-radius: 10px;
}

/* Keep text readable even if the computer is in dark mode */
[data-testid="stHeader"] {
    background-color: #f4f8ff;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h3,
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #17365d;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS (work on old and new Streamlit versions) ----------------
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])


def show_chart(fig):
    # readable chart text in light and dark mode
    ink = dict(color="#17365d")
    fig.update_layout(font_color="#17365d", legend_font=ink, legend_title_font=ink, title_font=ink)
    fig.update_xaxes(tickfont=ink, title_font=ink)
    fig.update_yaxes(tickfont=ink, title_font=ink)
    if _ST_VERSION >= (1, 50):
        st.plotly_chart(fig)
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(data, **kwargs):
    if _ST_VERSION >= (1, 50):
        st.dataframe(data, **kwargs)
    else:
        st.dataframe(data, use_container_width=True, **kwargs)


def add_engineered_features(data):
    """Same feature engineering as in the Jupyter Notebook."""
    data = data.copy()
    data["Temperature difference [K]"] = data["Process temperature [K]"] - data["Air temperature [K]"]
    data["Power [W]"] = data["Torque [Nm]"] * data["Rotational speed [rpm]"] * 2 * np.pi / 60
    data["Strain [min*Nm]"] = data["Tool wear [min]"] * data["Torque [Nm]"]
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("📊 SmartPCA")
st.sidebar.caption("Engineering Data Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Project Overview",
        "📈 PCA Analysis",
        "🔍 Data Explorer",
        "🤖 Predict Failure",
        "🎯 Model Accuracy"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### About PCA")
st.sidebar.write(
    "Principal Component Analysis reduces the number "
    "of features while retaining as much data variation "
    "as possible."
)

results_sidebar = load_results()
if results_sidebar:
    improved = results_sidebar["comparison"][-1]
    st.sidebar.markdown("### Trained model")
    st.sidebar.write(
        f"SVM on {results_sidebar['n_components']} principal components  \n"
        f"Test accuracy: **{improved['Accuracy (%)']:.2f} %**"
    )

st.sidebar.caption("Machine Learning Engineering Project")

# ---------------- MAIN HEADER ----------------
st.title("SmartPCA Dashboard")
st.write(
    "An interactive engineering application for "
    "dimensionality reduction using Principal Component Analysis "
    "and machine-failure prediction."
)

st.divider()


# =====================================================================
# PAGE 4 – PREDICT FAILURE  (uses the model trained in the notebook)
# =====================================================================
EXAMPLES = {
    "Normal machine": ("L", 303.7, 312.4, 1513, 40.1, 135),
    "Heat dissipation failure": ("L", 302.6, 310.4, 1359, 57.2, 67),
    "Power failure": ("L", 298.9, 310.2, 2737, 8.8, 142),
    "Overstrain failure": ("L", 299.1, 310.2, 1317, 54.8, 231),
}
INPUT_KEYS = ["in_type", "in_air", "in_process", "in_speed", "in_torque", "in_wear"]


def load_example():
    choice = st.session_state["example_choice"]
    if choice in EXAMPLES:
        for key, value in zip(INPUT_KEYS, EXAMPLES[choice]):
            st.session_state[key] = value


def predict_page():
    st.header("🤖 Predict Machine Failure")
    st.write(
        "Enter the machine type and its sensor readings. The model trained in the "
        "Jupyter Notebook (StandardScaler → PCA → SVM) predicts whether the machine will fail."
    )

    bundle = load_model()
    if bundle is None:
        st.error(
            "Trained model not found (artifacts/pca_failure_model.joblib). "
            "Open the Jupyter Notebook and run all cells first, then refresh this page."
        )
        return

    # default values (a normal machine)
    for key, value in zip(INPUT_KEYS, EXAMPLES["Normal machine"]):
        st.session_state.setdefault(key, value)

    st.selectbox(
        "Load an example machine (optional)",
        ["Custom input"] + list(EXAMPLES.keys()),
        key="example_choice",
        on_change=load_example,
    )

    c1, c2, c3 = st.columns(3)
    c1.selectbox("Machine type (quality variant)", ["L", "M", "H"], key="in_type",
                 help="L = low, M = medium, H = high quality variant")
    c2.number_input("Air temperature [K]", 290.0, 310.0, step=0.1, key="in_air",
                     help="Typical range in the dataset: 295 – 305 K")
    c3.number_input("Process temperature [K]", 300.0, 320.0, step=0.1, key="in_process",
                     help="Typical range in the dataset: 305 – 314 K")
    c4, c5, c6 = st.columns(3)
    c4.number_input("Rotational speed [rpm]", 1000, 3000, step=10, key="in_speed",
                     help="Typical range in the dataset: 1170 – 2890 rpm")
    c5.number_input("Torque [Nm]", 0.0, 100.0, step=0.1, key="in_torque",
                     help="Typical range in the dataset: 4 – 77 Nm")
    c6.number_input("Tool wear [min]", 0, 300, step=1, key="in_wear",
                     help="Typical range in the dataset: 0 – 253 min")

    if not st.button("🔍 Predict Failure", type="primary"):
        st.info("Set the values and click **Predict Failure**.")
        return

    machine = pd.DataFrame([{
        "Type": st.session_state["in_type"],
        "Air temperature [K]": float(st.session_state["in_air"]),
        "Process temperature [K]": float(st.session_state["in_process"]),
        "Rotational speed [rpm]": float(st.session_state["in_speed"]),
        "Torque [Nm]": float(st.session_state["in_torque"]),
        "Tool wear [min]": float(st.session_state["in_wear"]),
    }])
    machine = add_engineered_features(machine)
    machine["Type_code"] = machine["Type"].map(bundle["type_mapping"])
    X_new = machine[bundle["features"]]

    pipeline = bundle["pipeline"]
    prediction = int(pipeline.predict(X_new)[0])
    failure_probability = float(pipeline.predict_proba(X_new)[0, 1])

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ FAILURE PREDICTED – schedule maintenance for this machine")
    else:
        st.success("### ✅ NO FAILURE PREDICTED – the machine is operating normally")

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", "Failure" if prediction == 1 else "No Failure")
    m2.metric("Failure probability", f"{failure_probability * 100:.1f} %")
    m3.metric("Inputs used by the model", f"{len(bundle['features'])} → {bundle['n_components']} PCs")
    st.progress(min(max(failure_probability, 0.0), 1.0))

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Engineered features")
        temp_diff = machine["Temperature difference [K]"].iloc[0]
        power = machine["Power [W]"].iloc[0]
        strain = machine["Strain [min*Nm]"].iloc[0]
        strain_limit = {"L": 11000, "M": 12000, "H": 13000}[machine["Type"].iloc[0]]
        checks = pd.DataFrame({
            "Feature": ["Temperature difference [K]", "Power [W]", "Strain [min*Nm]"],
            "Value": [round(temp_diff, 2), round(power, 1), round(strain, 1)],
            "Risk condition (dataset documentation)": [
                "< 8.6 K and speed < 1380 rpm",
                "< 3500 W or > 9000 W",
                f"> {strain_limit} (type {machine['Type'].iloc[0]})",
            ],
            "At risk?": [
                "Yes" if (temp_diff < 8.6 and machine["Rotational speed [rpm]"].iloc[0] < 1380) else "No",
                "Yes" if (power < 3500 or power > 9000) else "No",
                "Yes" if strain > strain_limit else "No",
            ],
        })
        show_table(checks, hide_index=True)
    with right:
        st.subheader("Principal components of this machine")
        pcs = pipeline[:-1].transform(X_new)[0]
        pc_df = pd.DataFrame({"Component": [f"PC{i + 1}" for i in range(len(pcs))], "Value": pcs})
        fig = px.bar(pc_df, x="Component", y="Value", color_discrete_sequence=["#3977c3"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300,
                          margin=dict(l=10, r=10, t=10, b=10))
        show_chart(fig)
    st.caption(
        "The SVM only sees the principal components. Tool-wear failures (TWF) and random failures (RNF) "
        "happen at random times in this dataset, so no model can predict them reliably."
    )


# =====================================================================
# PAGE 5 – MODEL ACCURACY
# =====================================================================
def accuracy_page():
    st.header("🎯 Model Accuracy")
    results = load_results()
    if results is None:
        st.error(
            "Results not found (artifacts/model_results.json). "
            "Open the Jupyter Notebook and run all cells first, then refresh this page."
        )
        return

    comp = pd.DataFrame(results["comparison"])
    improved = comp.iloc[-1]
    baseline_pca = comp.iloc[2]

    st.write(
        f"Results on the **{results['test_records']:,} test records** that were never used for training. "
        f"Final model: **SVM (RBF kernel) on {results['n_components']} principal components** "
        f"of {results['n_features']} features ({results['variance_retained'] * 100:.2f} % variance retained)."
    )

    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{improved['Accuracy (%)']:.2f} %",
             delta=f"{improved['Accuracy (%)'] - baseline_pca['Accuracy (%)']:+.2f} % vs baseline")
    b.metric("Precision", f"{improved['Precision (%)']:.2f} %")
    c.metric("Recall (failures caught)", f"{improved['Recall (%)']:.2f} %",
             delta=f"{improved['Recall (%)'] - baseline_pca['Recall (%)']:+.2f} % vs baseline")
    d.metric("F1-score", f"{improved['F1-score (%)']:.2f} %")
    e.metric("5-fold CV accuracy", f"{results['best_cv_accuracy'] * 100:.2f} %")

    st.success(
        f"Accuracy requirement met: {improved['Accuracy (%)']:.2f} % on the test set "
        f"(cross-validation {results['best_cv_accuracy'] * 100:.2f} %)."
    )

    st.subheader("Comparison of all models")
    show_table(comp, hide_index=True)

    long_df = comp.melt(id_vars="Model", value_vars=["Accuracy (%)", "Recall (%)", "F1-score (%)"],
                        var_name="Metric", value_name="Score (%)")
    fig = px.bar(long_df, x="Model", y="Score (%)", color="Metric", barmode="group", text="Score (%)",
                 color_discrete_sequence=["#3977c3", "#eb6834", "#1baf7a"])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis_range=[0, 110])
    show_chart(fig)

    st.info(
        "Why accuracy alone is not enough: only 3.4 % of machines fail, so always answering "
        "'No Failure' already gives 96.6 % accuracy while catching no failures. The improved model "
        "therefore also raises recall (failures caught) and F1-score."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix – final model")
        cm = np.array(results["confusion_matrix"])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["No Failure", "Failure"], y=["No Failure", "Failure"],
                           labels=dict(x="Predicted", y="Actual", color="Records"))
        cm_fig.update_layout(height=380)
        show_chart(cm_fig)
    with right:
        st.subheader("Model details")
        details = pd.DataFrame({
            "Item": ["Features", "Principal components", "Variance retained", "Classifier",
                     "Best parameters", "Train / test split"],
            "Value": [str(results["n_features"]), str(results["n_components"]),
                      f"{results['variance_retained'] * 100:.2f} %", "SVM (RBF kernel)",
                      ", ".join(f"{k.replace('model__', '')} = {v}" for k, v in results["best_params"].items()),
                      "80 % / 20 % (stratified)"],
        })
        show_table(details, hide_index=True)


if page == "🤖 Predict Failure":
    predict_page()
elif page == "🎯 Model Accuracy":
    accuracy_page()
else:
    # ---------------- DATASET UPLOAD ----------------
    uploaded = st.file_uploader(
        "📂 Upload your engineering dataset (optional – the project dataset is used by default)",
        type=["csv"]
    )

    if uploaded is not None:
        source = uploaded
        source_name = uploaded.name
    elif DATA_PATH.exists():
        source = DATA_PATH
        source_name = "ai4i2020.csv (project dataset)"
    else:
        st.info("Upload your engineering CSV file to begin the analysis.")
        st.stop()

    # ---------------- READ DATASET ----------------
    try:
        df = pd.read_csv(source)
    except Exception as e:
        st.error(f"Unable to read the CSV file: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    st.caption(f"Dataset in use: {source_name}")

    # ---------------- DATA PREPROCESSING ----------------
    excluded_names = {
        "udi",
        "product id",
        "machine failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
        "target",
        "label"
    }

    numeric = df.select_dtypes(include=np.number)

    feature_cols = [
        col for col in numeric.columns
        if col.strip().lower() not in excluded_names
        and not col.strip().lower().endswith(" failure")
    ]

    if len(feature_cols) < 2:
        st.error(
            "At least two eligible numerical features are required for PCA."
        )
        st.stop()

    X = numeric[feature_cols].copy()

    # Replace infinite values with missing values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove completely empty columns
    X = X.dropna(axis=1, how="all")

    # Remove rows containing missing values
    X = X.dropna(axis=0, how="any")

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    if X.shape[0] < 2 or X.shape[1] < 2:
        st.error(
            "Not enough usable rows or features remain after cleaning."
        )
        st.stop()

    # ---------------- FEATURE SCALING ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- PCA VARIANCE CALCULATION ----------------
    max_components = min(X_scaled.shape)

    pca_all = PCA()
    pca_all.fit(X_scaled)

    variance = pca_all.explained_variance_ratio_ * 100
    cumulative = np.cumsum(variance)

    # ---------------- PROJECT OVERVIEW PAGE ----------------
    if page == "🏠 Project Overview":

        st.header("🏠 Project Overview")

        st.write(
            "This project applies Principal Component Analysis (PCA) "
            "to reduce the dimensionality of engineering data. "
            "The data is cleaned, standardized and analyzed to understand "
            "how much variance is captured by the principal components. "
            "A trained SVM model then predicts machine failure from the principal components "
            "(see the Predict Failure and Model Accuracy pages)."
        )

        st.subheader("Dataset Summary")

        a, b, c, d = st.columns(4)

        a.metric("Total Rows", f"{len(df):,}")
        b.metric("Numerical Features", len(feature_cols))
        c.metric("Usable Rows", f"{len(X):,}")
        d.metric("Rows Removed", f"{len(df) - len(X):,}")

        st.subheader("Dataset Preview")
        show_table(df.head(10))

        st.subheader("Project Workflow")

        w1, w2, w3, w4 = st.columns(4)

        w1.markdown("### 1. Preprocessing")
        w1.write("Select numerical features and clean missing or invalid values.")

        w2.markdown("### 2. Standardization")
        w2.write("Scale the features using StandardScaler.")

        w3.markdown("### 3. PCA")
        w3.write("Transform the scaled data into principal components.")

        w4.markdown("### 4. Prediction")
        w4.write("An SVM trained on the principal components predicts machine failure.")

        st.subheader("Numerical Feature Statistics")
        show_table(X.describe().round(3))

    # ---------------- PCA ANALYSIS PAGE ----------------
    elif page == "📈 PCA Analysis":

        st.header("📈 Principal Component Analysis")

        st.write(
            "Choose how many principal components you want to retain. "
            "The charts show the variance captured by each component."
        )

        n_components = st.slider(
            "Select number of principal components",
            min_value=1,
            max_value=max_components,
            value=min(2, max_components),
            step=1
        )

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        pc_columns = [
            f"PC{i + 1}" for i in range(n_components)
        ]

        pca_df = pd.DataFrame(
            X_pca,
            columns=pc_columns,
            index=X.index
        )

        retained = pca.explained_variance_ratio_.sum() * 100

        reduction = (
            1 - n_components / X.shape[1]
        ) * 100

        st.subheader("Analysis Results")

        a, b, c = st.columns(3)

        a.metric("Original Features", X.shape[1])
        b.metric("Reduced Features", n_components)
        c.metric("Variance Retained", f"{retained:.2f}%")

        st.write("Feature reduction")
        st.progress(
            max(0.0, min(1.0, reduction / 100))
        )
        st.caption(f"{reduction:.2f}% fewer features")

        st.divider()

        # Individual component variance
        st.subheader("Explained Variance by Component")

        variance_df = pd.DataFrame({
            "Principal Component": [
                f"PC{i + 1}" for i in range(len(variance))
            ],
            "Variance (%)": variance
        })

        fig1 = px.bar(
            variance_df,
            x="Principal Component",
            y="Variance (%)",
            title="Variance Captured by Each Component",
            color="Variance (%)",
            color_continuous_scale="Blues"
        )

        fig1.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        show_chart(fig1)

        # Cumulative variance
        st.subheader("Cumulative Explained Variance")

        cumulative_df = pd.DataFrame({
            "Principal Component": [
                f"PC{i + 1}" for i in range(len(cumulative))
            ],
            "Cumulative Variance (%)": cumulative
        })

        fig2 = px.line(
            cumulative_df,
            x="Principal Component",
            y="Cumulative Variance (%)",
            markers=True,
            title="Cumulative Variance Retained",
        )

        fig2.add_hline(
            y=95,
            line_dash="dash",
            annotation_text="95% reference"
        )

        fig2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis_range=[0, 105]
        )

        show_chart(fig2)

        # Feature count comparison
        st.subheader("Before vs After PCA")

        comparison_df = pd.DataFrame({
            "Dataset": ["Original Data", "PCA Reduced Data"],
            "Number of Features": [
                X.shape[1],
                n_components
            ]
        })

        fig3 = px.bar(
            comparison_df,
            x="Dataset",
            y="Number of Features",
            text="Number of Features",
            color="Dataset",
            title="Dimensionality Comparison"
        )

        fig3.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False
        )

        show_chart(fig3)

        st.subheader("Transformed Dataset Preview")
        show_table(pca_df.head(10))

        st.download_button(
            "⬇️ Download PCA Reduced Dataset",
            data=pca_df.to_csv(index=False).encode("utf-8"),
            file_name="pca_reduced_data.csv",
            mime="text/csv"
        )

        report_df = pd.DataFrame({
            "Principal Component": [
                f"PC{i + 1}" for i in range(len(variance))
            ],
            "Explained Variance (%)": variance,
            "Cumulative Variance (%)": cumulative
        })

        st.download_button(
            "⬇️ Download Variance Report",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="pca_variance_report.csv",
            mime="text/csv"
        )

    # ---------------- DATA EXPLORER PAGE ----------------
    elif page == "🔍 Data Explorer":

        st.header("🔍 Data Explorer")

        st.write(
            "Explore the numerical features in your dataset "
            "using search, histograms and a correlation heatmap."
        )

        selected_feature = st.selectbox(
            "Select a feature",
            options=list(X.columns)
        )

        search_text = st.text_input(
            "Search within the selected feature",
            placeholder="Enter a value or keyword"
        )

        display_df = X.copy()

        if search_text.strip():
            matches = display_df[selected_feature].astype(str).str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False
            )
            display_df = display_df[matches]

        st.caption(
            f"Showing {min(len(display_df), 500)} of "
            f"{len(display_df):,} matching rows"
        )

        show_table(display_df.head(500))

        st.subheader(f"Distribution of {selected_feature}")

        hist_fig = px.histogram(
            X,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature}",
            color_discrete_sequence=["#3977c3"]
        )

        hist_fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        show_chart(hist_fig)

        st.subheader("Feature Correlation Heatmap")

        corr = X.corr()

        heatmap_fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Between Numerical Features"
        )

        show_chart(heatmap_fig)

        st.download_button(
            "⬇️ Download Cleaned Numerical Data",
            data=X.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_engineering_data.csv",
            mime="text/csv"
        )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "SmartPCA | Engineering Machine Learning Project | "
    "Dimensionality Reduction using PCA"
)
