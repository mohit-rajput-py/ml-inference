"""
=================================================================
 ML Model Comparison & Prediction App
 BCA Mini-Project — Decision Tree vs SVM Classifier
 Dataset: Wine Classification (Scikit-Learn built-in)
=================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="ML Model Comparison App",
    page_icon="",
    layout="wide"
)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"


# -----------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# -----------------------------------------------------------------
@st.cache_data
def load_data():
    """Load Wine dataset and return as DataFrame + raw bunch object."""
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data


@st.cache_resource
def train_models(df, _data):
    """
    Train Decision Tree and SVM classifiers, evaluate them,
    pick the best one, and persist model + scaler to disk.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Basic preprocessing — feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Decision Tree ----
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_pred = dt_model.predict(X_test_scaled)

    # ---- SVM ----
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)

    # ---- Metrics ----
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    results = {
        "Decision Tree": get_metrics(y_test, dt_pred),
        "SVM": get_metrics(y_test, svm_pred),
    }

    models = {
        "Decision Tree": dt_model,
        "SVM": svm_model,
    }

    # ---- Select best model based on Accuracy ----
    best_model_name = max(results, key=lambda name: results[name]["Accuracy"])
    best_model = models[best_model_name]

    # ---- Save best model + scaler using Pickle ----
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return results, best_model_name, best_model, scaler, X.columns.tolist()


# -----------------------------------------------------------------
# 2. LOAD DATA & TRAIN
# -----------------------------------------------------------------
df, data = load_data()
results, best_model_name, best_model, scaler, feature_names = train_models(df, data)
target_names = data.target_names

# -----------------------------------------------------------------
# 3. TITLE & INTRO
# -----------------------------------------------------------------
st.title(" Wine Classification — Model Comparison App")
st.markdown(
    "A BCA mini-project demonstrating **Decision Tree vs SVM** classification "
    "with automatic best-model selection, persistence via **Pickle**, and "
    "live inference using **Streamlit**."
)
st.divider()

# -----------------------------------------------------------------
# 4. DATASET INFORMATION
# -----------------------------------------------------------------
st.header(" Dataset Information")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", df.shape[0])
col2.metric("Features", len(feature_names))
col3.metric("Classes", len(target_names))

with st.expander("View Sample Data"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Class Distribution"):
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)  # 400x250 px # width, height in inches

    df["target"].map(dict(enumerate(target_names))).value_counts().plot(
        kind="bar",
        ax=ax,
        color="#722F37"
    )

    ax.set_xlabel("Wine Class")
    ax.set_ylabel("Count")

    st.pyplot(fig, use_container_width=False)

st.divider()

# -----------------------------------------------------------------
# 5. MODEL COMPARISON DASHBOARD
# -----------------------------------------------------------------
st.header("️ Model Comparison Dashboard")

metrics_df = pd.DataFrame(results).T
metrics_df = metrics_df.round(4)

col_table, col_chart = st.columns([1, 1])

with col_table:
    st.subheader("Performance Metrics")
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color="lightgreen"),
        use_container_width=True
    )

with col_chart:
    st.subheader("Visual Comparison")
    st.bar_chart(metrics_df)

st.divider()

# -----------------------------------------------------------------
# 6. BEST MODEL SUMMARY
# -----------------------------------------------------------------
st.header(" Best Model Summary")

best_metrics = results[best_model_name]

st.success(f"**Selected Model:** {best_model_name} (Highest Accuracy)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{best_metrics['Accuracy']*100:.2f}%")
m2.metric("Precision", f"{best_metrics['Precision']*100:.2f}%")
m3.metric("Recall", f"{best_metrics['Recall']*100:.2f}%")
m4.metric("F1-Score", f"{best_metrics['F1-Score']*100:.2f}%")

st.caption(f" '{MODEL_PATH}' and '{SCALER_PATH}' saved automatically on app start.")

st.divider()

# -----------------------------------------------------------------
# 7. INFERENCE SECTION
# -----------------------------------------------------------------
st.header(" Try a Prediction")
st.markdown("Enter wine sample feature values below to predict its class.")

# Load saved model & scaler (demonstrates real Pickle load, not in-memory reuse)
with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    loaded_scaler = pickle.load(f)

# Dynamically generate input widgets based on dataset features
input_values = {}
cols = st.columns(3)

for idx, feature in enumerate(feature_names):
    col = cols[idx % 3]
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())

    input_values[feature] = col.slider(
        label=feature.replace("_", " ").title(),
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        value=round(mean_val, 2),
    )

predict_btn = st.button(" Predict Wine Class", type="primary", use_container_width=True)

if predict_btn:
    # Prepare input in correct feature order
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Scale using the saved scaler
    input_scaled = loaded_scaler.transform(input_df)

    # Predict
    prediction = loaded_model.predict(input_scaled)[0]
    predicted_class = target_names[prediction]

    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Predicted Class", predicted_class.title())

    with res_col2:
        if hasattr(loaded_model, "predict_proba"):
            proba = loaded_model.predict_proba(input_scaled)[0]
            confidence = np.max(proba) * 100
            st.metric("Confidence", f"{confidence:.2f}%")
        else:
            st.metric("Confidence", "N/A")

    # Probability breakdown chart (if available)
    if hasattr(loaded_model, "predict_proba"):
        proba_df = pd.DataFrame({
        "Class": target_names,
        "Probability": proba
    })

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)  # 400x250 px

    ax.bar(
        proba_df["Class"],
        proba_df["Probability"],
        color="#722F37"
    )

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    st.pyplot(fig, use_container_width=False)

    # Friendly explanation
    st.info(
        f" Based on the entered chemical properties, the model predicts this "
        f"wine sample most likely belongs to the **'{predicted_class}'** class, "
        f"using the **{best_model_name}** algorithm trained on the Wine dataset."
    )

st.divider()

# -----------------------------------------------------------------
# 8. FINAL CONCLUSION
# -----------------------------------------------------------------
st.header(" Final Conclusion")
st.markdown(f"""
This mini-project demonstrates a complete, lightweight ML workflow:

- **Dataset:** Wine recognition dataset ({df.shape[0]} samples, {len(feature_names)} features, {len(target_names)} classes)
- **Preprocessing:** Train-test split + StandardScaler feature scaling
- **Models Trained:** Decision Tree Classifier & Support Vector Machine (SVM)
- **Best Model Selected:** **{best_model_name}** with **{best_metrics['Accuracy']*100:.2f}% accuracy**
- **Persistence:** Model and scaler saved using **Pickle** (`model.pkl`, `scaler.pkl`)
- **Inference:** Real-time prediction via dynamically generated Streamlit widgets

This single-file application (`app.py`) covers the **entire ML pipeline** —
from data loading to model comparison, persistence, and live inference —
making it suitable as a self-contained academic mini-project.
""")

st.caption("Built with Python • Scikit-Learn • Pickle • Streamlit")