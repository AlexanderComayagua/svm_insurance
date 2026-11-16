import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Riesgo Actuarial - K-Means & PCA", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "insurance.csv")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")
KMEANS_PATH = os.path.join(BASE_DIR, "models", "kmeans_riesgo_actuarial.pkl")
PCA_PATH = os.path.join(BASE_DIR, "models", "pca_riesgo_actuarial.pkl")
SVM_PATH = os.path.join(BASE_DIR, "models", "svm_riesgo_actuarial.pkl")

@st.cache_resource
def cargar_artefactos():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    kmeans_pipe = joblib.load(KMEANS_PATH)
    pca_mod = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
    svm_pipe = joblib.load(SVM_PATH) if os.path.exists(SVM_PATH) else None
    return meta, kmeans_pipe, pca_mod, svm_pipe

@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

meta, kmeans_pipe, pca_mod, svm_pipe = cargar_artefactos()
data = cargar_datos()

st.title("📊 Segmentación de Riesgo Actuarial (K-Means, PCA y SVM)")
st.markdown(f"**Estudiante:** {meta.get('autor', 'Cristopher Alexander Flores Miranda')} | **Cuenta:** {meta.get('cuenta', '20221000321')}")
st.divider()

st.sidebar.header("🧾 Datos del Cliente")
age = st.sidebar.slider("Edad", 18, 100, 35)
sex = st.sidebar.selectbox("Sexo", ["female", "male"])
bmi = st.sidebar.number_input("BMI", 12.0, 60.0, 28.5, step=0.1)
children = st.sidebar.number_input("Número de hijos", 0, 10, 1)
smoker = st.sidebar.selectbox("Fumador", ["no", "yes"])
region = st.sidebar.selectbox("Región", ["southwest", "southeast", "northwest", "northeast"])
charges = st.sidebar.number_input("Cargos médicos (charges)", 500.0, 100000.0, 8500.0, step=100.0)

feature_cols = meta.get("features", ["age", "bmi", "children", "charges", "sex", "smoker", "region"])
cliente_df = pd.DataFrame([{
    "age": age, "bmi": bmi, "children": children, "charges": charges,
    "sex": sex, "smoker": smoker, "region": region
}])[feature_cols]

cluster_pred = int(kmeans_pipe.predict(cliente_df)[0])
mapa_riesgo = meta["cluster_to_risk"]
perfiles = meta["perfiles"]
nivel_riesgo = mapa_riesgo[str(cluster_pred)]
perfil_actual = perfiles[str(cluster_pred)]

col1, col2, col3 = st.columns(3)
col1.metric("Cluster Asignado", f"Cluster {cluster_pred}")
col2.metric("Nivel de Riesgo Actuarial", f"Riesgo {nivel_riesgo}")
col3.metric("Cargos Medios del Grupo", f"${perfil_actual['Cargos_Medios']:,.2f}")

if nivel_riesgo == "Bajo":
    st.success(f"✅ **Riesgo Bajo (Cluster {cluster_pred}):** Perfil de baja siniestralidad (edad media {perfil_actual['Edad_Media']:.1f} años, {perfil_actual['Pct_Fumadores']}% fumadores).")
elif nivel_riesgo == "Medio":
    st.warning(f"⚠️ **Riesgo Medio (Cluster {cluster_pred}):** Perfil de siniestralidad moderada (edad media {perfil_actual['Edad_Media']:.1f} años, BMI medio {perfil_actual['BMI_Medio']:.2f}).")
else:
    st.error(f"🚨 **Riesgo Alto (Cluster {cluster_pred}):** Perfil crítico con {perfil_actual['Pct_Fumadores']}% de fumadores y cargos medios de ${perfil_actual['Cargos_Medios']:,.2f}.")

if svm_pipe is not None:
    svm_pred = int(svm_pipe.predict(cliente_df)[0])
    svm_prob = svm_pipe.predict_proba(cliente_df)[0].max() * 100
    st.info(f"🤖 **Validación SVM (RBF):** Cluster {svm_pred} — Riesgo **{mapa_riesgo[str(svm_pred)]}** ({svm_prob:.1f}% confianza).")

st.divider()
col_g1, col_g2 = st.columns([1.3, 1])

with col_g1:
    st.subheader("Proyección PCA 2D")
    prep = kmeans_pipe.named_steps["preprocessor"]
    data["cluster"] = kmeans_pipe.predict(data[feature_cols])
    data["riesgo"] = data["cluster"].astype(str).map(mapa_riesgo)
    if pca_mod is not None:
        coords = pca_mod.transform(prep.transform(data[feature_cols]))
        c_cliente = pca_mod.transform(prep.transform(cliente_df))[0]
        data["PC1"], data["PC2"] = coords[:, 0], coords[:, 1]
        var_exp = meta.get("varianza_pca", [0, 0])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(
            data=data, x="PC1", y="PC2", hue="riesgo",
            hue_order=["Bajo", "Medio", "Alto"],
            palette={"Bajo": "#16a34a", "Medio": "#f59e0b", "Alto": "#dc2626"},
            alpha=0.6, ax=ax
        )
        ax.scatter(c_cliente[0], c_cliente[1], c="#1d4ed8", marker="*", s=300, edgecolors="black", label="Cliente", zorder=10)
        ax.set_xlabel(f"PC1 ({var_exp[0]}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]}%)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

with col_g2:
    st.subheader("Evaluación K-Means")
    eval_df = pd.DataFrame(meta["evaluacion_k"])
    fig2, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(eval_df["k"], eval_df["Inercia"], marker="o", color="#2563eb", label="Inercia")
    ax1.axvline(3, color="#dc2626", linestyle="--")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inercia", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(eval_df["k"], eval_df["Silhouette"], marker="s", color="#059669", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="#059669")
    fig2.tight_layout()
    st.pyplot(fig2)

st.subheader("Resumen de Perfiles por Cluster")
resumen = pd.DataFrame.from_dict(perfiles, orient="index").sort_values("Cargos_Medios")
resumen.index = [f"Cluster {i}" for i in resumen.index]
st.dataframe(resumen, use_container_width=True)
