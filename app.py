import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import cdist

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Pobreza Multidimensional Colombia",
    page_icon="🇨🇴",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a4a7a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555;
        margin-top: 4px;
    }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Variables ────────────────────────────────────────────────────────────────
variables_ipm = [
    'logro_educativo', 'analfabetismo', 'inasistencia_escolar', 'rezago_escolar',
    'atencion_integral', 'trabajo_infantil', 'aseguramiento_salud',
    'barreras_acceso_salud', 'desempleo_larga_duracion', 'empleo_formal',
    'acueducto', 'alcantarillado', 'pisos', 'paredes', 'hacinamiento'
]

nombres_dept = {
    5: "Antioquia", 8: "Atlántico", 11: "Bogotá D.C.", 13: "Bolívar",
    15: "Boyacá", 17: "Caldas", 18: "Caquetá", 19: "Cauca",
    20: "Cesar", 23: "Córdoba", 25: "Cundinamarca", 27: "Chocó",
    41: "Huila", 44: "La Guajira", 47: "Magdalena", 50: "Meta",
    52: "Nariño", 54: "Nte. de Santander", 63: "Quindío", 66: "Risaralda",
    68: "Santander", 70: "Sucre", 73: "Tolima", 76: "Valle del Cauca",
    81: "Arauca", 85: "Casanare", 86: "Putumayo", 88: "San Andrés",
    91: "Amazonas", 94: "Guainía", 95: "Guaviare", 97: "Vaupés", 99: "Vichada"
}

# ── Carga y procesamiento de datos (cacheado) ────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    url = "https://raw.githubusercontent.com/JohanAv1018/Proyecto-Final---Diplomado/main/IPM2025.csv"
    ipm = pd.read_csv(url)
    ipm_analisis = ipm[variables_ipm + ['IPM', 'POBRE', 'DEPARTAMENTO']].copy()
    return ipm_analisis

@st.cache_data(show_spinner=False)
def procesar_pca(_df):
    X = _df[variables_ipm]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_std)
    var_exp = pca_full.explained_variance_ratio_
    var_cum = np.cumsum(var_exp)

    pca_9 = PCA(n_components=9).fit_transform(X_std)

    pca_interp = PCA(n_components=9)
    pca_interp.fit(X_std)
    loadings_df = pd.DataFrame(
        pca_interp.components_[:4].T,
        index=variables_ipm,
        columns=['PC1', 'PC2', 'PC3', 'PC4']
    )

    return X, X_std, var_exp, var_cum, pca_9, loadings_df, pca_interp

@st.cache_data(show_spinner=False)
def calcular_tsne(_pca_9):
    np.random.seed(42)
    idx = np.random.choice(len(_pca_9), size=10_000, replace=False)
    X_tsne = TSNE(n_components=2, perplexity=40, learning_rate='auto',
                  init='pca', random_state=42).fit_transform(_pca_9[idx])
    return X_tsne, idx

@st.cache_data(show_spinner=False)
def calcular_clustering(_pca_9):
    # K-Means
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = km.fit_predict(_pca_9)

    # Jerárquico
    np.random.seed(42)
    idx_hier = np.random.choice(len(_pca_9), size=5_000, replace=False)
    X_hier = _pca_9[idx_hier]
    Z = linkage(X_hier, method='ward')
    labels_hier_muestra = fcluster(Z, t=3, criterion='maxclust') - 1
    centroides_hc = np.array([X_hier[labels_hier_muestra == c].mean(axis=0) for c in range(3)])
    distancias = cdist(_pca_9, centroides_hc, metric='euclidean')
    labels_hc = distancias.argmin(axis=1)

    return labels_km, labels_hc, Z

@st.cache_data(show_spinner=False)
def cargar_shapefile():
    base = "https://github.com/JohanAv1018/Proyecto-Final---Diplomado/raw/refs/heads/main/MGN2024_DPTO_POLITICO/"
    archivos = ["MGN_ADM_DPTO_POLITICO.shp", "MGN_ADM_DPTO_POLITICO.shx",
                "MGN_ADM_DPTO_POLITICO.dbf", "MGN_ADM_DPTO_POLITICO.prj"]
    os.makedirs("shapefile", exist_ok=True)
    for archivo in archivos:
        r = requests.get(base + archivo)
        with open(f"shapefile/{archivo}", 'wb') as f:
            f.write(r.content)
    colombia = gpd.read_file("shapefile/MGN_ADM_DPTO_POLITICO.shp")
    colombia['dpto_cnmbr'] = colombia['dpto_cnmbr'].str.encode('latin-1').str.decode('utf-8')
    colombia['dpto_ccdgo'] = colombia['dpto_ccdgo'].astype(int)
    return colombia

# ── Carga ────────────────────────────────────────────────────────────────────
with st.spinner("Cargando datos y procesando modelos..."):
    df = cargar_datos()
    X, X_std, var_exp, var_cum, pca_9, loadings_df, pca_interp = procesar_pca(df)
    X_tsne, idx = calcular_tsne(pca_9)
    labels_km, labels_hc, Z = calcular_clustering(pca_9)
    colombia = cargar_shapefile()

ipm_vals = df['IPM'].values[idx]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🇨🇴 Pobreza Multidimensional en Colombia")
st.markdown("Análisis de clustering aplicado al Índice de Pobreza Multidimensional (IPM) — ECV 2025, DANE")
st.divider()

# ── Pestañas ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📊 ACP", "🔵 Clustering", "🗺️ Territorio"])


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — DATOS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Resumen del dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>Total hogares</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['POBRE'].mean()*100:.1f}%</div>
            <div class='metric-label'>Hogares en pobreza</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['IPM'].mean():.3f}</div>
            <div class='metric-label'>IPM promedio</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['DEPARTAMENTO'].nunique()}</div>
            <div class='metric-label'>Departamentos</div></div>""", unsafe_allow_html=True)

    st.markdown("####")
    st.subheader("Exploración por departamento")

    dept_options = ["Todos"] + sorted(df['DEPARTAMENTO'].map(nombres_dept).dropna().unique().tolist())
    dept_sel = st.selectbox("Filtrar por departamento", dept_options)

    df_vista = df.copy()
    df_vista['DEPARTAMENTO'] = df_vista['DEPARTAMENTO'].map(nombres_dept)

    if dept_sel != "Todos":
        df_vista = df_vista[df_vista['DEPARTAMENTO'] == dept_sel]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Proporción de privaciones**")
        privaciones = df_vista[variables_ipm].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(privaciones.index, privaciones.values, color='#4C72B0', alpha=0.8)
        ax.set_xlabel("Proporción de hogares con privación")
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Distribución del IPM**")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df_vista['IPM'], bins=40, color='#DD8452', alpha=0.8, edgecolor='none')
        ax.axvline(df_vista['IPM'].mean(), color='#c0392b', linestyle='--', linewidth=2,
                   label=f"Promedio: {df_vista['IPM'].mean():.3f}")
        ax.set_xlabel("IPM")
        ax.set_ylabel("Número de hogares")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 — ACP
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Análisis de Componentes Principales")

    n_70 = np.argmax(var_cum >= 0.70) + 1
    n_80 = np.argmax(var_cum >= 0.80) + 1

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Varianza explicada**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, label='Varianza individual')
        ax.step(range(1, len(var_cum)+1), var_cum, where='mid', linewidth=2, label='Varianza acumulada')
        ax.axhline(0.70, color='orange', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(n_70, color='orange', linestyle=':', linewidth=1.2, alpha=0.8,
                   label=f'{n_70} componentes → 70%')
        ax.axvline(n_80, color='red', linestyle=':', linewidth=1.2, alpha=0.8,
                   label=f'{n_80} componentes → 80%')
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Proporción de varianza")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Cargas de las primeras 4 componentes**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(loadings_df, ax=ax, cmap='RdBu_r', center=0,
                    annot=True, fmt='.2f', linewidths=0.5,
                    cbar_kws={'label': 'Carga'})
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Proyección t-SNE — coloreado por IPM**")
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=ipm_vals,
                         cmap='YlOrRd', alpha=0.4, s=6)
    plt.colorbar(scatter, ax=ax, label='IPM')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Resultados del Clustering")

    metodo = st.radio("Método", ["K-Means", "Clustering Jerárquico"], horizontal=True)
    labels = labels_km if metodo == "K-Means" else labels_hc
    colors = ['#e74c3c', '#3498db', '#2ecc71'] if metodo == "K-Means" else ['#4C72B0', '#DD8452', '#55A868']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Proyección t-SNE — {metodo}**")
        labels_muestra = labels[idx]
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in range(3):
            mask = labels_muestra == c
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[c],
                       label=f'Cluster {c}', alpha=0.4, s=6, edgecolors='none')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Perfil de privaciones por cluster**")
        X_df = pd.DataFrame(X, columns=variables_ipm)
        X_df['cluster'] = labels
        medias = X_df.groupby('cluster')[variables_ipm].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(medias.T, ax=ax, cmap='Blues', annot=True, fmt='.2f',
                    linewidths=0.5, cbar_kws={'label': 'Proporción de hogares con privación'})
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Resumen por cluster**")
    resumen = df.copy()
    resumen['cluster'] = labels
    tabla = resumen.groupby('cluster').agg(
        Hogares=('IPM', 'count'),
        IPM_promedio=('IPM', 'mean'),
        Pct_pobres=('POBRE', 'mean')
    ).reset_index()
    tabla['% del total'] = tabla['Hogares'] / tabla['Hogares'].sum() * 100
    tabla['IPM_promedio'] = tabla['IPM_promedio'].round(3)
    tabla['Pct_pobres'] = (tabla['Pct_pobres'] * 100).round(1)
    tabla['% del total'] = tabla['% del total'].round(1)
    tabla.columns = ['Cluster', 'Hogares', 'IPM promedio', '% en pobreza', '% del total']
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    if metodo == "Clustering Jerárquico":
        st.markdown("**Dendrograma**")
        fig, ax = plt.subplots(figsize=(12, 4))
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
                   above_threshold_color='#2c3e50', truncate_mode='lastp', p=30)
        ax.axhline(y=97, color='#e74c3c', linestyle='--', linewidth=2, label='Corte → 3 clusters')
        ax.set_ylabel('Distancia')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 4 — TERRITORIO
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Distribución territorial de perfiles de pobreza")

    perfil_dept = df.copy()
    perfil_dept['cluster'] = labels_km

    titulos_clusters = [
        'Cluster 0 — Baja privación',
        'Cluster 1 — Privación educativa',
        'Cluster 2 — Déficit de infraestructura'
    ]

    cluster_sel = st.radio("Seleccionar perfil", titulos_clusters, horizontal=True)
    c = titulos_clusters.index(cluster_sel)

    prop = perfil_dept.groupby('DEPARTAMENTO').apply(
        lambda x: (x['cluster'] == c).mean()
    ).reset_index()
    prop.columns = ['DEPARTAMENTO', 'proporcion']

    mapa_c = colombia.merge(prop, left_on='dpto_ccdgo', right_on='DEPARTAMENTO', how='left')
    san_andres = mapa_c[mapa_c['dpto_ccdgo'] == 88]
    continental = mapa_c[mapa_c['dpto_ccdgo'] != 88]

    fig, ax = plt.subplots(figsize=(10, 12))

    continental.plot(column='proporcion', ax=ax, cmap='Blues', edgecolor='lightgray',
                     linewidth=0.5, vmin=0, vmax=1, legend=False)

    for _, row in continental.iterrows():
        if row['geometry'] is not None:
            centroid = row['geometry'].centroid
            ax.annotate(row['dpto_cnmbr'].title(), xy=(centroid.x, centroid.y),
                        ha='center', va='center', fontsize=5.5, color='black', fontweight='bold')

    proporcion_sa = san_andres['proporcion'].values[0] if not san_andres.empty else 0
    color_sa = plt.cm.Blues(proporcion_sa)
    axins = ax.inset_axes([0.01, 0.78, 0.12, 0.08])
    axins.add_patch(plt.Rectangle((0.3, 0), 0.3, 0.7, color=color_sa))
    axins.set_xlim(0, 1)
    axins.set_ylim(0, 1)
    axins.set_axis_off()
    axins.set_title('San Andrés', fontsize=7, fontweight='bold', pad=2)

    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.5, pad=0.02)
    cbar.set_label('Proporción de hogares', fontsize=10)

    ax.set_title(cluster_sel, fontsize=15, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Top 5 departamentos con mayor proporción**")
    top5 = prop.copy()
    top5['DEPARTAMENTO'] = top5['DEPARTAMENTO'].map(nombres_dept)
    top5 = top5.dropna().sort_values('proporcion', ascending=False).head(5)
    top5['proporcion'] = (top5['proporcion'] * 100).round(1)
    top5.columns = ['Departamento', '% de hogares en este cluster']
    st.dataframe(top5, use_container_width=True, hide_index=True)
