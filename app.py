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
    /* Fuente general un poco más grande */
    html, body, [class*="css"] { font-size: 15px; }

    /* Header centrado */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .header-title {
        font-size: 3.2rem;
        font-weight: 900;
        color: #1a4a7a;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #666;
    }

    /* Pestañas más grandes y ocupando todo el ancho */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        font-size: 1rem;
        font-weight: 600;
        padding: 12px 0;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe;
        border-bottom: 3px solid #1a4a7a;
        color: #1a4a7a;
    }

    /* Métricas */
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a4a7a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 4px;
    }

    /* Cajas de interpretación */
    .interp-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1a4a7a;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #333;
        line-height: 1.6;
    }

    /* Radio buttons más grandes */
    .stRadio > div { gap: 16px; }
    .stRadio label { font-size: 1rem !important; font-weight: 600; }

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

COLORS = ['#e74c3c', '#3498db', '#2ecc71']

PERFILES = {
    0: {
        "nombre": "Baja privación",
        "desc": "Agrupa el 79% de los hogares. El empleo informal es alto (0.78), pero las privaciones en educación, vivienda y servicios son bajas. Son los hogares relativamente mejor posicionados dentro de la muestra del IPM.",
        "variables_clave": ["empleo_formal"]
    },
    1: {
        "nombre": "Privación educativa",
        "desc": "El perfil más pequeño (3.3%) pero el más crítico: concentra el mayor índice de pobreza (60%) y sus hogares presentan rezago escolar (0.81), inasistencia (0.72) y trabajo infantil (0.47) de forma simultánea. La pobreza aquí tiene una dimensión intergeneracional clara.",
        "variables_clave": ["rezago_escolar", "inasistencia_escolar", "trabajo_infantil"]
    },
    2: {
        "nombre": "Déficit de infraestructura",
        "desc": "Representa el 17.5% de los hogares con carencias marcadas en acueducto (0.66), alcantarillado (0.66) y materiales de vivienda (pisos 0.45). Territorialmente concentrado en la periferia del país.",
        "variables_clave": ["acueducto", "alcantarillado", "pisos"]
    }
}

PERFILES_HC = {
    0: {
        "nombre": "Baja privación",
        "desc": "El grupo más amplio bajo clustering jerárquico. Hogares con empleo informal dominante pero acceso razonable a servicios y educación. La distancia de Ward minimiza la varianza intragrupo, agrupando aquí a los hogares con menores carencias generales.",
        "variables_clave": ["empleo_formal"]
    },
    1: {
        "nombre": "Privación educativa y salud",
        "desc": "El clustering jerárquico resalta en este grupo la combinación de rezago escolar, trabajo infantil y barreras de acceso a salud. El método Ward tiende a crear grupos más compactos, por lo que este cluster es más homogéneo que en K-Means.",
        "variables_clave": ["rezago_escolar", "trabajo_infantil", "barreras_acceso_salud"]
    },
    2: {
        "nombre": "Déficit de infraestructura",
        "desc": "Los hogares sin acueducto, alcantarillado ni materiales de vivienda adecuados forman el tercer grupo jerárquico. La estructura de árbol (dendrograma) confirma que estas carencias crean un sub-espacio claramente separado del resto.",
        "variables_clave": ["acueducto", "alcantarillado", "pisos"]
    }
}

# ── Carga y procesamiento (cacheado) ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    url = "https://raw.githubusercontent.com/JohanAv1018/Proyecto-Final---Diplomado/main/IPM2025.csv"
    ipm = pd.read_csv(url)
    ipm_analisis = ipm[variables_ipm + ['IPM', 'POBRE', 'DEPARTAMENTO']].copy()
    ipm_analisis['DEPT_NOMBRE'] = ipm_analisis['DEPARTAMENTO'].map(nombres_dept)
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
    return X, X_std, var_exp, var_cum, pca_9, loadings_df

@st.cache_data(show_spinner=False)
def calcular_tsne(_pca_9):
    np.random.seed(42)
    idx = np.random.choice(len(_pca_9), size=10_000, replace=False)
    X_tsne = TSNE(n_components=2, perplexity=40, learning_rate='auto',
                  init='pca', random_state=42).fit_transform(_pca_9[idx])
    return X_tsne, idx

@st.cache_data(show_spinner=False)
def calcular_clustering(_pca_9):
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = km.fit_predict(_pca_9)
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
    X, X_std, var_exp, var_cum, pca_9, loadings_df = procesar_pca(df)
    X_tsne, idx = calcular_tsne(pca_9)
    labels_km, labels_hc, Z = calcular_clustering(pca_9)
    colombia = cargar_shapefile()

ipm_vals = df['IPM'].values[idx]
lista_depts = sorted(df['DEPT_NOMBRE'].dropna().unique().tolist())

# ── Header centrado ──────────────────────────────────────────────────────────
st.markdown("""
<div class='header-container'>
    <div class='header-title'>Pobreza Multidimensional en Colombia</div>
    <div class='header-subtitle'>Análisis de clustering aplicado al Índice de Pobreza Multidimensional (IPM) · ECV 2025 · DANE</div>
    <div style='margin-top:0.6rem;font-size:0.88rem;color:#888'>
        Ana Sofía Salazar Álvarez &nbsp;·&nbsp; Johan Steven Avilan Peñaloza
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Pestañas ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋  Datos", "📊  ACP", "🔵  Clustering", "🗺️  Territorio"])


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — DATOS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploración por filtros")

    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        variable_sel = st.selectbox("Variable para ranking de departamentos", variables_ipm)

    st.markdown("---")

    # Layout 60 / 20 / 20
    col_graf, col_tabla, col_cards = st.columns([3, 1, 1])

    with col_graf:
        st.markdown("**Proporción de privaciones por variable**")
        privaciones = df[variables_ipm].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(privaciones.index, privaciones.values, color='#4C72B0', alpha=0.8)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.set_xlabel("Proporción de hogares con privación", fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.tick_params(labelsize=10)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_tabla:
        st.markdown(f"**Ranking — {variable_sel}**")
        ranking_tabla = df.groupby('DEPT_NOMBRE')[variable_sel].mean().reset_index()
        ranking_tabla.columns = ['Departamento', 'Proporción con privación']
        ranking_tabla['Proporción con privación'] = ranking_tabla['Proporción con privación'].round(3)
        ranking_tabla = ranking_tabla.sort_values('Proporción con privación', ascending=False)
        st.dataframe(ranking_tabla, use_container_width=True, hide_index=True, height=460)

    with col_cards:
        st.markdown("**Indicadores generales**")
        n_hogares = len(df)
        pct_pobres = df['POBRE'].mean() * 100
        ipm_prom = df['IPM'].mean()
        n_depts = df['DEPARTAMENTO'].nunique()

        tarjetas = [
            (f"{n_hogares:,}", "Total hogares"),
            (f"{pct_pobres:.1f}%", "En pobreza multidimensional"),
            (f"{ipm_prom:.3f}", "IPM promedio nacional"),
            (f"{n_depts}", "Departamentos"),
        ]
        for val, label in tarjetas:
            st.markdown(f"<div class='metric-card' style='margin-bottom:10px'><div class='metric-value'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)


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
        ax.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, label='Varianza individual', color='#4C72B0')
        ax.step(range(1, len(var_cum)+1), var_cum, where='mid', linewidth=2, label='Varianza acumulada', color='#1a4a7a')
        ax.axhline(0.70, color='orange', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.axvline(n_70, color='orange', linestyle=':', linewidth=1.2, alpha=0.9, label=f'{n_70} componentes → 70%')
        ax.axvline(n_80, color='red', linestyle=':', linewidth=1.2, alpha=0.9, label=f'{n_80} componentes → 80%')
        ax.set_xlabel("Componente principal", fontsize=11)
        ax.set_ylabel("Proporción de varianza", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""<div class='interp-box'>
        La varianza se distribuye de forma gradual entre las 15 componentes — no hay un "codo" claro,
        lo cual es típico de variables binarias con correlaciones moderadas.
        Con <b>{n_70} componentes</b> se acumula el 70% de la varianza y con <b>{n_80}</b> el 80%.
        El análisis usa <b>9 componentes</b> como punto de equilibrio entre simplicidad e información retenida.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Cargas de las primeras 4 componentes**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(loadings_df, ax=ax, cmap='RdBu_r', center=0,
                    annot=True, fmt='.2f', linewidths=0.5,
                    cbar_kws={'label': 'Carga'}, annot_kws={'size': 9})
        ax.set_xlabel("Componente principal", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    pc_descripciones = [
        ("PC1", "Déficit de infraestructura", "Acueducto, alcantarillado y pisos son las variables con mayor carga. Resume las carencias en servicios básicos del hogar."),
        ("PC2", "Brecha educativa en menores", "Inasistencia escolar, rezago y trabajo infantil tienen los valores más altos. Captura la dimensión intergeneracional de la pobreza."),
        ("PC3", "Privación mixta educativa-vivienda", "Combina privaciones educativas con condiciones de vivienda, diferenciando entre ambos tipos de carencia."),
        ("PC4", "Acceso al sistema de salud", "Dominada por el aseguramiento en salud (0.67). Es la dimensión de acceso al sistema de salud formal."),
    ]
    cols_pc = st.columns(4)
    for col, (pc, titulo, desc) in zip(cols_pc, pc_descripciones):
        with col:
            st.markdown(f"""<div class='interp-box'>
            <b style='color:#1a4a7a;font-size:1.05rem'>{pc}</b> — <b>{titulo}</b><br><br>
            {desc}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Proyección t-SNE — hogares coloreados por IPM**")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=ipm_vals,
                         cmap='YlOrRd', alpha=0.55, s=7)
    plt.colorbar(scatter, ax=ax, label='IPM')
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Resultados del Clustering")

    st.markdown("**Seleccionar método de clustering:**")
    col_btn1, col_btn2, col_btn_rest = st.columns([1, 1, 4])
    with col_btn1:
        btn_km = st.button("🔵 K-Means", use_container_width=True,
                           type="primary" if st.session_state.get("metodo_clustering", "K-Means") == "K-Means" else "secondary")
    with col_btn2:
        btn_hc = st.button("🌳 Jerárquico", use_container_width=True,
                           type="primary" if st.session_state.get("metodo_clustering", "K-Means") == "Clustering Jerárquico" else "secondary")

    if btn_km:
        st.session_state["metodo_clustering"] = "K-Means"
    if btn_hc:
        st.session_state["metodo_clustering"] = "Clustering Jerárquico"

    metodo = st.session_state.get("metodo_clustering", "K-Means")
    st.markdown("---")

    labels = labels_km if metodo == "K-Means" else labels_hc
    perfiles_activos = PERFILES

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Proyección t-SNE — {metodo}**")
        labels_muestra = labels[idx]
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in range(3):
            mask = labels_muestra == c
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=COLORS[c],
                       label=f'Cluster {c} — {perfiles_activos[c]["nombre"]}',
                       alpha=0.45, s=7, edgecolors='none')
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.legend(markerscale=3, fontsize=9)
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
                    linewidths=0.5, cbar_kws={'label': 'Proporción de hogares con privación'},
                    annot_kws={'size': 9})
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Perfiles identificados**")
    cols_perf = st.columns(3)
    for c, col in enumerate(cols_perf):
        with col:
            st.markdown(f"""<div class='interp-box'>
            <b style='color:#1a4a7a'>Cluster {c} — {perfiles_activos[c]['nombre']}</b><br><br>
            {perfiles_activos[c]['desc']}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
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
    tabla['Perfil'] = tabla['cluster'].map({c: perfiles_activos[c]['nombre'] for c in range(3)})
    tabla = tabla[['cluster', 'Perfil', 'Hogares', '% del total', 'IPM_promedio', 'Pct_pobres']]
    tabla.columns = ['Cluster', 'Perfil', 'Hogares', '% del total', 'IPM promedio', '% en pobreza']
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    if metodo == "Clustering Jerárquico":
        st.markdown("---")
        st.markdown("**Dendrograma**")
        fig, ax = plt.subplots(figsize=(12, 4))
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
                   above_threshold_color='#2c3e50', truncate_mode='lastp', p=30)
        ax.axhline(y=97, color='#e74c3c', linestyle='--', linewidth=2, label='Corte → 3 clusters')
        ax.set_ylabel('Distancia', fontsize=11)
        ax.legend(fontsize=10)
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

    descripciones_territorio = {
        0: "El perfil de baja privación domina en la mayor parte del territorio, especialmente en el centro del país. Bogotá, Cundinamarca, Antioquia, Santander y el eje cafetero concentran las proporciones más altas, lo que refleja su mayor acceso a servicios y desarrollo relativo.",
        1: "La privación educativa no responde a una geografía específica — está dispersa a lo largo del país. Los hogares con estas carencias coexisten con otros perfiles en los mismos departamentos, lo que sugiere que este fenómeno es transversal y no exclusivo de una región.",
        2: "El déficit de infraestructura se concentra claramente en la periferia del país: Chocó, La Guajira, Vichada, Guainía y Vaupés lideran la proporción de hogares en este grupo. Son los departamentos históricamente más rezagados en cobertura de servicios públicos."
    }

    cluster_sel = st.radio("Seleccionar perfil", titulos_clusters, horizontal=True)
    c = titulos_clusters.index(cluster_sel)
    st.markdown("---")

    prop = perfil_dept.groupby('DEPARTAMENTO').apply(
        lambda x: (x['cluster'] == c).mean()
    ).reset_index()
    prop.columns = ['DEPARTAMENTO', 'proporcion']

    mapa_c = colombia.merge(prop, left_on='dpto_ccdgo', right_on='DEPARTAMENTO', how='left')
    san_andres = mapa_c[mapa_c['dpto_ccdgo'] == 88]
    continental = mapa_c[mapa_c['dpto_ccdgo'] != 88]

    col_mapa, col_info = st.columns([1, 1])

    with col_mapa:
        fig, ax = plt.subplots(figsize=(6, 7))
        continental.plot(column='proporcion', ax=ax, cmap='Blues', edgecolor='lightgray',
                         linewidth=0.5, vmin=0, vmax=1, legend=False)
        for _, row in continental.iterrows():
            if row['geometry'] is not None:
                centroid = row['geometry'].centroid
                ax.annotate(row['dpto_cnmbr'].title(), xy=(centroid.x, centroid.y),
                            ha='center', va='center', fontsize=4.5, color='black', fontweight='bold')
        proporcion_sa = san_andres['proporcion'].values[0] if not san_andres.empty else 0
        color_sa = plt.cm.Blues(proporcion_sa)
        axins = ax.inset_axes([0.01, 0.78, 0.12, 0.08])
        axins.add_patch(plt.Rectangle((0.3, 0), 0.3, 0.7, color=color_sa))
        axins.set_xlim(0, 1)
        axins.set_ylim(0, 1)
        axins.set_axis_off()
        axins.set_title('San Andrés', fontsize=6, fontweight='bold', pad=2)
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.6, pad=0.02)
        cbar.set_label('Proporción de hogares', fontsize=9)
        ax.set_title(cluster_sel, fontsize=13, fontweight='bold')
        ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_info:
        st.markdown("**Interpretación**")
        st.markdown(f"<div class='interp-box'>{descripciones_territorio[c]}</div>", unsafe_allow_html=True)

        st.markdown("####")
        st.markdown("**Top 10 departamentos**")
        top10 = prop.copy()
        top10['DEPARTAMENTO'] = top10['DEPARTAMENTO'].map(nombres_dept)
        top10 = top10.dropna().sort_values('proporcion', ascending=False).head(10)
        top10['proporcion'] = (top10['proporcion'] * 100).round(1)
        top10.columns = ['Departamento', '% de hogares en este perfil']
        st.dataframe(top10, use_container_width=True, hide_index=True)

