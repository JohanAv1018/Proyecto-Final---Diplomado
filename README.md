# Clustering de Pobreza Multidimensional en Colombia

Proyecto final del Diplomado en Modelado Predictivo, Automatización y Proyectos Inteligentes.

## Descripción

Análisis de clustering aplicado a los datos del Índice de Pobreza Multidimensional (IPM) de Colombia, con el objetivo de identificar perfiles de hogares según sus patrones de privación.

Se utilizan técnicas de reducción de dimensionalidad (ACP) y dos métodos de clustering (K-Means y jerárquico) para responder las siguientes preguntas:

> ¿Existen perfiles territoriales de pobreza multidimensional en Colombia?
> ¿Qué combinaciones de privaciones caracterizan a cada tipo de hogar pobre?

## Datos

Los datos provienen de la Encuesta Nacional de Calidad de Vida (ECV 2025) del DANE, disponibles en el portal de microdatos. El dataset contiene 87,060 hogares descritos mediante 15 indicadores binarios del IPM.

## Estructura del repositorio

```
├── MGN2024_DPTO_POLITICO/                                <--- Carpeta obtenida del geoportal del DANE usada para visualizaciones
├── app.py                                                <--- Aplicación en streamlit del proyecto
├── Clustering_Pobreza_Multidimensional_Colombia.ipynb    <--- Notebook
├── IPM2025.csv                                           <--- Dataset
├── README.md                                             <--- Archivo README con descripción e instrucciones
└── requirements.txt                                      <--- Archivo de texto con las librerías necesarias
```

## Cómo reproducir el análisis

1. Clona el repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Abre y ejecuta el notebook de principio a fin

## Resultados

Se identificaron tres perfiles de pobreza: hogares con baja privación general (79%), hogares con privación educativa severa (3.3%) y hogares con déficit de infraestructura y servicios básicos (17.5%). 
Los dos últimos perfiles tienen una distribución territorial diferenciada dentro del país.
