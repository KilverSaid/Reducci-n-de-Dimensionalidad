import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuración de la página
st.set_page_config(page_title="MNIST: PCA + K-Means + SVM", layout="wide")

st.title("Pipeline de IA: Reducción, Clusterización y Clasificación (MNIST)")
st.write("Esta aplicación aplica PCA para reducir dimensionalidad, K-Means para agrupar y SVM para clasificar dígitos manuscritos.")

# 1. Cargar el Dataset optimizado
@st.cache_data
def cargar_datos():
    # 1. Leer el archivo CSV
    data = pd.read_csv("mnist_sample.csv")
    
    # 2. LIMPIEZA: Eliminar cualquier fila que tenga valores nulos (NaN)
    data = data.dropna()
    
    # 3. Separar características (píxeles) y etiquetas
    X = data.iloc[:, 1:].values / 255.0  # Normalizar píxeles (0-1)
    y = data.iloc[:, 0].values          # Etiquetas (0-9)
    
    return X, y

try:
    X, y = cargar_datos()
    st.sidebar.success(f"Dataset cargado con éxito ({X.shape[0]} muestras).")
except FileNotFoundError:
    st.error("Por favor, asegúrate de tener el archivo 'mnist_sample.csv' en el mismo directorio.")
    st.stop()

# --- CONTROLES EN LA BARRA LATERAL ---
st.sidebar.header("Configuración del Pipeline")

# Selección de Componentes Principales para el entrenamiento de SVM
n_components = st.sidebar.slider(
    "Componentes Principales (PCA) para SVM", 
    min_value=2, max_value=50, value=15, step=1
)

# Selección de Clusters para K-Means
n_clusters = st.sidebar.slider(
    "Número de Clústeres (K-Means)", 
    min_value=2, max_value=15, value=10, step=1
)

# --- PASO 1: APLICACIÓN DE PCA ---
st.header("1. Reducción de Dimensionalidad (PCA)")

# PCA para visualización fija en 2D
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X)

# PCA dinámico según lo seleccionado por el usuario para el modelo
pca_modelo = PCA(n_components=n_components, random_state=42)
X_pca_modelo = pca_modelo.fit_transform(X)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Proyección de los Datos en 2D")
    fig_pca, ax_pca = plt.subplots(figsize=(6, 5))
    scatter = ax_pca.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='tab10', alpha=0.6, s=5)
    legend = ax_pca.legend(*scatter.legend_elements(), title="Dígitos")
    ax_pca.add_artist(legend)
    ax_pca.set_xlabel("Componente Principal 1")
    ax_pca.set_ylabel("Componente Principal 2")
    st.pyplot(fig_pca)

with col2:
    st.subheader("Información de Varianza")
    varianza_explicada = np.sum(pca_modelo.explained_variance_ratio_) * 100
    st.metric(label=f"Varianza Total Explicada ({n_components} comp.)", value=f"{varianza_explicada:.2f}%")
    st.write("El algoritmo reduce las 784 variables originales (píxeles) a los componentes seleccionados, manteniendo la mayor cantidad de información posible.")


# --- PASO 2: IMPLEMENTACIÓN DE K-MEANS ---
st.header("2. Aprendizaje No Supervisado (K-Means)")
st.write(f"Agrupando los datos proyectados en 2D utilizando **{n_clusters} clústeres**.")

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
clusters_labels = kmeans.fit_predict(X_pca_2d)

fig_km, ax_km = plt.subplots(figsize=(8, 4))
scatter_km = ax_km.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters_labels, cmap='viridis', alpha=0.6, s=5)
# Dibujar los centroides
centroides = kmeans.cluster_centers_
ax_km.scatter(centroides[:, 0], centroides[:, 1], marker='X', s=150, c='red', label='Centroides')
ax_km.legend()
ax_km.set_title("Clústeres descubiertos por K-Means")
st.pyplot(fig_km)


# --- PASO 3: ENTRENAMIENTO Y EVALUACIÓN DEL MODELO SVM ---
st.header("3. Aprendizaje Supervisado (Support Vector Machine)")
st.write(f"Entrenando un clasificador SVM (Kernel RBF) utilizando los **{n_components} componentes** del PCA.")

# División del dataset reducido
X_train, X_test, y_train, y_test = train_test_split(X_pca_modelo, y, test_size=0.3, random_state=42)

# Ejecutar el modelo con spinner de carga
with st.spinner("Entrenando SVM en tiempo real..."):
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)

col_met1, col_met2 = st.columns([1, 2])

with col_met1:
    st.subheader("Métricas Globales")
    st.metric(label="Exactitud (Accuracy) del Modelo", value=f"{accuracy * 100:.2f}%")
    
    st.write("**Reporte de Clasificación por Dígito:**")
    reporte_dict = classification_report(y_test, y_pred, output_dict=True)
    reporte_df = pd.DataFrame(reporte_dict).transpose().iloc[:-3, :-1] # Limpiar para mostrar solo dígitos
    st.dataframe(reporte_df.style.format("{:.2f}"))

with col_met2:
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
    ax_cm.set_xlabel("Predicción del Modelo")
    ax_cm.set_ylabel("Clase Real (Dígito)")
    st.pyplot(fig_cm)


# --- BONUS: PREDICCIÓN INTERACTIVA ---
st.divider()
st.header("4. Prueba el Clasificador en tiempo real")
idx_test = st.number_input("Selecciona un índice del conjunto de datos original para clasificar (0 a 7999):", min_value=0, max_value=len(X)-1, value=12)

col_img, col_pred = st.columns(2)

with col_img:
    # Mostrar la imagen original reconstruida (28x28)
    fig_digit, ax_digit = plt.subplots(figsize=(2, 2))
    ax_digit.imshow(X[idx_test].reshape(28, 28), cmap='gray')
    ax_digit.axis('off')
    st.pyplot(fig_digit)

with col_pred:
    # Extraer la muestra y pasarla por el mismo pipeline de PCA entrenado
    muestra_reducida = pca_modelo.transform(X[idx_test].reshape(1, -1))
    prediccion_individual = svm_model.predict(muestra_reducida)[0]
    
    st.write(f"**Valor Real Etiquetado:** `{y[idx_test]}`")
    st.write(f"### Predicción SVM:  `{prediccion_individual}`")
    if prediccion_individual == y[idx_test]:
        st.success("¡Predicción Correcta!")
    else:
        st.error("El modelo se equivocó.")
