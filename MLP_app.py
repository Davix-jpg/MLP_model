import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

#? Activación Heaviside personalizada
def heaviside(x):
    return tf.where(x > 0, 1.0, 0.0)

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('perceptron\oral_cancer_prediction_dataset.csv')
    return df

# Preprocesamiento
def preprocesar(df):
    X = df[['Age', 'Tobacco_Use', 'Alcohol_Use']]
    y = df['Survival_Rate'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Crear modelo según opciones del usuario
def crear_modelo(activacion, funcion_perdida, input_dim):
    # Mapear activación a función o string
    if activacion == 'heaviside':
        activacion_func = heaviside
    else:
        activacion_func = activacion  # 'relu', 'tanh', 'sigmoid'

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=activacion_func, input_shape=(input_dim,)),
        tf.keras.layers.Dense(8, activation=activacion_func),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=funcion_perdida, metrics=['mae'])
    return model

# Visualización simple del perceptrón
def mostrar_perceptron():
    st.subheader("Esquema del perceptrón multicapa (MLP)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Coordenadas (x, y) de cada capa
    capas = {
        "Entrada": [(0.1, y) for y in np.linspace(0.8, 0.2, 3)],
        "Oculta1": [(0.4, y) for y in np.linspace(0.9, 0.1, 16)],
        "Oculta2": [(0.7, y) for y in np.linspace(0.85, 0.15, 8)],
        "Salida": [(0.9, 0.5)]
    }

    # Dibujar neuronas
    for capa, neuronas in capas.items():
        for (x, y) in neuronas:
            ax.add_patch(plt.Circle((x, y), 0.02, color='skyblue', ec='black'))
        ax.text(np.mean([n[0] for n in neuronas]), 1.0, capa, fontsize=10, ha='center')

    # Dibujar conexiones con pesos simbólicos
    for origen in capas["Entrada"]:
        for destino in capas["Oculta1"]:
            ax.plot([origen[0], destino[0]], [origen[1], destino[1]], 'gray', linewidth=0.5)

    for origen in capas["Oculta1"]:
        for destino in capas["Oculta2"]:
            ax.plot([origen[0], destino[0]], [origen[1], destino[1]], 'gray', linewidth=0.5)

    for origen in capas["Oculta2"]:
        for destino in capas["Salida"]:
            ax.plot([origen[0], destino[0]], [origen[1], destino[1]], 'gray', linewidth=0.5)

    st.pyplot(fig)

# App principal
def main():
    st.title("Predicción de Supervivencia con MLP")
    df = cargar_datos()
    X, y = preprocesar(df)

    # Barra lateral
    st.sidebar.header("Configuración del modelo")

    # Sliders y selectbox
    epochs = st.sidebar.slider("Epochs", 10, 50, 100, step=5)

    activacion = st.sidebar.selectbox("Función de activación", ["relu", "tanh", "sigmoid", "heaviside"])
    funcion_perdida = st.sidebar.selectbox("Función de pérdida", ["mean_squared_error", "mean_absolute_error"])

    # División datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear modelo
    model = crear_modelo(activacion, funcion_perdida, X.shape[1])

    # Entrenar
    with st.spinner("Entrenando el modelo..."):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1, verbose=0)

    # Evaluación
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.success("Entrenamiento finalizado")
    st.write(f"**Mean Squared Error:** {mse:.4f}")
    st.write(f"**Mean Absolute Error:** {mae:.4f}")

    # Gráfico de entrenamiento
    st.subheader("Curva de entrenamiento")
    st.line_chart({
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    })

    # Mostrar esquema
    mostrar_perceptron()

if __name__ == "__main__":
    main()
