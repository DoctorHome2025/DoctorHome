import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- Cargar datos ---
df = pd.read_excel("enf2025.xlsx")

# --- Preprocesamiento de síntomas ---
df['Sintomas'] = df['Sintomas'].str.lower().fillna('')
df['Sintomas'] = df['Sintomas'].apply(lambda x: [s.strip() for s in x.split(',')])

# --- Extraer síntomas únicos ---
todos_sintomas = sorted(set(s for lista in df['Sintomas'] for s in lista))

# --- Convertir lista de síntomas a vector binario ---
def sintomas_a_vector(sintoma_lista):
    return [1 if s in sintoma_lista else 0 for s in todos_sintomas]

# --- Preparar datos de entrenamiento ---
X = np.array([sintomas_a_vector(s) for s in df['Sintomas']])
y_etiquetas = df['Enfermedad'].values

# --- Codificación de etiquetas (enfermedades) ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_etiquetas)
y = to_categorical(y_encoded)

# --- Crear y entrenar modelo ---
modelo = Sequential([
    Dense(32, activation='relu', input_shape=(len(todos_sintomas),)),
    Dense(16, activation='relu'),
    Dense(len(set(y_etiquetas)), activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(X, y, epochs=200, verbose=0)  # Silencioso

# --- Función para predecir enfermedad desde síntomas ---
def predecir_enfermedad(sintomas_usuario):
    vector = np.array([1 if s in sintomas_usuario else 0 for s in todos_sintomas])
    pred = modelo.predict(np.array([vector]), verbose=0)
    idx = np.argmax(pred)
    enfermedad = encoder.inverse_transform([idx])[0]
    probabilidad = pred[0][idx] * 100
    return enfermedad, probabilidad

# --- Función para obtener todos los síntomas disponibles ---
def get_todos_sintomas():
    return todos_sintomas
