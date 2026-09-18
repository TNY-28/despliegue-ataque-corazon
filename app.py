# CARGA DE LIBRERÍAS
import pickle

import pandas as pd
import streamlit as st

# CARGA DEL MODELO
# El orden debe ser el mismo del pickle.dump del notebook
filename = 'modelo-cla.pkl'
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# INTERFAZ GRÁFICA
st.title('Predicción de ataque al corazón')

age = st.slider('Edad', min_value=1, max_value=82, value=45, step=1)
avg_glucose_level = st.slider('Nivel promedio de glucosa', min_value=55.0, max_value=272.0, value=92.0, step=0.1)
hypertension = st.selectbox('Hipertensión', ['No', 'Yes'])
heart_disease = st.selectbox('Enfermedad cardíaca', ['No', 'Yes'])
ever_married = st.selectbox('Alguna vez casado(a)', ['No', 'Yes'])
smoking_status = st.selectbox('Estado de fumador', ["'never smoked'", "'formerly smoked'", 'smokes', 'Unknown'])

# DATOS FUTUROS
# Mismos nombres de variables que en el entrenamiento
datos = [[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]]
data = pd.DataFrame(datos, columns=['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'smoking_status'])

# Recordar medida de calidad del modelo
st.warning('El modelo tiene un F1 macro cercano al 81% (validación cruzada)')

# PREPARACIÓN DE DATOS
data_preparada = data.copy()

# En despliegue drop_first=False
data_preparada = pd.get_dummies(data_preparada, columns=['hypertension', 'heart_disease', 'ever_married', 'smoking_status'], drop_first=False, dtype=int)

# Se adicionan las columnas faltantes y se quitan las sobrantes
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# El modelo guardado es Knn, así que sí se normaliza. Con Tree o RF esta línea se comenta
# En los despliegues no se llama fit
data_preparada[['age', 'avg_glucose_level']] = min_max_scaler.transform(data_preparada[['age', 'avg_glucose_level']])

# PREDICCIÓN
Y_pred = modelo.predict(data_preparada)
data['Prediccion'] = labelencoder.inverse_transform(Y_pred)

st.subheader('Predicción')
st.dataframe(data)
