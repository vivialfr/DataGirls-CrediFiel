
import streamlit as st
import joblib
import zipfile
import os
import numpy as np
import pandas as pd
from itertools import product
import plotly.express as px
from datetime import date

# ======================
# Logo 
# ======================
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50; font-family: "Trebuchet MS", sans-serif;'>
        📊  Modelo de Predicción de Cobranza
    </h2>
    """,
    unsafe_allow_html=True)
with col2:
    st.image("logo_CF.jpeg", width=700)  # Ajusta el nombre o tamaño según tu imagen


# ======================
# Configuración inicial
# ======================
#st.set_page_config(page_title="Modelo de Estrategia de Cobranza", layout="wide")
#st.title("📊 Modelo de Predicción: Estrategia de Cobranza Personalizada")

st.markdown("""
<p style='font-size:16px'; color: #2C3E50; font-family: "Trebuchet MS", sans-serif;'>
Este modelo predice la mejor <b>estrategia personalizada de cobranza</b> con base en las características del crédito.
Explora cuál combinación de banco, servicio y emisora maximiza tus probabilidades de éxito.
</p>
""", unsafe_allow_html=True)

# =====================
# Cargar modelos 
# =====================

# Cargar el modelos
model_clas = joblib.load("model.pkl")
model_reg = joblib.load("model_reg.pkl")

# ======================
# Diccionarios de mapeo
# ======================
nombre_bancos = {2: 'Banamex', 14: 'Santander', 12: 'BBVA México', 21: 'Banorte'}
nombre_emisoras = {
    21: '7455', 20: '6114', 36: 'N/A', 79: '7167', 25: '7167', 23: '6111', 62: '6111', 68: '623', 
    34: '7167', 75: '496', 24: '6111', 19: '6114', 51: 'N/A', 78: 'N/A', 1: 'N/A', 17: '7167', 
    10: '623', 9: '623', 5: 'N/A', 12: '496', 22: '7455', 18: '7167', 13: '496', 6: 'Reintento', 
    15: '639', 2: 'N/A', 16: '639', 14: '496'
}

# ======================
# Parámetros del usuario
# ======================
st.sidebar.header("🔧 Características del crédito")

capital = st.sidebar.number_input("💰 Capital:", min_value=0.0, step=0.01)
pagare = st.sidebar.number_input("📝 Pagaré:", min_value=0.0, step=0.01)
fechaAperturaCrédito = st.sidebar.date_input("📅 Fecha de apertura del crédito:")

# Calculamos la antigüedad del crédito en días
antiguedad = (date.today() - fechaAperturaCrédito).days 

base_credito = {
    'capital': capital,
    'pagare': pagare,
    'antiguedad': antiguedad
}

# ======================
# Combinaciones posibles
# ======================
bancos = [2, 14, 12, 21]
servicio = ['En linea', 'Matutino', 'No aplica', 'Tradicional', 'Interbancario', 'Parcial']
emisoras = list(nombre_emisoras.keys())

combinaciones = list(product(bancos, servicio, emisoras))
data = []
for b, s, e in combinaciones:
    row = base_credito.copy()
    row.update({'idBanco': b, 'Servicio': s, 'idEmisora': e})
    data.append(row)

df_combos = pd.DataFrame(data)

# ======================
# Predicciones
# ======================
if st.button("🔍 Predecir mejor estrategia y número de intentos"):
    with st.spinner('Calculando mejores combinaciones y número de intentos...'):
        # Estrategia
        probas = model_clas.predict_proba(df_combos)
        df_combos['proba_exito'] = probas[:, 1]
        
        # Ordenar por probabilidad
        mejor = df_combos.sort_values(by='proba_exito', ascending=False).head(1)
        top5 = df_combos.sort_values(by='proba_exito', ascending=False).head(5)

        # Mapeo para presentación
        mejor_banco = nombre_bancos.get(mejor["idBanco"].values[0], "Desconocido")
        mejor_servicio = mejor["Servicio"].values[0]
        mejor_emisora = nombre_emisoras.get(mejor["idEmisora"].values[0], "Desconocido")
        mejor_proba = mejor["proba_exito"].values[0]

        st.markdown("<h3 style='color:#234F1E'>✅ Mejor combinación recomendada:</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Banco", mejor_banco)
            st.metric("Servicio", mejor_servicio)
        with col2:
            st.metric("Emisora", mejor_emisora)
            st.metric("Probabilidad de Éxito", f"{mejor_proba:.2%}")
            
        # Crear una etiqueta única para cada combinación
        top5['Estrategia'] = (
        top5['idBanco'].astype(str) + ' - ' +
        top5['Servicio'].astype(str) + ' - ' +
        top5['idEmisora'].astype(str)
        )

        # Mapeo para top 5
        top5_mapeado = top5.copy()
        top5_mapeado['idBanco'] = top5_mapeado['idBanco'].replace(nombre_bancos)
        top5_mapeado['idEmisora'] = top5_mapeado['idEmisora'].replace(nombre_emisoras)

        # Mostrar tabla
        st.markdown("### 🏆 Top 5 combinaciones recomendadas:")
        st.dataframe(
            top5_mapeado[['idBanco', 'Servicio', 'idEmisora', 'proba_exito']].rename(columns={
                'idBanco': 'Banco',
                'Servicio': 'Tipo de Servicio',
                'idEmisora': 'Emisora',
                'proba_exito': 'Probabilidad de Éxito'
            }),
            use_container_width=True
        )
        
        # Etiqueta mapeada
        top5_mapeado['Estrategia'] = (
        top5_mapeado['idBanco'].astype(str) + ' - ' +
        top5_mapeado['Servicio'].astype(str) + ' - ' +
        top5_mapeado['idEmisora'].astype(str)
        )
        
        # Gráfico de barras
        fig = px.bar(
            top5_mapeado,
            y='proba_exito',
            x='Estrategia',
            color='idBanco',  
            labels={'proba_exito': 'Probabilidad de Éxito', 'Estrategia': 'Estrategia'},
            title='🎯 Top 5 estrategias por probabilidad de éxito',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Intentos de cobranza
        top5['capital'] = capital
        top5['pagare'] = pagare
        top5['antiguedad'] = antiguedad   

        # Ahora sí puedes hacer la predicción
        probas_reg = model_reg.predict_proba(top5)
        top5['num_intentos_Tot_pred'] = model_reg.predict(top5)
        
        st.markdown("### 🧮 Intentos de cobranza predichos:")
        st.dataframe(top5)  

        # Mostrar resultado
        st.success(f'Los intentos de cobranza son: {top5["num_intentos_Tot_pred"].values[0]} intento(s) para la mejor combinación recomendada.')
        
    
