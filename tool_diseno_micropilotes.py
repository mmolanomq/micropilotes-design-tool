import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Diseño de Micropilotes FHWA", layout="wide")

st.title("🛡️ Herramienta de Diseño de Micropilotes - FHWA NHI-05-039")

# --- FUNCIÓN DE GRAFICACIÓN ESTRATIGRÁFICA ---
def plot_estratigrafia(df_suelo, diametro_perforacion, longitud_micropilote):
    """
    Genera un gráfico profesional del perfil de suelo y el micropilote.
    """
    fig, ax = plt.subplots(figsize=(4, 8))
    
    # Definir estilos para tipos de suelo (Colores y Tramas)
    estilos = {
        "Arcilla": {"color": "#D2B48C", "hatch": "///"},
        "Arena":   {"color": "#F4A460", "hatch": "..." },
        "Roca":    {"color": "#808080", "hatch": "x"  },
        "Relleno": {"color": "#A9A9A9", "hatch": "*"  }
    }
    
    profundidad_actual = 0
    max_width = 4 # Ancho visual del suelo
    
    # Dibujar capas de suelo
    for index, row in df_suelo.iterrows():
        espesor = row['Espesor (m)']
        tipo = row['Tipo de Material']
        
        estilo = estilos.get(tipo, {"color": "white", "hatch": ""})
        
        # Rectángulo del suelo
        rect = patches.Rectangle((0, profundidad_actual), max_width, espesor, 
                                 linewidth=1, edgecolor='black', facecolor=estilo['color'], 
                                 hatch=estilo['hatch'], alpha=0.6)
        ax.add_patch(rect)
        
        # Etiqueta de la capa
        ax.text(max_width/2, profundidad_actual + espesor/2, 
                f"{tipo}\n$\\alpha_{{bond}}$={row['Alpha Bond (kPa)']} kPa", 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        profundidad_actual += espesor
        
    # Dibujar el Micropilote (Esquemático)
    # Asumimos que el micropilote empieza en 0 y baja hasta la longitud definida
    micropilote_rect = patches.Rectangle((max_width/2 - 0.2, 0), 0.4, longitud_micropilote,
                                         linewidth=2, edgecolor='black', facecolor='#4682B4')
    ax.add_patch(micropilote_rect)
    
    # Configuración del eje
    ax.set_ylim(profundidad_actual + 1, -1) # Invertir eje Y y dar margen
    ax.set_xlim(0, max_width)
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel("Perfil Estratigráfico")
    ax.get_xaxis().set_visible(False) # Ocultar eje X
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig

# --- CREACIÓN DE PESTAÑAS ---
tab1, tab2 = st.tabs(["🏗️ Capacidad Geotécnica y Estructural", "📊 Verificaciones (Momento/Cortante/Losa)"])

# ==============================================================================
# PESTAÑA 1: CAPACIDAD GEOTÉCNICA Y ESTRUCTURAL
# ==============================================================================
with tab1:
    col_inputs, col_visual = st.columns([1, 1])
    
    with col_inputs:
        st.subheader("1. Definición del Perfil de Suelo")
        
        # Entrada dinámica de capas
        num_capas = st.number_input("Número de estratos de suelo", min_value=1, max_value=10, value=3)
        
        datos_suelo = []
        for i in range(int(num_capas)):
            st.markdown(f"**Estrato {i+1}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                tipo = st.selectbox(f"Material {i+1}", ["Arcilla", "Arena", "Roca", "Relleno"], key=f"tipo_{i}")
            with c2:
                espesor = st.number_input(f"Espesor (m) {i+1}", min_value=0.1, value=3.0, key=f"esp_{i}")
            with c3:
                alpha = st.number_input(f"Bond Stress (kPa) {i+1}", value=100.0, help="Valor de adherencia unitaria", key=f"alp_{i}")
            datos_suelo.append({"Tipo de Material": tipo, "Espesor (m)": espesor, "Alpha Bond (kPa)": alpha})
        
        df_suelo = pd.DataFrame(datos_suelo)
        
        st.divider()
        st.subheader("2. Geometría y Materiales")
        d_perforacion = st.number_input("Diámetro de Perforación (m)", 0.1, 0.5, 0.2)
        l_micropilote = st.number_input("Longitud Total Micropilote (m)", 1.0, 50.0, 10.0)
        
    with col_visual:
        st.subheader("Visualización Estratigráfica")
        if not df_suelo.empty:
            fig = plot_estratigrafia(df_suelo, d_perforacion, l_micropilote)
            st.pyplot(fig)
        else:
            st.info("Define las capas para ver el gráfico.")

    st.markdown("---")
    
    # --- SECCIÓN DE ECUACIONES Y RESULTADOS ---
    st.subheader("3. Cálculo de Capacidades (FHWA NHI-05-039)")
    
    c_geo, c_est = st.columns(2)
    
    with c_geo:
        st.markdown("### 🪨 Capacidad Geotécnica (Bond)")
        st.latex(r"""
        P_{G, allowable} = \frac{\alpha_{bond} \cdot \pi \cdot D_b \cdot L_b}{FS}
        """)
        st.caption("Donde $D_b$ es el diámetro de perforación y $L_b$ la longitud de empotramiento.")
        
        # Muestra Tabla Típica FHWA (Valores referenciales)
        with st.expander("Ver Tabla FHWA - Valores Típicos de Adherencia (Grout-to-Ground)"):
            df_fhwa = pd.DataFrame({
                "Tipo de Suelo / Roca": ["Arena/Grava (Densidad Media)", "Arcilla (Rígida)", "Lutita (Shale)", "Arenisca (Sandstone)"],
                "Rango Alpha Bond (kPa)": ["70 - 175", "40 - 100", "205 - 550", "520 - 1380"]
            })
            st.table(df_fhwa)

    with c_est:
        st.markdown("### 🔩 Capacidad Estructural (Compresión)")
        st.latex(r"""
        P_{c, allowable} = 0.40 f'_c A_{grout} + 0.47 f_y A_{steel}
        """)
        st.info("Nota: Los factores 0.40 y 0.47 corresponden a combinaciones de carga de servicio (ASD).")
        
        st.markdown("### ⛓️ Capacidad Estructural (Tensión)")
        st.latex(r"""
        P_{t, allowable} = 0.55 f_y A_{steel}
        """)

# ==============================================================================
# PESTAÑA 2: VERIFICACIONES ADICIONALES
# ==============================================================================
with tab2:
    st.header("Verificaciones de Estado Límite y Losa")
    
    col_lat, col_losa = st.columns(2)
    
    with col_lat:
        st.subheader("Cargas Laterales y Momento")
        tipo_tuberia = st.radio("Configuración de Tubería (Casing)", ["Sin Tubería", "Con Tubería Permanente"])
        
        st.markdown("**Ecuación de Rigidez a Flexión ($EI$):**")
        if tipo_tuberia == "Con Tubería Permanente":
            st.latex(r"EI_{eff} = E_s I_{casing} + E_g I_{grout} + E_s I_{bar}")
        else:
            st.latex(r"EI_{eff} = E_g I_{grout} + E_s I_{bar}")
            
        st.warning("⚠️ Aquí iría el módulo de cálculo de deflexiones (p-y curves o similar).")

    with col_losa:
        st.subheader("Verificación de la Losa de Cabezal")
        st.latex(r"V_{punzonamiento} < \phi V_c")
        st.latex(r"M_{u} < \phi M_n")
        
        st.text_area("Notas de diseño para la losa:", height=100)
