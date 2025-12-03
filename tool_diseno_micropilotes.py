import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

# Configuración de la página
st.set_page_config(page_title="Diseño de Micropilotes FHWA", layout="wide")

st.title("🛡️ Herramienta de Diseño de Micropilotes - FHWA NHI-05-039")

# --- BARRA LATERAL: MATERIALES GLOBALES ---
with st.sidebar:
    st.header("Propiedades de los Materiales")
    st.markdown("### 🧱 Grout (Lechada)")
    fc_grout = st.number_input("f'c Grout (MPa)", value=28.0, step=1.0)
    
    st.markdown("### ⛓️ Acero de Refuerzo (Barra)")
    fy_acero = st.number_input("fy Acero (MPa)", value=420.0, step=10.0)
    diam_barra_mm = st.selectbox("Diámetro Barra (mm)", [20, 25, 32, 40, 50, 63], index=2)
    area_barra_cm2 = (math.pi * (diam_barra_mm/10)**2) / 4
    st.caption(f"Área de barra: {area_barra_cm2:.2f} cm²")

    st.markdown("### 🚇 Casing (Tubería)")
    usar_casing = st.checkbox("Incluir Casing en análisis estructural")
    if usar_casing:
        fy_casing = st.number_input("fy Casing (MPa)", value=240.0)
        diam_ext_casing = st.number_input("Diámetro Ext. Casing (mm)", value=178.0)
        espesor_casing = st.number_input("Espesor Pared Casing (mm)", value=12.0)
    else:
        fy_casing = 0; diam_ext_casing = 0; espesor_casing = 0

# --- FUNCIONES DE CÁLCULO ---

def calcular_capacidad_geotecnica(df_suelo, d_perf, l_micro, fs_geo):
    """Calcula la capacidad por fricción sumando el aporte de cada estrato."""
    capacidad_total_kN = 0
    profundidad_actual = 0
    perimetro = math.pi * d_perf
    
    detalles = []

    for index, row in df_suelo.iterrows():
        espesor_capa = row['Espesor (m)']
        alpha = row['Alpha Bond (kPa)']
        
        # Determinar segmento de micropilote en esta capa
        inicio_capa = profundidad_actual
        fin_capa = profundidad_actual + espesor_capa
        
        # Intersección entre el micropilote (0 a l_micro) y la capa
        inicio_interseccion = max(0, inicio_capa)
        fin_interseccion = min(l_micro, fin_capa)
        
        longitud_efectiva = max(0, fin_interseccion - inicio_interseccion)
        
        carga_capa = longitud_efectiva * perimetro * alpha # kN
        capacidad_total_kN += carga_capa
        
        if longitud_efectiva > 0:
            detalles.append(f"Estrato {index+1}: {longitud_efectiva:.2f}m contacto x {alpha} kPa = {carga_capa:.1f} kN")
            
        profundidad_actual += espesor_capa

    q_admisible = capacidad_total_kN / fs_geo
    return q_admisible, capacidad_total_kN, detalles

def plot_estratigrafia(df_suelo, diametro_perforacion, longitud_micropilote):
    fig, ax = plt.subplots(figsize=(4, 6))
    estilos = {
        "Arcilla": {"color": "#D2B48C", "hatch": "///"}, "Arena": {"color": "#F4A460", "hatch": "..."},
        "Roca": {"color": "#808080", "hatch": "x"}, "Relleno": {"color": "#A9A9A9", "hatch": "*"}
    }
    prof_acum = 0
    max_w = 4
    
    for _, row in df_suelo.iterrows():
        h = row['Espesor (m)']
        tipo = row['Tipo de Material']
        st_dict = estilos.get(tipo, {"color": "white", "hatch": ""})
        rect = patches.Rectangle((0, prof_acum), max_w, h, linewidth=1, edgecolor='black', 
                                 facecolor=st_dict['color'], hatch=st_dict['hatch'], alpha=0.6)
        ax.add_patch(rect)
        ax.text(max_w/2, prof_acum + h/2, f"{tipo}\n$\\alpha$={row['Alpha Bond (kPa)']}", 
                ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        prof_acum += h
        
    # Micropilote
    rect_p = patches.Rectangle((max_w/2 - 0.2, 0), 0.4, longitud_micropilote, linewidth=2, edgecolor='k', facecolor='#4682B4')
    ax.add_patch(rect_p)
    
    ax.set_ylim(max(prof_acum, longitud_micropilote)+1, -1)
    ax.set_xlim(0, max_w); ax.axis('off')
    return fig

# --- INICIO DE INTERFAZ ---

tab1, tab2 = st.tabs(["🏗️ Capacidad Geotécnica y Estructural", "📊 Verificaciones (Momento y Losa)"])

# ==============================================================================
# PESTAÑA 1
# ==============================================================================
with tab1:
    col_inputs, col_visual = st.columns([1.2, 0.8])
    
    with col_inputs:
        st.subheader("1. Definición del Perfil y Geometría")
        
        c1, c2 = st.columns(2)
        d_perforacion = c1.number_input("Diámetro Perforación (m)", 0.1, 0.5, 0.20)
        l_micropilote = c2.number_input("Longitud Micropilote (m)", 1.0, 50.0, 12.0)
        fs_geo = c1.number_input("F.S. Geotécnico", 1.0, 5.0, 2.0)
        
        st.markdown("**Estratigrafía:**")
        # Inicialización de estado para capas si no existe
        if 'num_capas' not in st.session_state: st.session_state.num_capas = 3
        
        # Editor de datos simple con st.data_editor si se prefiere, o inputs manuales
        # Usaremos inputs manuales para control total como pidió el usuario
        cols_input = st.columns(3)
        num_capas = cols_input[0].number_input("N° Estratos", 1, 10, 3)
        
        datos_suelo = []
        for i in range(int(num_capas)):
            cc1, cc2, cc3 = st.columns([1, 1, 1])
            tipo = cc1.selectbox(f"Mat. {i+1}", ["Arcilla", "Arena", "Roca", "Relleno"], key=f"t{i}")
            espesor = cc2.number_input(f"Esp. {i+1} (m)", 0.1, 50.0, 4.0, key=f"e{i}")
            alpha = cc3.number_input(f"Alpha {i+1} (kPa)", 0.0, 5000.0, 80.0, key=f"a{i}")
            datos_suelo.append({"Tipo de Material": tipo, "Espesor (m)": espesor, "Alpha Bond (kPa)": alpha})
        
        df_suelo = pd.DataFrame(datos_suelo)

    with col_visual:
        st.markdown("#### Perfil Gráfico")
        fig = plot_estratigrafia(df_suelo, d_perforacion, l_micropilote)
        st.pyplot(fig)

    st.divider()
    
    # --- RESULTADOS PESTAÑA 1 ---
    st.subheader("2. Resultados de Capacidad Axial")
    
    # Cálculos
    q_adm_geo, q_ult_geo, detalles_geo = calcular_capacidad_geotecnica(df_suelo, d_perforacion, l_micropilote, fs_geo)
    
    # Área Grout
    area_total = math.pi * ((d_perforacion*1000)/2)**2 # mm2
    area_barra = math.pi * (diam_barra_mm/2)**2 # mm2
    area_grout = area_total - area_barra # mm2
    
    # FHWA Ec. Compresión (Servicio/ASD aprox con factores 0.40/0.47)
    # P_c_all = 0.40 * f'c * A_grout + 0.47 * fy * A_bar
    p_comp_est = (0.40 * fc_grout * area_grout + 0.47 * fy_acero * area_barra) / 1000 # kN
    
    # FHWA Ec. Tensión
    # P_t_all = 0.55 * fy * A_bar
    p_tens_est = (0.55 * fy_acero * area_barra) / 1000 # kN

    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.info(f"**Capacidad Geotécnica (Q_all)**\n# **{q_adm_geo:.2f} kN**")
        st.caption(f"Carga Última: {q_ult_geo:.2f} kN")
        with st.expander("Ver desglose por estrato"):
            for d in detalles_geo:
                st.write(f"- {d}")
        
    with res_col2:
        st.success(f"**Cap. Estructural Compresión**\n# **{p_comp_est:.2f} kN**")
        st.latex(r"P_{c,all} = 0.40 f'_c A_g + 0.47 f_y A_s")
        
    with res_col3:
        st.warning(f"**Cap. Estructural Tensión**\n# **{p_tens_est:.2f} kN**")
        st.latex(r"P_{t,all} = 0.55 f_y A_s")

# ==============================================================================
# PESTAÑA 2
# ==============================================================================
with tab2:
    st.header("Verificaciones Adicionales")
    
    col_flexion, col_punz = st.columns(2)
    
    with col_flexion:
        st.subheader("1. Verificación a Flexión (Momentos)")
        
        # Calculo de Inercias (mm4)
        I_barra = (math.pi * diam_barra_mm**4) / 64
        I_grout = (math.pi * (d_perforacion*1000)**4) / 64 - I_barra
        
        # Módulos de elasticidad (estimados)
        E_acero = 200000 # MPa
        E_grout = 4700 * math.sqrt(fc_grout) # MPa ACI aprox
        
        EI_eff = (E_acero * I_barra) + (E_grout * I_grout)
        texto_casing = "Sin Casing"
        
        if usar_casing:
            diam_int_casing = diam_ext_casing - 2*espesor_casing
            I_casing = (math.pi * (diam_ext_casing**4 - diam_int_casing**4)) / 64
            EI_eff += (E_acero * I_casing) # Asumimos E_casing = E_acero
            texto_casing = "Con Casing Permanente"
            
        st.metric("Rigidez a Flexión (EI_eff)", f"{EI_eff/1e9:.2f} kN·m²", delta=texto_casing)
        
        st.markdown("##### Ecuación de Rigidez Combinada")
        if usar_casing:
            st.latex(r"EI_{eff} = E_s I_{casing} + E_g I_{grout} + E_s I_{bar}")
        else:
            st.latex(r"EI_{eff} = E_g I_{grout} + E_s I_{bar}")
            
        st.info("💡 Usa este valor de $EI_{eff}$ en tu software de interacción suelo-estructura (L-Pile, PY-Wall) para obtener las deflexiones reales.")

    with col_punz:
        st.subheader("2. Verificación Punzonamiento (Losa)")
        
        # Inputs Losa
        h_losa = st.number_input("Espesor Losa Cabezal (m)", 0.2, 2.0, 0.5)
        recubrimiento = st.number_input("Recubrimiento (m)", 0.0, 0.1, 0.05)
        fc_losa = st.number_input("f'c Losa (MPa)", 21.0, 50.0, 28.0)
        pu_aplicada = st.number_input("Carga Pu actuante (kN)", 0.0, 5000.0, 500.0)
        
        # Calculo Punzonamiento (Simplificado ACI)
        d_eff = h_losa - recubrimiento # Altura efectiva
        
        # Perímetro crítico (b0) a d/2 de la cara del micropilote (o placa)
        # Asumimos que el punzonamiento es alrededor del casing o del micropilote
        diametro_accion = (diam_ext_casing/1000) if usar_casing else d_perforacion
        b0 = math.pi * (diametro_accion + d_eff) 
        
        # Capacidad al cortante (Vc) - ACI 318 Simplificado (sin armadura de cortante)
        # Vc = 0.33 * lambda * sqrt(fc) * b0 * d (Unidades SI: MPa, m -> MN)
        # 0.33 * sqrt(MPa) da resultado en MPa. Multiplicamos por area en m2 -> MN -> *1000 -> kN
        phi_v = 0.75
        vc_punz = 0.33 * math.sqrt(fc_losa) * b0 * d_eff * 1000 
        vu_punz_adm = phi_v * vc_punz
        
        ratio = pu_aplicada / vu_punz_adm
        
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Capacidad (phi*Vc)", f"{vu_punz_adm:.1f} kN")
        c_res2.metric("Demanda (Pu)", f"{pu_aplicada:.1f} kN")
        
        st.markdown("**Estado:**")
        if pu_aplicada < vu_punz_adm:
            st.success(f"✅ PASA (Ratio: {ratio:.2f})")
        else:
            st.error(f"❌ FALLA (Ratio: {ratio:.2f})")
            
        st.latex(r"\phi V_c = \phi \cdot 0.33 \sqrt{f'_c} \cdot b_o \cdot d")
        st.caption(f"Perímetro crítico $b_o$: {b0:.2f} m")
