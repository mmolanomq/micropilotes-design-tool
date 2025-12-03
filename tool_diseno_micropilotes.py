import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

# Configuración de la página
st.set_page_config(page_title="Diseño Micropilotes Dywidag", layout="wide")

st.title("🛡️ Diseño de Micropilotes - Sistema Dywidag & FHWA")
st.markdown("---")

# --- BASE DE DATOS DYWIDAG (GEWI B500B / S550) ---
# Valores típicos de catálogo. fy = 500 MPa, fu = 550 MPa (aprox)
CATALOGO_DYWIDAG = {
    "GEWI Ø20 mm": {"diam": 20, "area_mm2": 314,  "fy": 500, "peso": 2.47},
    "GEWI Ø25 mm": {"diam": 25, "area_mm2": 491,  "fy": 500, "peso": 3.85},
    "GEWI Ø28 mm": {"diam": 28, "area_mm2": 616,  "fy": 500, "peso": 4.83},
    "GEWI Ø32 mm": {"diam": 32, "area_mm2": 804,  "fy": 500, "peso": 6.31},
    "GEWI Ø40 mm": {"diam": 40, "area_mm2": 1257, "fy": 500, "peso": 9.86},
    "GEWI Ø50 mm": {"diam": 50, "area_mm2": 1963, "fy": 500, "peso": 15.41},
    "GEWI Ø63.5 mm": {"diam": 63.5, "area_mm2": 3167, "fy": 500, "peso": 24.86}
}

# --- BARRA LATERAL: CONFIGURACIÓN ESTRUCTURAL ---
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("1. Acero de Refuerzo (Dywidag)")
    opcion_barra = st.selectbox("Seleccione Barra Dywidag:", list(CATALOGO_DYWIDAG.keys()), index=3)
    
    # Recuperar datos de la barra seleccionada
    prop_barra = CATALOGO_DYWIDAG[opcion_barra]
    diam_barra_mm = prop_barra["diam"]
    area_barra_mm2 = prop_barra["area_mm2"]
    fy_acero = prop_barra["fy"]
    
    # Mostrar ficha técnica rápida
    st.info(f"""
    **Propiedades {opcion_barra}:**
    - Área: {area_barra_mm2} mm²
    - Diámetro: {diam_barra_mm} mm
    - Límite Elástico ($f_y$): {fy_acero} MPa
    """)
    
    st.subheader("2. Grout (Lechada)")
    fc_grout = st.number_input("f'c Grout (MPa)", value=30.0, step=1.0)
    
    st.subheader("3. Casing (Tubería)")
    usar_casing = st.checkbox("Incluir Casing Permanente")
    if usar_casing:
        diam_ext_casing = st.number_input("Diámetro Ext. Casing (mm)", value=178.0)
        espesor_casing = st.number_input("Espesor Pared (mm)", value=12.7)
        fy_casing = st.number_input("fy Casing (MPa)", value=240.0)
    else:
        diam_ext_casing = 0; espesor_casing = 0; fy_casing = 0

# --- FUNCIONES DE CÁLCULO Y GRÁFICAS ---

def calcular_capacidad_geotecnica(df_suelo, d_perf, l_micro, fs_geo):
    capacidad_total_kN = 0
    profundidad_actual = 0
    perimetro = math.pi * d_perf
    detalles = []

    for index, row in df_suelo.iterrows():
        espesor_capa = row['Espesor (m)']
        alpha = row['Alpha Bond (kPa)']
        
        inicio_interseccion = max(0, profundidad_actual)
        fin_interseccion = min(l_micro, profundidad_actual + espesor_capa)
        longitud_efectiva = max(0, fin_interseccion - inicio_interseccion)
        
        carga_capa = longitud_efectiva * perimetro * alpha # kN
        capacidad_total_kN += carga_capa
        
        if longitud_efectiva > 0:
            detalles.append(f"Estrato {index+1} ({row['Tipo de Material']}): {longitud_efectiva:.2f}m x {alpha} kPa = {carga_capa:.1f} kN")
            
        profundidad_actual += espesor_capa

    q_admisible = capacidad_total_kN / fs_geo
    return q_admisible, capacidad_total_kN, detalles

def plot_estratigrafia(df_suelo, d_perf, l_micro):
    fig, ax = plt.subplots(figsize=(4, 6))
    
    # Estilos profesionales (Tramas)
    estilos = {
        "Arcilla": {"c": "#D2B48C", "h": "///"}, 
        "Arena":   {"c": "#F4A460", "h": "..."},
        "Roca":    {"c": "#A9A9A9", "h": "x"},   
        "Relleno": {"c": "#D3D3D3", "h": "*"}
    }
    
    prof_acum = 0
    max_w = 4
    
    # Dibujar suelo
    for _, row in df_suelo.iterrows():
        h = row['Espesor (m)']
        tipo = row['Tipo de Material']
        st_dict = estilos.get(tipo, {"c": "white", "h": ""})
        
        rect = patches.Rectangle((0, prof_acum), max_w, h, 
                                 linewidth=1, edgecolor='black', 
                                 facecolor=st_dict['c'], hatch=st_dict['h'], alpha=0.6)
        ax.add_patch(rect)
        
        # Etiqueta centrada
        ax.text(max_w/2, prof_acum + h/2, 
                f"{tipo}\n$\\alpha$={row['Alpha Bond (kPa)']} kPa", 
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        prof_acum += h
        
    # Dibujar Micropilote
    # El ancho del micropilote en el gráfico es esquemático (no a escala real horizontal para visibilidad)
    w_micro_vis = 0.5 
    rect_micro = patches.Rectangle((max_w/2 - w_micro_vis/2, 0), w_micro_vis, l_micro,
                                   linewidth=1.5, edgecolor='#2F4F4F', facecolor='#4682B4', label='Micropilote')
    ax.add_patch(rect_micro)
    
    # Línea de la barra central (simbólica)
    ax.plot([max_w/2, max_w/2], [0, l_micro], color='black', linestyle='--', linewidth=1)

    ax.set_ylim(max(prof_acum, l_micro) + 1, -1)
    ax.set_xlim(0, max_w)
    ax.axis('off') # Apagar ejes para look esquemático
    ax.set_title(f"Perfil Estratigráfico (L={l_micro}m)", fontsize=10)
    
    return fig

# --- INTERFAZ PRINCIPAL ---

tab1, tab2 = st.tabs(["🏗️ Diseño Geotécnico & Estructural", "📊 Verificaciones & Losa"])

# ==============================================================================
# PESTAÑA 1: DISEÑO PRINCIPAL
# ==============================================================================
with tab1:
    col_izq, col_der = st.columns([1.5, 1])
    
    with col_izq:
        st.subheader("1. Geometría y Estratigrafía")
        
        c1, c2, c3 = st.columns(3)
        d_perforacion = c1.number_input("Diámetro Perforación (m)", 0.1, 0.6, 0.20)
        l_micropilote = c2.number_input("Longitud Micropilote (m)", 1.0, 60.0, 12.0)
        fs_geo = c3.number_input("F.S. Geotécnico", 1.0, 4.0, 2.0)
        
        st.markdown("**Definición de Capas del Suelo:**")
        # Mantener persistencia simple
        if 'n_capas' not in st.session_state: st.session_state.n_capas = 3
        
        n_capas = st.number_input("Número de Estratos", 1, 10, 3)
        datos_suelo = []
        
        for i in range(int(n_capas)):
            col_a, col_b, col_c = st.columns([1, 1, 1])
            t = col_a.selectbox(f"Tipo {i+1}", ["Arcilla", "Arena", "Roca", "Relleno"], key=f"t{i}")
            e = col_b.number_input(f"Espesor {i+1} (m)", 0.1, 50.0, 5.0, key=f"e{i}")
            a = col_c.number_input(f"Alpha Bond {i+1} (kPa)", 0.0, 5000.0, 100.0, key=f"a{i}")
            datos_suelo.append({"Tipo de Material": t, "Espesor (m)": e, "Alpha Bond (kPa)": a})
            
        df_suelo = pd.DataFrame(datos_suelo)
        
        st.divider()
        st.subheader("2. Resultados de Capacidad Axial")
        
        # --- CÁLCULOS ---
        # 1. Geotécnica
        q_adm_geo, q_ult_geo, detalles = calcular_capacidad_geotecnica(df_suelo, d_perforacion, l_micropilote, fs_geo)
        
        # 2. Estructural (FHWA)
        # Áreas en mm2
        area_total_perf = math.pi * ((d_perforacion*1000)/2)**2
        area_grout = area_total_perf - area_barra_mm2
        
        # P_compresion = 0.4*fc*Ag + 0.47*fy*As (Factores ASD servicio típicos FHWA)
        p_comp_est = (0.40 * fc_grout * area_grout + 0.47 * fy_acero * area_barra_mm2) / 1000 # kN
        
        # P_tension = 0.55*fy*As
        p_tens_est = (0.55 * fy_acero * area_barra_mm2) / 1000 # kN
        
        # Mostrar Resultados
        r1, r2, r3 = st.columns(3)
        r1.metric("Cap. Geotécnica Adm.", f"{q_adm_geo:.2f} kN", help=f"Q_ult = {q_ult_geo:.1f} kN")
        r2.metric("Cap. Estruct. Compresión", f"{p_comp_est:.2f} kN", delta="Controla" if p_comp_est < q_adm_geo else None, delta_color="inverse")
        r3.metric("Cap. Estruct. Tensión", f"{p_tens_est:.2f} kN")
        
        # Ecuaciones Referencia
        with st.expander("Ver Ecuaciones Utilizadas (FHWA)"):
            st.latex(r"P_{geo, adm} = \frac{\sum (\alpha_{bond} \cdot \pi D \cdot L_i)}{FS}")
            st.latex(r"P_{c, est} = 0.40 f'_c A_{grout} + 0.47 f_y A_{bar}")
            st.latex(r"P_{t, est} = 0.55 f_y A_{bar}")

    with col_der:
        st.markdown("#### Visualización")
        if not df_suelo.empty:
            fig = plot_estratigrafia(df_suelo, d_perforacion, l_micropilote)
            st.pyplot(fig)

# ==============================================================================
# PESTAÑA 2: VERIFICACIONES & LOSA
# ==============================================================================
with tab2:
    col_flex, col_punz = st.columns(2)
    
    with col_flex:
        st.subheader("Verificación a Flexión ($EI_{eff}$)")
        st.info("Cálculo de rigidez compuesta para entrada en software (L-Pile / Group)")
        
        # Inercias (mm4)
        I_barra = (math.pi * diam_barra_mm**4) / 64
        I_grout = (math.pi * (d_perforacion*1000)**4) / 64 - I_barra
        
        # Módulos (MPa)
        E_acero = 200000 
        E_grout = 4700 * math.sqrt(fc_grout)
        
        EI_val = (E_acero * I_barra) + (E_grout * I_grout)
        
        msg_casing = "Sin Casing"
        if usar_casing:
            d_int_c = diam_ext_casing - 2*espesor_casing
            I_casing = (math.pi * (diam_ext_casing**4 - d_int_c**4)) / 64
            EI_val += (E_acero * I_casing)
            msg_casing = "Con Casing"
            
        st.metric("Rigidez Flexión (EI)", f"{EI_val/1e9:.3f} MN·m²", delta=msg_casing)
        st.latex(r"EI_{eff} = E_s I_{casing} + E_g I_{grout} + E_s I_{bar}")

    with col_punz:
        st.subheader("Verificación Punzonamiento Losa")
        
        hl = st.number_input("Espesor Losa (m)", 0.2, 2.0, 0.6)
        fcl = st.number_input("f'c Losa (MPa)", 21.0, 40.0, 28.0)
        pu_load = st.number_input("Carga Pu (kN)", 100.0, 5000.0, 600.0)
        
        d_eff = hl - 0.075 # recubrimiento asumido
        
        # Perímetro crítico b0
        # Diámetro de la columna de reacción (placa o micropilote)
        d_reaccion = (diam_ext_casing/1000) if usar_casing else d_perforacion
        b0 = math.pi * (d_reaccion + d_eff)
        
        # ACI Vc
        phi = 0.75
        vc = 0.33 * math.sqrt(fcl) * b0 * d_eff * 1000 # kN
        phi_vc = phi * vc
        
        ratio = pu_load / phi_vc
        
        c_p1, c_p2 = st.columns(2)
        c_p1.metric("Capacidad Losa", f"{phi_vc:.1f} kN")
        c_p2.metric("Demanda", f"{pu_load:.1f} kN")
        
        if ratio < 1.0:
            st.success(f"✅ OK (Ratio: {ratio:.2f})")
        else:
            st.error(f"❌ FALLA (Ratio: {ratio:.2f})")
