import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass

# ==============================================================================
# 1. BASE DE DATOS DYWIDAG & CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="Diseño Micropilotes FHWA", layout="wide")

def get_dywidag_db():
    data = {
        "Sistema": [
            # --- Serie R32 ---
            "R32-210 (R32L)", "R32-280 (R32N)", "R32-320", "R32-360 (R32S)",
            # --- Serie R38 ---
            "R38-420", "R38-500 (R38N)", "R38-550",
            # --- Serie R51 ---
            "R51-550 (R51L)", "R51-660", "R51-800 (R51N)",
            # --- Serie T76 ---
            "T76-1200", "T76-1600", "T76-1900",
            # --- Ischebeck Titan (Ref) ---
            "Titan 30/11", "Titan 40/16", "Titan 52/26"
        ],
        # Diámetro Exterior (mm)
        "D_ext_bar_mm": [
            32, 32, 32, 32,      # R32
            38, 38, 38,          # R38
            51, 51, 51,          # R51
            76, 76, 76,          # T76
            30, 40, 52           # Titan
        ],
        # Área Efectiva (mm²) - Valores de Catálogo DSI
        "As_mm2": [
            340, 410, 470, 510,        # R32 (L, N, -, S)
            660, 750, 800,             # R38
            890, 970, 1150,            # R51 (L, -, N) -> R51-800 es 1150 mm2
            1610, 1990, 2360,          # T76
            532, 960, 1590             # Titan (Aprox)
        ],
        # Esfuerzo de Fluencia (MPa)
        # Calculado aprox: Carga Fluencia / Area. 
        # Ej: R51-800 -> Py=640kN / 1150mm2 = 556 MPa.
        # Algunos aceros Ischebeck son S460 o S550+.
        "fy_MPa": [
            470, 535, 530, 550,   # R32
            530, 533, 560,        # R38
            505, 555, 556,        # R51 (R51-800 fy ≈ 556 MPa)
            620, 600, 635,        # T76
            540, 560, 580         # Titan
        ],
        # Carga Última (kN) - Referencia Visual
        "P_ult_kN": [
            210, 280, 320, 360,
            420, 500, 550,
            550, 660, 800,
            1200, 1600, 1900,
            260, 660, 925
        ]
    }
    return pd.DataFrame(data)

# ==============================================================================
# 2. MOTOR DE CÁLCULO (GEOTECNIA Y ESTRUCTURA)
# ==============================================================================

@dataclass
class SoilLayer:
    z_top: float
    z_bot: float
    tipo: str
    alpha_bond: float
    kh: float

    def contains(self, z):
        return self.z_top <= z <= self.z_bot

class MicropileCore:
    def __init__(self, L, D, layers, fs_geo):
        self.L = L
        self.D = D
        self.layers = layers
        self.fs_geo = fs_geo
        self.perimeter = np.pi * D

    def get_layer_prop(self, z, prop):
        if z < 0: return 0.0
        if z > self.layers[-1].z_bot: return getattr(self.layers[-1], prop)
        for l in self.layers:
            if l.contains(z): return getattr(l, prop)
        return getattr(self.layers[-1], prop)

    def calc_axial_capacity(self):
        """Calcula la capacidad geotécnica paso a paso en profundidad."""
        dz = 0.05
        z_arr = np.arange(0, self.L + dz, dz)
        
        q_ult_accum = []
        current_q = 0.0
        alphas = []
        
        for z in z_arr:
            alpha = self.get_layer_prop(z, 'alpha_bond')
            alphas.append(alpha)
            
            # Solo sumar fricción si hay profundidad (evitar z=0)
            delta_q = alpha * self.perimeter * dz
            current_q += delta_q
            q_ult_accum.append(current_q)
            
        q_ult = np.array(q_ult_accum)
        q_adm = q_ult / self.fs_geo
        
        return {
            'z': z_arr,
            'alpha': np.array(alphas),
            'Q_ult_profile': q_ult,
            'Q_adm_profile': q_adm,
            'Q_ult_total': q_ult[-1],
            'Q_adm_total': q_adm[-1]
        }

def calc_winkler_lateral(L, D, EI, kh_ref, V_load, M_load):
    """Método analítico de Winkler para pilotes (modelo de viga sobre fundación elástica)."""
    # Beta factor (Rigidez relativa suelo-pilote)
    beta = ((kh_ref * D) / (4 * EI))**0.25
    
    z_nodes = np.linspace(0, L, 200)
    y_list, M_list, V_list = [], [], []
    
    for z in z_nodes:
        bz = beta * z
        if bz > 15: # Amortiguamiento numérico para profundidades grandes
            y_list.append(0); M_list.append(0); V_list.append(0)
            continue
            
        exp_beta = np.exp(-bz)
        sin_b, cos_b = np.sin(bz), np.cos(bz)
        
        # Funciones de forma Winkler (Shape Functions)
        A = exp_beta * (cos_b + sin_b)
        B = exp_beta * sin_b
        C = exp_beta * (cos_b - sin_b)
        D_fact = exp_beta * cos_b
        
        # Deflexión (m) - Solución para Carga y Momento en cabeza
        y = (2 * V_load * beta / (kh_ref * D)) * D_fact + (2 * M_load * beta**2 / (kh_ref * D)) * C
        
        # Momento (kN.m)
        M = (V_load / beta) * B + M_load * A
        
        # Cortante (kN)
        V = V_load * C - 2 * M_load * beta * D_fact
        
        y_list.append(y)
        M_list.append(M)
        V_list.append(V)
        
    return z_nodes, np.array(y_list), np.array(M_list), np.array(V_list), beta

# ==============================================================================
# 3. INTERFAZ GRÁFICA STREAMLIT
# ==============================================================================

st.title("🛡️ Diseño de Micropilotes - FHWA NHI-05-039")

# --- SIDEBAR: DATOS GLOBALES ---
with st.sidebar:
    st.header("1. Materiales y Geometría")
    
    # Selección de Barra
    df_sys = get_dywidag_db()
    sys_sel = st.selectbox("Sistema de Refuerzo:", df_sys['Sistema'], index=3)
    row_sys = df_sys[df_sys['Sistema'] == sys_sel].iloc[0]
    
    fy_bar = row_sys['fy_MPa']
    As_bar = row_sys['As_mm2']
    
    st.caption(f"**{sys_sel}**: fy={fy_bar} MPa, As={As_bar} mm²")
    
    fc_grout = st.number_input("f'c Grout (MPa)", 20.0, 60.0, 30.0)
    
    st.divider()
    L_tot = st.number_input("Longitud Micropilote (m)", 1.0, 50.0, 12.0)
    D_perf = st.number_input("Diámetro Perforación (m)", 0.1, 0.6, 0.20)
    
    st.header("2. Estratigrafía")
    # Editor de Suelos
    default_soil = pd.DataFrame([
        {"z_top": 0.0, "z_bot": 5.0, "tipo": "Arcilla", "alpha_bond": 60.0, "kh_kN_m3": 8000.0},
        {"z_top": 5.0, "z_bot": 10.0, "tipo": "Arena", "alpha_bond": 120.0, "kh_kN_m3": 18000.0},
        {"z_top": 10.0, "z_bot": 15.0, "tipo": "Roca", "alpha_bond": 350.0, "kh_kN_m3": 50000.0},
    ])
    edited_soil = st.data_editor(default_soil, num_rows="dynamic")
    
    # Crear objetos de capa para el cálculo
    layers_objs = []
    for _, r in edited_soil.iterrows():
        layers_objs.append(SoilLayer(r['z_top'], r['z_bot'], r['tipo'], r['alpha_bond'], r['kh_kN_m3']))

# --- TABS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["🏗️ Capacidad Geotécnica & Axial", "📉 Análisis Lateral & Tubería", "🧱 Diseño de Losa"])

# ==============================================================================
# PESTAÑA 1: GEOTECNIA Y AXIAL (SIN CASING)
# ==============================================================================
with tab1:
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.subheader("Cargas y Factores")
        FS_geo = st.number_input("FS Geotécnico (Bond)", 1.0, 4.0, 2.0)
        P_actuante = st.number_input("Carga Axial Compresión (kN)", 0.0, 5000.0, 450.0)
        
        # --- CÁLCULO GEOTÉCNICO ---
        mp_core = MicropileCore(L_tot, D_perf, layers_objs, FS_geo)
        res_geo = mp_core.calc_axial_capacity()
        
        # --- CÁLCULO ESTRUCTURAL (Solo Barra + Grout) ---
        # Áreas
        A_grout_m2 = (np.pi * (D_perf/2)**2) - (As_bar * 1e-6)
        
        # Compresión (FHWA)
        P_c_all = (0.40 * (fc_grout * 1000) * A_grout_m2) + \
                  (0.47 * (fy_bar * 1000) * (As_bar * 1e-6))
                  
        # Tensión (FHWA)
        P_t_all = 0.55 * (fy_bar * 1000) * (As_bar * 1e-6)
        
        st.markdown("### Resultados Generales")
        m1, m2, m3 = st.columns(3)
        
        delta_geo = None
        if P_actuante > 0:
            delta_geo = "OK" if res_geo['Q_adm_total'] >= P_actuante else "FALLA"
        
        m1.metric("Q Admisible Geotécnico", f"{res_geo['Q_adm_total']:.2f} kN", delta=delta_geo)
        m2.metric("P Admisible Compresión", f"{P_c_all:.2f} kN", help="FHWA (Barra + Grout)")
        m3.metric("P Admisible Tensión", f"{P_t_all:.2f} kN", help="FHWA (Solo Barra)")

    with col_g2:
        # --- GRÁFICO DE ESTRATIGRAFÍA CON DIBUJO DE MICROPILOTE ---
        st.subheader("Perfil del Suelo y Micropilote")
        fig_est, ax_est = plt.subplots(figsize=(4, 6))
        estilos = {"Arcilla": "#D2B48C", "Arena": "#F4A460", "Roca": "#808080"}
        
        # 1. Dibujar Capas
        max_depth_plot = max(L_tot + 2, layers_objs[-1].z_bot)
        
        for l in layers_objs:
            h = l.z_bot - l.z_top
            color = estilos.get(l.tipo, "#D3D3D3")
            # Rectangulo del suelo
            rect = patches.Rectangle((0, l.z_top), 4, h, facecolor=color, edgecolor='none', alpha=0.5)
            ax_est.add_patch(rect)
            # Lineas divisorias
            ax_est.axhline(l.z_top, color='k', linewidth=0.5, alpha=0.3)
            # Texto
            ax_est.text(3.5, l.z_top + h/2, f"{l.tipo}\n$\\alpha$={l.alpha_bond}", 
                        ha='center', va='center', fontsize=7, rotation=90)
            
        # 2. DIBUJAR MICROPILOTE (VISUALIZACIÓN)
        center_x = 2.0
        width_micro = 0.4 # Escala visual
        
        # Cuerpo Grout (Gris)
        rect_micro = patches.Rectangle((center_x - width_micro/2, 0), width_micro, L_tot, 
                                       facecolor='#708090', edgecolor='k', linewidth=1.5, label='Micropilote')
        ax_est.add_patch(rect_micro)
        
        # Refuerzo Central (Rojo)
        ax_est.plot([center_x, center_x], [0, L_tot], color='#8B0000', linestyle='--', linewidth=1.5, label='Barra Central')

        ax_est.set_ylim(max_depth_plot, 0)
        ax_est.set_xlim(0, 4)
        ax_est.axis('off')
        ax_est.legend(loc='lower left', fontsize=8)
        st.pyplot(fig_est)

    st.markdown("---")
    st.subheader("Gráficas de Profundidad")
    
    fig_depth, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    # Gráfica Alpha Bond
    ax1.step(res_geo['alpha'], res_geo['z'], where='post', color='brown', linewidth=2)
    ax1.set_title(r"Adherencia Unit. ($\alpha_{bond}$)")
    ax1.set_xlabel("Alpha Bond (kPa)")
    ax1.set_ylabel("Profundidad (m)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.invert_yaxis()
    
    # Gráfica Capacidad
    ax2.plot(res_geo['Q_adm_profile'], res_geo['z'], 'b-', label='Q admisible')
    ax2.plot(res_geo['Q_ult_profile'], res_geo['z'], 'k--', label='Q última', alpha=0.5)
    if P_actuante > 0:
        ax2.axvline(P_actuante, color='r', linestyle=':', label='Carga Pu')
        
    ax2.set_title("Capacidad Axial Acumulada")
    ax2.set_xlabel("Carga (kN)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig_depth)
    
    with st.expander("Ver Ecuaciones y Referencias (FHWA NHI-05-039)"):
        st.latex(r"Q_{ult} = \sum (\alpha_{bond} \cdot \pi \cdot D_b \cdot \Delta L)")
        st.latex(r"Q_{all} = \frac{Q_{ult}}{FS}")
        st.latex(r"P_{c,all} = 0.40 f'_c A_{grout} + 0.47 f_y A_{bar}")
        st.latex(r"P_{t,all} = 0.55 f_y A_{bar}")

# ==============================================================================
# PESTAÑA 2: ANÁLISIS LATERAL Y TUBERÍA
# ==============================================================================
with tab2:
    st.subheader("1. Configuración y Cargas")
    
    col_lat1, col_lat2 = st.columns(2)
    
    with col_lat1:
        usar_casing = st.checkbox("Incluir Tubería Permanente (Casing)", value=False)
        D_cas_ext = 0; t_cas = 0; fy_cas = 0
        if usar_casing:
            c1, c2, c3 = st.columns(3)
            D_cas_ext = c1.number_input("Ø Ext. (mm)", value=178.0)
            t_cas = c2.number_input("Espesor (mm)", value=12.7)
            fy_cas = c3.number_input("Fy Casing (MPa)", value=240.0)

    with col_lat2:
        V_lat = st.number_input("Carga Lateral Actuante (Vu) - kN", value=30.0)
        M_top = st.number_input("Momento Actuante (Mu) - kNm", value=10.0)

    st.divider()
    
    # --- CÁLCULO DE PROPIEDADES DE SECCIÓN ---
    # 1. Inercias
    I_bar = (np.pi * (row_sys['D_ext_bar_mm']/1000)**4) / 64
    I_grout_tot = (np.pi * D_perf**4) / 64
    I_grout_net = I_grout_tot - I_bar
    
    E_steel = 200000 * 1000 
    E_grout = 4700 * np.sqrt(fc_grout) * 1000 
    
    EI_eff = (E_steel * I_bar) + (E_grout * I_grout_net)
    
    # 2. Casing (si aplica)
    EI_casing = 0
    Mn_casing = 0
    Vn_casing = 0
    Z_cas = 0
    
    if usar_casing:
        D_ext_m = D_cas_ext / 1000
        t_m = t_cas / 1000
        D_int_m = D_ext_m - 2*t_m
        
        # Rigidez
        I_cas = (np.pi * (D_ext_m**4 - D_int_m**4)) / 64
        EI_casing = E_steel * I_cas
        EI_eff += EI_casing
        
        # Resistencia Momento (Plástica - Z)
        Z_cas = (D_ext_m**3 - D_int_m**3) / 6
        Mn_casing = Z_cas * (fy_cas * 1000) # kNm
        
        # Cortante Tubo (AISC: 0.6 Fy Ag/2 por shear lag)
        Ag_cas = (np.pi/4)*(D_ext_m**2 - D_int_m**2)
        Vn_casing = 0.6 * (fy_cas * 1000) * (Ag_cas * 0.5) # kN
    
    # 3. Resistencia Barra
    d_bar_eq = np.sqrt(4 * As_bar / np.pi) / 1000 
    # Módulo Plástico Barra Redonda Sólida
    Z_bar = (d_bar_eq**3) / 6
    
    Mn_bar = Z_bar * (fy_bar * 1000) # kNm
    Vn_bar = 0.6 * (fy_bar * 1000) * (As_bar * 1e-6) # kN
    
    # Totales Resistentes (LRFD phi=0.9 aprox)
    phi_f = 0.9
    phi_v = 0.9
    
    Mn_total = phi_f * (Mn_bar + Mn_casing)
    Vn_total = phi_v * (Vn_bar + Vn_casing)

    # --- CÁLCULO WINKLER ---
    kh_ref = layers_objs[0].kh
    z_lat, y_lat, M_lat, V_lat_arr, beta = calc_winkler_lateral(L_tot, D_perf, EI_eff, kh_ref, V_lat, M_top)
    
    # Máximos
    y_max_mm = np.max(np.abs(y_lat)) * 1000
    M_max = np.max(np.abs(M_lat))
    V_max = np.max(np.abs(V_lat_arr))

    # --- RESULTADOS COMPARATIVOS ---
    st.subheader("2. Verificación Estructural Detallada")
    
    col_check1, col_check2 = st.columns([1, 1])
    
    with col_check1:
        st.markdown("**Comparativa Actuante vs Resistente**")
        
        df_res = pd.DataFrame({
            "Tipo": ["Momento (Flexión)", "Cortante (Shear)"],
            "Actuante (u)": [f"{M_max:.2f} kNm", f"{V_max:.2f} kN"],
            "Capacidad (phi*Rn)": [f"{Mn_total:.2f} kNm", f"{Vn_total:.2f} kN"],
            "Ratio (DCR)": [M_max/Mn_total if Mn_total>0 else 99, V_max/Vn_total if Vn_total>0 else 99],
            "Estado": ["✅ OK" if (M_max/Mn_total)<1 else "❌ FALLA", "✅ OK" if (V_max/Vn_total)<1 else "❌ FALLA"]
        })
        st.dataframe(df_res, hide_index=True)
        
        st.metric("Deflexión Máxima Calculada", f"{y_max_mm:.2f} mm")

    with col_check2:
        # --- BLOQUE DE DETALLE Y ECUACIONES ---
        st.markdown("### 📚 Memoria de Cálculo Detallada")
        
        # 1. Deflexiones (Winkler)
        st.markdown("**1. Deflexión Lateral (Modelo Winkler)**")
        st.caption("Solución analítica para viga larga sobre medio elástico ($L > 4/\lambda$).")
        
        st.markdown("Longitud Característica ($\lambda$ o $\beta$):")
        st.latex(r"\beta = \sqrt[4]{\frac{k_h D}{4 EI_{eff}}}")
        st.write(f"Valor calculado $\\beta$: **{beta:.3f} m⁻¹**")
        
        st.markdown("Ecuación de Deflexión ($y(z)$):")
        st.latex(r"y(z) = \frac{2 V_u \beta}{k_h D} D_{\beta z} + \frac{2 M_u \beta^2}{k_h D} C_{\beta z}")
        st.caption("Donde $C_{\\beta z}$ y $D_{\\beta z}$ son funciones de forma trigonométricas ($e^{-\\beta z} \cos ...$)")
        
        st.markdown("---")
        
        # 2. Resistencias
        st.markdown("**2. Capacidades Resistentes (AISC 360-16)**")
        st.latex(r"M_n = F_y \cdot Z \quad \text{(Plástico)}")
        st.latex(r"V_n = 0.6 F_y \cdot A_{eff}")
        st.caption("Nota: Para tubería (casing), $A_{eff} = A_g/2$ debido al shear lag.")

        st.markdown("---")
        st.markdown("### 🔗 Referencias Normativas")
        st.markdown("""
        * **FHWA NHI-05-039:** *Micropile Design and Construction Reference Manual*. [Ver Manual (FHWA)](https://www.fhwa.dot.gov/engineering/geotech/pubs/05039/)
        * **AISC 360-16:** *Specification for Structural Steel Buildings*. [Ver Norma (AISC)](https://www.aisc.org/globalassets/aisc/publications/standards/a360-16-spec-and-commentary.pdf)
        """)

    st.markdown("---")
    st.subheader("3. Diagramas de Solicitaciones")
    
    fig_lat, (ax_def, ax_mom, ax_shr) = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    
    # Deflexión
    ax_def.plot(y_lat*1000, z_lat, 'm-', linewidth=2)
    ax_def.set_title("Deflexión (mm)")
    ax_def.set_ylabel("Profundidad (m)")
    ax_def.invert_yaxis()
    ax_def.grid(True, linestyle=':')
    ax_def.axvline(0, color='k', linewidth=0.5)
    
    # Momento
    ax_mom.plot(M_lat, z_lat, 'g-', linewidth=2)
    ax_mom.set_title("Momento (kNm)")
    ax_mom.axvline(Mn_total, color='r', ls='--', label='phi*Mn')
    ax_mom.axvline(-Mn_total, color='r', ls='--')
    ax_mom.grid(True, linestyle=':')
    ax_mom.legend(fontsize=8, loc='lower right')
    
    # Cortante
    ax_shr.plot(V_lat_arr, z_lat, 'b-', linewidth=2)
    ax_shr.set_title("Cortante (kN)")
    ax_shr.axvline(Vn_total, color='orange', ls='--', label='phi*Vn')
    ax_shr.axvline(-Vn_total, color='orange', ls='--')
    ax_shr.axvline(0, color='k', linewidth=0.5)
    ax_shr.grid(True, linestyle=':')
    ax_shr.legend(fontsize=8, loc='lower right')
    
    st.pyplot(fig_lat)

# ==============================================================================
# PESTAÑA 3: LOSA
# ==============================================================================
with tab3:
    st.subheader("Verificación de Losa de Cabezal")
    
    cl1, cl2 = st.columns(2)
    with cl1:
        B_losa = st.number_input("Ancho Losa (m)", 1.0, 50.0, 10.0)
        L_losa = st.number_input("Largo Losa (m)", 1.0, 50.0, 10.0)
        h_losa = st.number_input("Espesor Losa (m)", 0.2, 2.0, 0.5)
        fcl_losa = st.number_input("f'c Losa (MPa)", 21.0, 40.0, 28.0)
        
    with cl2:
        q_sobrecarga = st.number_input("Sobrecarga (kN/m²)", 0.0, 100.0, 20.0)
        
    st.markdown("---")
    
    # 1. Distribución
    st.markdown("#### 1. Distribución de Micropilotes")
    w_propio = h_losa * 24 # kN/m2
    q_total = q_sobrecarga + w_propio
    F_total = q_total * B_losa * L_losa # kN
    
    Q_adm_unit = res_geo['Q_adm_total'] # Traido de Tab 1
    
    if Q_adm_unit > 0:
        N_req = int(np.ceil(F_total / Q_adm_unit))
        st.write(f"Carga Total Losa: **{F_total:.1f} kN** (Inc. Peso Propio)")
        st.write(f"Capacidad Unitaria Micropilote: **{Q_adm_unit:.1f} kN**")
        st.metric("Cantidad Mínima Requerida", f"{N_req} unds")
        
        # Sugerencia Grid
        nx = int(np.sqrt(N_req * L_losa/B_losa))
        ny = int(np.ceil(N_req/nx)) if nx > 0 else 1
        st.info(f"Configuración Sugerida: **{nx} x {ny}** = {nx*ny} micropilotes")
        
    # 2. Punzonamiento (Simplificado)
    st.markdown("#### 2. Verificación Punzonamiento (Punching Shear)")
    
    # Carga factorada (asumimos 1.4D por simplicidad o input usuario)
    Pu_punz = st.number_input("Carga Factorada Crítica por Micropilote (Pu) - kN", value=float(res_geo['Q_adm_total']*1.4))
    
    # Perimetro critico b0
    d_eff = h_losa - 0.075 # Recubrimiento
    D_accion = D_cas_ext/1000 if usar_casing else D_perf
    
    b0 = np.pi * (D_accion + d_eff)
    phi_v = 0.75
    
    Vc = 0.33 * np.sqrt(fcl_losa) * b0 * d_eff * 1000
    phi_Vc = phi_v * Vc
    
    cp1, cp2 = st.columns(2)
    cp1.metric("Capacidad Punzonamiento ($\phi V_c$)", f"{phi_Vc:.1f} kN")
    cp2.metric("Demanda ($P_u$)", f"{Pu_punz:.1f} kN")
    
    ratio_punz = Pu_punz / phi_Vc
    if ratio_punz < 1.0:
        st.success(f"✅ PASA Punzonamiento (Ratio: {ratio_punz:.2f})")
    else:
        st.error(f"❌ FALLA Punzonamiento (Ratio: {ratio_punz:.2f})")
        
    st.latex(r"\phi V_c = 0.75 \cdot 0.33 \sqrt{f'_c} \cdot b_o \cdot d")

