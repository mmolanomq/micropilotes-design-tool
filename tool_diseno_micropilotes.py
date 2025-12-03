import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# ==============================================================================
# 1. BASE DE DATOS DE SISTEMAS (DYWIDAG / ISCHEBECK)
# ==============================================================================
def get_systems_database():
    """Retorna DataFrame con propiedades mecánicas de barras huecas y sólidas."""
    data = {
        "Sistema": [
            "R32-280", "R32-320", "R32-360", "R38-500", "R51-800", 
            "R51-660", "T76-1300", "T76-1900", "GEWI 20", "GEWI 25", 
            "GEWI 28", "GEWI 32", "GEWI 40", "GEWI 50", "Ischebeck 30/11", "Ischebeck 40/16"
        ],
        # Yield Strength (MPa)
        "fy_MPa": [
            280, 320, 360, 500, 800, 
            660, 1300, 1900, 500, 500, 
            500, 500, 500, 500, 540, 500
        ],
        # Área (mm2)
        "As_mm2": [
            260, 350, 430, 735, 1452, 
            1100, 1750, 3277, 314, 491, 
            616, 804, 1257, 1963, 532, 1250
        ],
        # Diámetro Exterior Barra (mm) - Para cálculo de inercias aproximadas
        "D_ext_bar_mm": [
            32, 32, 32, 38, 51, 
            51, 76, 76, 20, 25, 
            28, 32, 40, 50, 30, 40
        ]
    }
    return pd.DataFrame(data)

# ==============================================================================
# 2. CLASES DE CÁLCULO (Motor Físico & Estructural)
# ==============================================================================

@dataclass
class SoilLayer:
    z_top: float
    z_bot: float
    tipo: str
    alpha_bond: float
    kh: float # Modulo de reaccion horizontal (kN/m3)

    def contains(self, z):
        return self.z_top <= z <= self.z_bot
    
    @property
    def thickness(self):
        return self.z_bot - self.z_top

class MicropileAnalyzer:
    def __init__(self, mp_params, soil_layers, loads):
        self.mp = mp_params
        self.loads = loads
        self.layers = soil_layers
        self.L = mp_params['L_total']
        self.D = mp_params['D_perforacion']
        self.Perimetro = np.pi * self.D

    def get_soil_prop(self, z, prop):
        if z < 0: return 0.0
        # Si la profundidad excede la definida, usa la ultima capa
        if z > self.layers[-1].z_bot: return getattr(self.layers[-1], prop)
        for l in self.layers:
            if l.contains(z): return getattr(l, prop)
        return getattr(self.layers[-1], prop)

    # --- 1. Geotecnia (Bond Stress Integration) ---
    def calc_geo(self):
        dz = 0.1
        z_arr = np.arange(0, self.L + dz, dz)
        q_accum = []
        curr = 0.0
        alphas = []
        
        for z in z_arr:
            alpha = self.get_soil_prop(z, 'alpha_bond')
            alphas.append(alpha)
            # Solo suma capacidad si z > 0 (evita errores de borde)
            curr += alpha * self.Perimetro * dz
            q_accum.append(curr)
        
        q_accum = np.array(q_accum)
        return {
            'z': z_arr, 'alpha': np.array(alphas),
            'Q_ult': q_accum, 'Q_adm': q_accum / self.loads['FS_geo'],
            'Q_ult_tot': q_accum[-1], 'Q_adm_tot': q_accum[-1] / self.loads['FS_geo']
        }

    # --- 2. Estructural (FHWA Compresión/Tensión) ---
    def calc_structural(self):
        # Datos del sistema seleccionado
        As_bar = self.mp['As_bar_mm2'] * 1e-6 # m2
        fy_bar = self.mp['fy_bar_MPa'] * 1000 # kPa
        
        # Casing
        As_casing, fy_casing = 0.0, 0.0
        if self.mp['casing']['usar']:
            De = self.mp['casing']['D_ext']
            t = self.mp['casing']['t']
            As_casing = (np.pi/4)*(De**2 - (De - 2*t)**2)
            fy_casing = self.mp['casing']['fy'] * 1000
        
        # Grout
        Ag_tot = np.pi * (self.D/2)**2
        Ag_grout = Ag_tot - As_bar - As_casing
        fc = self.mp['fc_grout'] * 1000 # kPa
        
        # Compresión (FHWA Service Load)
        P_grout = 0.40 * fc * Ag_grout
        P_bar = 0.47 * fy_bar * As_bar
        P_casing = 0.47 * fy_casing * As_casing
        
        # Tensión
        P_tens = 0.55 * fy_bar * As_bar
        if self.mp['casing']['usar']: 
            P_tens += 0.55 * fy_casing * As_casing
        
        return {
            'P_adm_comp': P_grout + P_bar + P_casing, 
            'P_adm_tens': P_tens,
            'P_grout': P_grout, 'P_bar': P_bar
        }

    # --- 3. Lateral (Finite Difference / Winkler) ---
    def calc_lateral(self):
        n = 100
        z_nodes = np.linspace(0, self.L, n)
        
        # Rigidez EI Efectiva
        Es = 200000 * 1000 # kPa (Acero)
        E_grout = 4700 * np.sqrt(self.mp['fc_grout']) * 1000 # kPa
        
        # Inercia Barra (aproximada como solida o tubo según DB, asumimos solida eq para simplificar o hueca)
        # Usaremos inercia equivalente del area de acero centrada
        d_eq_bar = np.sqrt(4 * (self.mp['As_bar_mm2']*1e-6) / np.pi)
        I_bar = (np.pi/64)*(d_eq_bar**4)
        EI = Es * I_bar
        
        # Aporte Grout (sin agrietar, optimista, o agrietado. FHWA recomienda ignorar grout en tensión, pero se incluye para rigidez inicial)
        I_grout_tot = (np.pi/64)*(self.D**4) - I_bar
        EI += E_grout * I_grout_tot * 0.5 # Factor 0.5 por agrietamiento
        
        if self.mp['casing']['usar']:
            De = self.mp['casing']['D_ext']
            Di = De - 2*self.mp['casing']['t']
            I_casing = (np.pi/64)*(De**4 - Di**4)
            EI += Es * I_casing

        # Solución Analítica Winkler (beta) para suelo constante o promedio
        # Para hacerlo robusto con capas, tomamos kh promedio superior (zona crítica)
        kh_ref = self.get_soil_prop(1.5, 'kh') # a 1.5m de prof
        beta = ((kh_ref * self.D)/(4*EI))**0.25
        
        V0 = self.loads['V_lat']
        M0 = self.loads['M_top']
        
        y_lst, M_lst, V_lst = [], [], []
        
        for z in z_nodes:
            bz = beta * z
            # Evitar overflow en exp
            if bz > 20: 
                y_lst.append(0); M_lst.append(0); V_lst.append(0)
                continue
                
            ebz = np.exp(-bz)
            sin, cos = np.sin(bz), np.cos(bz)
            
            A = ebz * (cos + sin)
            B = ebz * sin
            C = ebz * (cos - sin)
            D_ = ebz * cos
            
            # Deflexion (m)
            y = (2*V0*beta/(kh_ref*self.D))*D_ + (2*M0*beta**2/(kh_ref*self.D))*C
            y_lst.append(y * 1000) # mm
            
            # Momento (kNm)
            M = (V0/beta)*B + M0*A
            M_lst.append(M)
            
            # Cortante (kN)
            V = V0*C - 2*M0*beta*D_
            V_lst.append(V)
            
        return {'z': z_nodes, 'def': np.array(y_lst), 'mom': np.array(M_lst), 
                'shear': np.array(V_lst), 'EI': EI, 'beta': beta}

    # --- 4. Resistencias LRFD (AISC/AASHTO) ---
    def calc_limits(self):
        # Barra
        As = self.mp['As_bar_mm2'] * 1e-6
        fy = self.mp['fy_bar_MPa'] * 1000
        # Inercia Plastica Z aprox
        d_bar = np.sqrt(4*As/np.pi)
        Z_bar = d_bar**3 / 6.0
        Vn_bar = 0.6 * fy * As
        
        # Casing
        Z_cas, Vn_cas, fy_cas = 0.0, 0.0, 0.0
        if self.mp['casing']['usar']:
            De = self.mp['casing']['D_ext']
            t = self.mp['casing']['t']
            Di = De - 2*t
            fy_cas = self.mp['casing']['fy'] * 1000
            
            Z_cas = (De**3 - Di**3)/6.0
            Ag_cas = (np.pi/4)*(De**2 - Di**2)
            Av_cas = Ag_cas / 2.0 # Shear lag reduction
            Vn_cas = 0.6 * fy_cas * Av_cas
            
        phi_f, phi_v = 0.90, 0.90
        MRd = phi_f * (Z_bar * fy + Z_cas * fy_cas)
        VRd = phi_v * (Vn_bar + Vn_cas)
        
        return MRd, VRd

# --- FUNCIÓN VISUAL ESTRATIGRAFÍA (Estilo Profesional) ---
def plot_estratigrafia_visual(df_layers, d_perf, l_micro):
    fig, ax = plt.subplots(figsize=(4, 6))
    
    estilos = {
        "Arcilla": {"c": "#D2B48C", "h": "///"}, 
        "Arena":   {"c": "#F4A460", "h": "..."},
        "Roca":    {"c": "#A9A9A9", "h": "x"},   
        "Relleno": {"c": "#D3D3D3", "h": "*"},
        "Limo":    {"c": "#EEE8AA", "h": "-"}
    }
    
    max_w = 4
    
    for _, row in df_layers.iterrows():
        h = row['z_bot'] - row['z_top']
        tipo = row['tipo']
        # Buscar estilo, si no encuentra usa default
        st_dict = next((v for k, v in estilos.items() if k in tipo), {"c": "white", "h": ""})
        
        rect = patches.Rectangle((0, row['z_top']), max_w, h, 
                                 linewidth=1, edgecolor='black', 
                                 facecolor=st_dict['c'], hatch=st_dict['h'], alpha=0.6)
        ax.add_patch(rect)
        
        ax.text(max_w/2, row['z_top'] + h/2, 
                f"{tipo}\n$\\alpha$={row['alpha_bond']} kPa\n$k_h$={row['kh']/1000:.0f} MPa/m", 
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
    # Micropilote
    rect_micro = patches.Rectangle((max_w/2 - 0.2, 0), 0.4, l_micro,
                                   linewidth=1.5, edgecolor='#2F4F4F', facecolor='#4682B4')
    ax.add_patch(rect_micro)
    
    ax.set_ylim(max(df_layers['z_bot'].max(), l_micro) + 1, -1) # Invertido
    ax.set_xlim(0, max_w)
    ax.axis('off')
    ax.set_title(f"Esquema (L={l_micro}m)", fontsize=10)
    return fig

# ==============================================================================
# 3. INTERFAZ DE USUARIO STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Diseño Micropilotes Avanzado", layout="wide")

st.title("🛡️ Diseño de Micropilotes - Sistema Dywidag & FHWA")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Sistema Estructural")
    
    # Base de Datos
    df_sys = get_systems_database()
    sys_list = df_sys['Sistema'].tolist()
    
    sel_sys = st.selectbox("Seleccione Barra/Tubo:", sys_list, index=3) # Default R38
    
    # Extraer datos
    row_sys = df_sys[df_sys['Sistema'] == sel_sys].iloc[0]
    st.success(f"**{sel_sys}**\n\n$f_y$: {row_sys['fy_MPa']} MPa\nArea: {row_sys['As_mm2']} mm²")
    
    fc_grout = st.number_input("f'c Grout (MPa)", 20.0, 60.0, 30.0)
    
    st.divider()
    st.header("2. Casing Permanente")
    usar_casing = st.checkbox("Incluir Casing")
    
    casing_params = {'usar': False, 'D_ext': 0, 't': 0, 'fy': 0}
    if usar_casing:
        d_cas = st.number_input("Ø Ext. (mm)", value=178.0)
        t_cas = st.number_input("Espesor (mm)", value=12.7)
        fy_cas = st.number_input("fy Casing (MPa)", value=240.0)
        casing_params = {
            'usar': True, 'D_ext': d_cas/1000, 't': t_cas/1000, 'fy': fy_cas
        }

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["🏗️ Diseño Geotécnico & Estructural", "📊 Verificación Lateral", "🧱 Diseño de Losa"])

# ==============================================================================
# TAB 1: INPUTS Y RESULTADOS PRINCIPALES
# ==============================================================================
with tab1:
    col_input, col_vis = st.columns([1.5, 1])
    
    with col_input:
        st.subheader("Geometría y Cargas")
        c1, c2, c3 = st.columns(3)
        L_tot = c1.number_input("Longitud (m)", 1.0, 60.0, 12.0)
        D_perf = c2.number_input("Ø Perforación (m)", 0.1, 0.6, 0.20)
        FS_geo = c3.number_input("FS Geotécnico", 1.0, 4.0, 2.0)
        
        cc1, cc2, cc3 = st.columns(3)
        P_load = cc1.number_input("Carga Compresión (kN)", value=450.0)
        V_load = cc2.number_input("Carga Lateral (kN)", value=30.0)
        M_load = cc3.number_input("Momento Cabeza (kNm)", value=10.0)
        
        st.markdown("---")
        st.subheader("Estratigrafía del Suelo")
        
        # Editor de datos tipo Excel
        default_data = pd.DataFrame([
            {"z_top": 0.0, "z_bot": 4.0, "tipo": "Arcilla", "alpha_bond": 40, "kh": 5000},
            {"z_top": 4.0, "z_bot": 10.0, "tipo": "Arena", "alpha_bond": 90, "kh": 18000},
            {"z_top": 10.0, "z_bot": 15.0, "tipo": "Roca", "alpha_bond": 250, "kh": 50000}
        ])
        
        st.info("Edite la tabla de suelos abajo. 'kh' es la rigidez horizontal (kN/m³) para el análisis lateral.")
        edited_df = st.data_editor(default_data, num_rows="dynamic")
        
    with col_vis:
        # Gráfica Visual de Capas (La que te gustó)
        if not edited_df.empty:
            fig_soil = plot_estratigrafia_visual(edited_df, D_perf, L_tot)
            st.pyplot(fig_soil)

    # --- CÁLCULO CENTRAL ---
    if not edited_df.empty:
        # 1. Crear objetos SoilLayer
        layers_objs = []
        for _, r in edited_df.iterrows():
            layers_objs.append(SoilLayer(r['z_top'], r['z_bot'], r['tipo'], r['alpha_bond'], r['kh']))
            
        # 2. Configurar Analizador
        mp_params = {
            'L_total': L_tot, 'D_perforacion': D_perf, 'fc_grout': fc_grout,
            'As_bar_mm2': row_sys['As_mm2'], 'fy_bar_MPa': row_sys['fy_MPa'],
            'casing': casing_params
        }
        loads = {'FS_geo': FS_geo, 'V_lat': V_load, 'M_top': M_load}
        
        analyzer = MicropileAnalyzer(mp_params, layers_objs, loads)
        
        # 3. Ejecutar Cálculos
        geo_res = analyzer.calc_geo()
        str_res = analyzer.calc_structural()
        lat_res = analyzer.calc_lateral()
        MRd, VRd = analyzer.calc_limits()
        
        st.divider()
        st.header("Resultados de Capacidad Axial")
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Q Admisible (Geo)", f"{geo_res['Q_adm_tot']:.1f} kN", 
                  delta=f"FS Real: {geo_res['Q_ult_tot']/P_load:.2f}" if P_load>0 else None)
        
        k2.metric("P Compresión (Est)", f"{str_res['P_adm_comp']:.1f} kN",
                  delta="Controla Diseño" if str_res['P_adm_comp'] < geo_res['Q_adm_tot'] else None, delta_color="inverse")
        
        k3.metric("P Tensión (Est)", f"{str_res['P_adm_tens']:.1f} kN")

        # --- GRÁFICAS DE COMPORTAMIENTO (GRIDSPEC RESTAURADO) ---
        st.subheader("Gráficas de Comportamiento Detallado")
        
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 4) # 4 Paneles como pediste

        # Panel 0: Alpha Bond
        ax0 = plt.subplot(gs[0])
        ax0.plot(geo_res['alpha'], geo_res['z'], 'brown', linewidth=2)
        ax0.set_title(r'$\alpha_{bond}$ (kPa)')
        ax0.set_ylabel('Profundidad (m)')
        ax0.set_ylim(L_tot, 0) # Invertir
        ax0.grid(True, alpha=0.3)

        # Panel 1: Capacidad Axial Acumulada
        ax1 = plt.subplot(gs[1], sharey=ax0)
        ax1.plot(geo_res['Q_adm'], geo_res['z'], 'b--', label='Q adm')
        ax1.axvline(P_load, color='r', linestyle=':', label='Carga Pu')
        ax1.set_title('Capacidad (kN)')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Deflexión Lateral
        ax2 = plt.subplot(gs[2], sharey=ax0)
        ax2.plot(lat_res['def'], lat_res['z'], 'm', linewidth=2)
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.set_title('Deflexión (mm)')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Momento Flector
        ax3 = plt.subplot(gs[3], sharey=ax0)
        ax3.plot(lat_res['mom'], lat_res['z'], 'g', linewidth=2)
        ax3.axvline(0, color='k', linewidth=0.5)
        # Limites estructurales graficos
        ax3.axvline(MRd, color='r', linestyle='--', alpha=0.5, label='MRd')
        ax3.axvline(-MRd, color='r', linestyle='--', alpha=0.5)
        ax3.set_title('Momento (kNm)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

# ==============================================================================
# TAB 2: VERIFICACIÓN LATERAL (LRFD)
# ==============================================================================
with tab2:
    if not edited_df.empty:
        st.subheader("Verificación LRFD (AISC / AASHTO)")
        
        col_res1, col_res2 = st.columns(2)
        
        max_mom = np.max(np.abs(lat_res['mom']))
        max_shr = np.max(np.abs(lat_res['shear']))
        
        with col_res1:
            st.write(f"**Momento Máximo Actuante:** {max_mom:.2f} kNm")
            st.write(f"**Momento Resistente ($M_{{Rd}}$):** {MRd:.2f} kNm")
            ratio_m = max_mom / MRd if MRd > 0 else 999
            if ratio_m < 1.0: st.success(f"✅ OK Flexión (DCR = {ratio_m:.2f})")
            else: st.error(f"❌ FALLA Flexión (DCR = {ratio_m:.2f})")
            
        with col_res2:
            st.write(f"**Cortante Máximo Actuante:** {max_shr:.2f} kN")
            st.write(f"**Cortante Resistente ($V_{{Rd}}$):** {VRd:.2f} kN")
            ratio_v = max_shr / VRd if VRd > 0 else 999
            if ratio_v < 1.0: st.success(f"✅ OK Cortante (DCR = {ratio_v:.2f})")
            else: st.error(f"❌ FALLA Cortante (DCR = {ratio_v:.2f})")
            
        st.markdown("---")
        st.info(f"**Rigidez Flexionante Calculada ($EI_{{eff}}$):** {lat_res['EI']/1e9:.3f} MN·m²")
        st.latex(r"M_{Rd} = \phi_f [Z_{bar}f_y + Z_{casing}f_{yc}]")

# ==============================================================================
# TAB 3: DISEÑO DE LOSA
# ==============================================================================
with tab3:
    if not edited_df.empty:
        st.subheader("Distribución en Planta")
        
        sl1, sl2 = st.columns(2)
        L_losa = sl1.number_input("Largo Losa (m)", value=15.0)
        B_losa = sl2.number_input("Ancho Losa (m)", value=10.0)
        q_losa = st.number_input("Carga Sobrecarga (ton/m²)", value=5.0)
        
        # Peso propio losa aprox (0.5m espesor * 24)
        w_losa = 0.5 * 24 
        q_total = (q_losa * 9.81) + (w_losa * 9.81) # kPa
        
        F_total = q_total * L_losa * B_losa
        Q_unit = geo_res['Q_adm_tot']
        
        st.write(f"Carga Total en Losa: **{F_total:.1f} kN**")
        
        if Q_unit > 0:
            N_req = int(np.ceil(F_total / Q_unit))
            st.metric("Micropilotes Requeridos", f"{N_req} unds")
            
            # Grid
            nx = int(np.sqrt(N_req * L_losa/B_losa))
            ny = int(np.ceil(N_req/nx))
            
            sx = L_losa / nx
            sy = B_losa / ny
            
            st.write(f"Configuración sugerida: **{nx} x {ny}** ({nx*ny} total)")
            st.write(f"Separación aprox: {sx:.2f}m x {sy:.2f}m")
            
            # Plot Planta
            fig_p, ax_p = plt.subplots(figsize=(6, 4))
            x = np.linspace(sx/2, L_losa-sx/2, nx)
            y = np.linspace(sy/2, B_losa-sy/2, ny)
            X, Y = np.meshgrid(x, y)
            ax_p.scatter(X, Y, c='black', marker='o')
            ax_p.set_xlim(0, L_losa); ax_p.set_ylim(0, B_losa)
            ax_p.set_aspect('equal')
            ax_p.grid(True, linestyle='--')
            ax_p.set_title("Vista en Planta")
            st.pyplot(fig_p)
        if ratio < 1.0:
            st.success(f"✅ OK (Ratio: {ratio:.2f})")
        else:
            st.error(f"❌ FALLA (Ratio: {ratio:.2f})")

