# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 17:15:15 2025

@author: Usuario
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# ==============================================================================
# 1. BASE DE DATOS DE SISTEMAS (Extraído de tus archivos CSV)
# ==============================================================================
# He recopilado los sistemas más comunes del archivo "Systems Properties.csv"
def get_systems_database():
    data = {
        "Sistema": [
            "R32-280", "R32-320", "R32-360", "R38-500", "R51-800", 
            "R51-950", "T76-1300", "T76-1900", "GEWI 28", "GEWI 32", 
            "GEWI 40", "GEWI 50", "Ischebeck 30/11", "Ischebeck 40/16"
        ],
        # Yield Strength (MPa) - Aproximados según tu CSV
        "fy_MPa": [
            500, 657, 651, 678, 550, 
            670, 580, 580, 500, 500, 
            500, 500, 540, 500
        ],
        # Área (mm2)
        "As_mm2": [
            560, 481, 430, 735, 1452, 
            1418, 2240, 3277, 616, 804, 
            1257, 1963, 532, 1250
        ],
        # Diámetro Exterior Real aprox (mm)
        "D_ext_mm": [
            32, 32, 32, 38, 51, 
            51, 76, 76, 28, 32, 
            40, 50, 30, 40
        ]
    }
    return pd.DataFrame(data)

# ==============================================================================
# 2. CLASES DE CÁLCULO (Motor Físico)
# ==============================================================================

@dataclass
class SoilLayer:
    z_top: float
    z_bot: float
    alpha_bond: float
    kh: float
    desc: str

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
        if z > self.layers[-1].z_bot: return getattr(self.layers[-1], prop)
        for l in self.layers:
            if l.contains(z): return getattr(l, prop)
        return getattr(self.layers[-1], prop)

    # --- 1. Geotecnia ---
    def calc_geo(self):
        dz = 0.1
        z_arr = np.arange(0, self.L + dz, dz)
        q_accum = []
        curr = 0.0
        alphas = []
        for z in z_arr:
            alpha = self.get_soil_prop(z, 'alpha_bond')
            alphas.append(alpha)
            curr += alpha * self.Perimetro * dz
            q_accum.append(curr)
        
        q_accum = np.array(q_accum)
        return {
            'z': z_arr, 'alpha': np.array(alphas),
            'Q_ult': q_accum, 'Q_adm': q_accum / self.loads['FS_geo'],
            'Q_ult_tot': q_accum[-1], 'Q_adm_tot': q_accum[-1] / self.loads['FS_geo']
        }

    # --- 2. Estructural ---
    def calc_structural(self):
        # Datos del sistema seleccionado
        As_bar = self.mp['As_bar_mm2'] * 1e-6 # mm2 a m2
        fy_bar = self.mp['fy_bar_MPa'] * 1000 # MPa a kPa
        
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
        fc = self.mp['fc_grout'] * 1000
        
        # Compresión Compuesta (FHWA)
        P_grout = 0.40 * fc * Ag_grout
        P_bar = 0.47 * fy_bar * As_bar
        P_casing = 0.47 * fy_casing * As_casing
        
        return {'P_adm': P_grout + P_bar + P_casing, 
                'P_grout': P_grout, 'P_bar': P_bar, 'P_casing': P_casing}

    # --- 3. Lateral (Winkler) ---
    def calc_lateral(self):
        n = 100
        z_nodes = np.linspace(0, self.L, n)
        
        # Rigidez EI
        Es = 200000 * 1000 # kPa
        if self.mp['casing']['usar']:
            De = self.mp['casing']['D_ext']
            Di = De - 2*self.mp['casing']['t']
            I = (np.pi/64)*(De**4 - Di**4)
        else:
            # Barra solida equivalente
            d_eq = np.sqrt(4 * (self.mp['As_bar_mm2']*1e-6) / np.pi)
            I = (np.pi/64)*(d_eq**4)
        
        EI = Es * I
        
        # Beta
        kh_ref = self.get_soil_prop(1.5, 'kh') # a 1.5m de prof
        beta = ((kh_ref * self.D)/(4*EI))**0.25
        
        V0 = self.loads['V_lat']
        M0 = self.loads['M_top']
        
        y_lst, M_lst, V_lst = [], [], []
        
        for z in z_nodes:
            bz = beta * z
            if bz > 15:
                y_lst.append(0); M_lst.append(0); V_lst.append(0)
            else:
                ebz = np.exp(-bz)
                sin, cos = np.sin(bz), np.cos(bz)
                
                A = ebz * (cos + sin)
                B = ebz * sin
                C = ebz * (cos - sin)
                D_ = ebz * cos
                
                # Deflexion (mm)
                y = (2*V0*beta/(kh_ref*self.D))*D_ + (2*M0*beta**2/(kh_ref*self.D))*C
                y_lst.append(y * 1000)
                
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
            Av_cas = Ag_cas / 2.0 # Corrección Tubo
            Vn_cas = 0.6 * fy_cas * Av_cas
            
        phi_f, phi_v = 0.90, 0.90
        MRd = phi_f * (Z_bar * fy + Z_cas * fy_cas)
        VRd = phi_v * (Vn_bar + Vn_cas)
        
        return MRd, VRd

# ==============================================================================
# 3. INTERFAZ STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Diseño Micropilotes", layout="wide")

st.title("🏗️ Herramienta de Diseño de Micropilotes")
st.markdown("Cálculo geotécnico, estructural y lateral con **Sistemas Dywidag/Ischebeck**.")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Configuración del Sistema")
    
    # Selección de Refuerzo (Base de Datos)
    df_sys = get_systems_database()
    
    # Crear etiquetas para el selectbox
    sys_options = df_sys['Sistema'].tolist()
    selected_sys_name = st.selectbox("Seleccione Refuerzo Principal:", sys_options)
    
    # Obtener datos de la fila seleccionada
    row = df_sys[df_sys['Sistema'] == selected_sys_name].iloc[0]
    st.info(f"FY: {row['fy_MPa']} MPa | As: {row['As_mm2']} mm² | Øext: {row['D_ext_mm']} mm")
    
    st.divider()
    
    st.header("2. Geometría")
    L_tot = st.number_input("Longitud Total (m)", value=12.0, step=0.5)
    D_perf = st.number_input("Diámetro Perforación (m)", value=0.20, step=0.01)
    fc_grout = st.number_input("Resistencia Grout (MPa)", value=25.0)
    
    use_casing = st.checkbox("Usar Casing (Tubería)", value=True)
    casing_dict = {'usar': False, 'D_ext':0, 't':0, 'fy':0}
    if use_casing:
        colc1, colc2 = st.columns(2)
        with colc1:
            D_cas_mm = st.number_input("Ø Ext Casing (mm)", value=178.0)
            t_cas_mm = st.number_input("Espesor (mm)", value=9.0)
        with colc2:
            fy_cas = st.number_input("Fy Casing (MPa)", value=350.0)
            L_cas = st.number_input("Longitud Casing (m)", value=6.0) # Solo informativo para nota
        
        casing_dict = {
            'usar': True,
            'D_ext': D_cas_mm/1000,
            't': t_cas_mm/1000,
            'fy': fy_cas
        }

    st.divider()
    st.header("3. Cargas")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        P_comp = st.number_input("Compresión (kN)", value=450.0)
        V_lat = st.number_input("Lateral (kN)", value=35.0)
    with col_l2:
        M_top = st.number_input("Momento (kNm)", value=10.0)
        FS_geo = st.number_input("FS Geotécnico", value=2.0)

# --- PANEL PRINCIPAL: ESTRATIGRAFÍA ---
st.subheader("📍 Perfil Estratigráfico (Editable)")
default_soil = pd.DataFrame([
    {"z_top": 0.0, "z_bot": 4.0, "alpha_bond": 40, "kh": 5000, "desc": "Arcilla Blanda"},
    {"z_top": 4.0, "z_bot": 8.0, "alpha_bond": 90, "kh": 15000, "desc": "Limo Arenoso"},
    {"z_top": 8.0, "z_bot": 15.0, "alpha_bond": 180, "kh": 40000, "desc": "Roca Meteorizada"},
])
edited_soil = st.data_editor(default_soil, num_rows="dynamic")

# --- PROCESAMIENTO ---

# 1. Preparar Objetos
soil_objs = []
for index, r in edited_soil.iterrows():
    soil_objs.append(SoilLayer(r['z_top'], r['z_bot'], r['alpha_bond'], r['kh'], r['desc']))

mp_params = {
    'L_total': L_tot,
    'D_perforacion': D_perf,
    'fc_grout': fc_grout,
    'As_bar_mm2': row['As_mm2'],
    'fy_bar_MPa': row['fy_MPa'],
    'casing': casing_dict
}
loads_dict = {'FS_geo': FS_geo, 'V_lat': V_lat, 'M_top': M_top}

analyzer = MicropileAnalyzer(mp_params, soil_objs, loads_dict)

# 2. Calcular
geo_res = analyzer.calc_geo()
struc_res = analyzer.calc_structural()
lat_res = analyzer.calc_lateral()
MRd, VRd = analyzer.calc_limits()

# --- RESULTADOS VISUALES ---

st.divider()
st.header("📊 Resultados del Análisis")

# Métricas Principales
m1, m2, m3, m4 = st.columns(4)
m1.metric("Q admisible (Geo)", f"{geo_res['Q_adm_tot']:.1f} kN", 
          delta=f"{geo_res['Q_adm_tot'] - P_comp:.1f} vs Carga", 
          delta_color="normal")
m2.metric("P admisible (Est)", f"{struc_res['P_adm']:.1f} kN",
          delta=f"Grout: {struc_res['P_grout']:.0f} kN")
m3.metric("Deflexión Máx", f"{np.max(np.abs(lat_res['def'])):.2f} mm",
          delta="-25mm Límite", delta_color="inverse")
m4.metric("DCR Flexión", f"{(np.max(np.abs(lat_res['mom']))/MRd):.2f}",
          help="Relación Demanda / Capacidad (debe ser < 1.0)")

# Pestañas
tab1, tab2, tab3 = st.tabs(["Gráficas de Comportamiento", "Verificación Estructural", "Diseño de Losa"])

with tab1:
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 4)
    
    # 1. Suelo
    ax0 = plt.subplot(gs[0])
    ax0.plot(geo_res['alpha'], geo_res['z'], 'brown')
    ax0.set_title(r'$\alpha_{bond}$ (kPa)')
    ax0.set_ylabel('Profundidad (m)')
    ax0.invert_yaxis(); ax0.grid(True, alpha=0.5)
    
    # 2. Axial
    ax1 = plt.subplot(gs[1], sharey=ax0)
    ax1.plot(geo_res['Q_adm'], geo_res['z'], 'b--', label='Q adm')
    ax1.axvline(P_comp, color='r', ls=':', label='Carga')
    ax1.set_title('Capacidad (kN)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.5)
    
    # 3. Deflexion
    ax2 = plt.subplot(gs[2], sharey=ax0)
    ax2.plot(lat_res['def'], lat_res['z'], 'm')
    ax2.set_title('Deflexión (mm)')
    ax2.grid(True, alpha=0.5)
    
    # 4. Momento
    ax3 = plt.subplot(gs[3], sharey=ax0)
    ax3.plot(lat_res['mom'], lat_res['z'], 'g')
    ax3.set_title('Momento (kNm)')
    ax3.grid(True, alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    col_v1, col_v2 = st.columns([1, 2])
    
    with col_v1:
        st.markdown("### Límites Calculados (LRFD)")
        st.write(f"**Momento Resistente ($M_{{Rd}}$):** {MRd:.2f} kNm")
        st.write(f"**Cortante Resistente ($V_{{Rd}}$):** {VRd:.2f} kN")
        st.info("Nota: Cortante corregido para secciones tubulares ($A_v = A_g/2$) según AISC.")
        
        st.markdown("### Ecuaciones Usadas")
        st.latex(r"M_{Rd} = 0.9 [Z_{bar}f_y + Z_{casing}f_{y,c}]")
        st.latex(r"V_{Rd} = 0.9 [0.6 f_y A_s + 0.6 f_{y,c} (A_{g,c}/2)]")

    with col_v2:
        # Gráfica Comparativa
        fig2, (ax_m, ax_v) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        
        # Momento
        ax_m.plot(lat_res['mom'], lat_res['z'], 'b', label='Actuante')
        ax_m.axvline(MRd, c='r', ls='--', label='MRd')
        ax_m.axvline(-MRd, c='r', ls='--')
        ax_m.set_title("Flexión (kNm)"); ax_m.invert_yaxis(); ax_m.grid(True); ax_m.legend()
        
        # Cortante
        ax_v.plot(lat_res['shear'], lat_res['z'], 'k', label='Actuante')
        ax_v.axvline(VRd, c='orange', ls='--', label='VRd')
        ax_v.axvline(-VRd, c='orange', ls='--')
        ax_v.set_title("Cortante (kN)"); ax_v.grid(True); ax_v.legend()
        
        st.pyplot(fig2)

with tab3:
    st.markdown("### Distribución en Planta (Losa)")
    c_slab1, c_slab2 = st.columns(2)
    with c_slab1:
        L_losa = st.number_input("Largo Losa (m)", value=20.0)
        B_losa = st.number_input("Ancho Losa (m)", value=20.0)
    with c_slab2:
        q_losa = st.number_input("Carga Distribuida (ton/m²)", value=39.0)
    
    # Calculo Losa
    Q_tot_kN = q_losa * 9.81 * L_losa * B_losa
    Q_unit = geo_res['Q_adm_tot'] # Usar la geotecnica calculada
    
    if Q_unit > 0:
        N_req = int(np.ceil(Q_tot_kN / Q_unit))
        ny = np.sqrt(N_req / (L_losa/B_losa))
        Nx, Ny = int(np.ceil(N_req/ny)), int(np.ceil(ny))
        Sx, Sy = L_losa/Nx, B_losa/Ny
        
        st.success(f"**Requeridos:** {N_req} micropilotes | **Configuración:** {Nx} x {Ny} ({Nx*Ny} total)")
        st.write(f"Separación X: {Sx:.2f} m | Separación Y: {Sy:.2f} m")
        
        # Plot Losa
        fig_slab, ax_s = plt.subplots(figsize=(6, 6))
        x = np.linspace(Sx/2, L_losa-Sx/2, Nx)
        y = np.linspace(Sy/2, B_losa-Sy/2, Ny)
        X, Y = np.meshgrid(x, y)
        ax_s.scatter(X, Y, c='k', marker='o')
        ax_s.set_xlim(0, L_losa); ax_s.set_ylim(0, B_losa)
        ax_s.set_aspect('equal')
        ax_s.set_title(f"Planta {L_losa}x{B_losa}m")
        ax_s.grid(True, ls=':')
        st.pyplot(fig_slab)
    else:
        st.error("La capacidad del micropilote es 0. Revise los datos.")