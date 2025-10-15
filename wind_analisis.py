import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CSS Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    h1 {
        color: #2196F3 !important;
        font-size: 40px !important;
        text-align: center;
        font-weight: bold;
    }
    .stSelectbox label, .stNumberInput label {
        color: #E0E0E0 !important;
        font-weight: bold;
    }   
    div.stButton > button:first-child {
        background-color: #1565C0;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #0D47A1;
        color: white;
    }
    .stSuccess {
        background-color: #1B5E20;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    .stMarkdown p, .css-16huue1, .css-10trblm, .css-1offfwp {
        color: #E0E0E0 !important;
    }
    .stFileUploader label {
        color: #E0E0E0 !important;
        font-weight: bold;
    }
    .stRadio label {
        color: #E0E0E0 !important;
        font-weight: bold;
    }
    .stRadio div[role="radiogroup"] > label, 
    .stRadio div[role="radiogroup"] > div > label,
    .stRadio div[role="radiogroup"] p {
        color: #E0E0E0 !important;
    }
    .stFileUploader {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 8px;
    }
    div[data-testid="stCheckbox"] label p {
        color: #E0E0E0 !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Wind Load Calculation Function (BS EN 1991.1.4) ---
def calculate_wind_load(H, omega, g, rho_air, Ax=303.3, Ay=592.5, z0=0.01, c_dir=1, c_season=1, c0=1, cp=1.2):
    """
    Calculate wind load according to BS EN 1991.1.4
    
    Parameters:
    H: Total height [m]
    omega: Angular velocity [rad/s]
    g: Gravity [m/s²]
    rho_air: Air density [kg/m³]
    Ax: Cross-sectional area in x-direction [m²]
    Ay: Cross-sectional area in y-direction [m²]
    z0: Roughness length [m]
    c_dir: Directional factor
    c_season: Season factor
    c0: Orography factor
    cp: Pressure coefficient
    
    Returns:
    dict with results including heights, wind loads, and velocities
    """
    # Height array
    z = np.arange(1, H + 1)
    
    # Basic wind velocity (convert from your input if needed)
    # Using default v_b0 = 100/3.6 m/s (from original MATLAB)
    v_b0 = 100 / 3.6
    
    # Basic wind velocity with factors
    vb = c_dir * c_season * v_b0
    
    # Terrain factor
    kr = 0.19 * (z0 / 0.05) ** 0.07
    
    # Roughness factor
    cr = kr * np.log(z / z0)
    
    # Mean wind velocity
    vm = cr * c0 * vb
    vm_max = vm[-1]
    
    # Turbulence intensity
    kl = 1
    Iv = kl / (c0 * np.log(z / z0))
    
    # Peak velocity pressure
    q_p = 0.5 * rho_air * (vm ** 2) * (1 + 7 * Iv)
    
    # Wind loads
    Fwy = q_p * cp * Ay / 1e3  # kN (y-direction)
    Fwx = q_p * cp * Ax / 1e3  # kN (x-direction)
    
    return {
        'z': z,
        'vm': vm,
        'vm_max': vm_max,
        'Fwy': Fwy,
        'Fwx': Fwx,
        'q_p': q_p,
        'Iv': Iv,
        'omega': omega,
        'g': g
    }


# --- Read CSV/TXT file for height data ---
def read_height_file(uploaded_file):
    """
    Read a CSV/TXT file containing height data.
    Expected format: single column with heights in meters
    """
    try:
        df = pd.read_csv(uploaded_file, header=None, names=['height'])
        return df['height'].values
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# --- Title ---
st.title("Wind Load Calculator (BS EN 1991.1.4) 🌬️")

st.write("""
This application calculates wind loads on structures according to **BS EN 1991.1.4** 
(Eurocode 1: Actions on structures - Wind actions).
""")

# --- Mode Selection ---
mode = st.radio("Select Input Mode", ["Manual Input", "Upload Dataset"])

# --- Manual Mode ---
if mode == "Manual Input":
    st.subheader("📊 Manual Parameter Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Parameters**")
        H = st.number_input("Total Height (H) [m]", value=66.7, min_value=1.0, step=1.0)
        omega_rpm = st.number_input("Angular Velocity (ω) [RPM]", value=1.0, min_value=0.0, step=0.1)
        omega = omega_rpm * 2 * np.pi / 60  # Convert RPM to rad/s
        st.info(f"ω = {omega:.4f} rad/s")
    
    with col2:
        st.write("**Environmental Parameters**")
        g = st.number_input("Gravity (g) [m/s²]", value=9.81, min_value=0.1, step=0.01)
        rho_air = st.number_input("Air Density (ρ) [kg/m³]", value=1.225, min_value=0.1, step=0.001, format="%.3f")
    
    if st.button("Calculate Wind Load"):
        with st.spinner("Calculating..."):
            results = calculate_wind_load(int(H), omega, g, rho_air)
            
            st.success("✅ Calculation Complete!")
            
            # Display key results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
            with col2:
                st.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
            with col3:
                st.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")
            with col4:
                st.metric("ω (rad/s)", f"{omega:.4f}")
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#121212')
            
            # Wind Load Plot
            ax1.plot(results['Fwy'], results['z'], label='Y-direction', color='cyan', linewidth=2)
            ax1.plot(results['Fwx'], results['z'], label='X-direction', color='orange', linewidth=2)
            ax1.set_xlabel('Wind Load [kN]', color='white', fontsize=11)
            ax1.set_ylabel('Height [m]', color='white', fontsize=11)
            ax1.set_title('Wind Load Distribution', color='white', fontsize=13, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#1e1e1e')
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_color('white')
            
            # Wind Velocity Plot
            ax2.plot(results['vm'], results['z'], label='Mean Wind Velocity', color='green', linewidth=2)
            ax2.set_xlabel('Wind Velocity [m/s]', color='white', fontsize=11)
            ax2.set_ylabel('Height [m]', color='white', fontsize=11)
            ax2.set_title('Wind Velocity Profile', color='white', fontsize=13, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#1e1e1e')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Data table
            if st.checkbox("Show Detailed Results Table", value=False):
                df_results = pd.DataFrame({
                    'Height [m]': results['z'],
                    'Wind Velocity [m/s]': results['vm'],
                    'Wind Load X [kN]': results['Fwx'],
                    'Wind Load Y [kN]': results['Fwy'],
                    'Peak Pressure [Pa]': results['q_p'],
                    'Turbulence Intensity': results['Iv']
                })
                st.dataframe(df_results)

# --- Dataset Mode ---
else:
    st.subheader("📁 Upload Height Dataset")
    
    st.write("""
    Upload a CSV or TXT file containing height values (one value per line).
    The calculator will compute wind loads at each specified height.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Parameters**")
        omega_rpm = st.number_input("Angular Velocity (ω) [RPM]", value=1.0, min_value=0.0, step=0.1, key="omega_upload")
        omega = omega_rpm * 2 * np.pi / 60
        st.info(f"ω = {omega:.4f} rad/s")
    
    with col2:
        st.write("**Environmental Parameters**")
        g = st.number_input("Gravity (g) [m/s²]", value=9.81, min_value=0.1, step=0.01, key="g_upload")
        rho_air = st.number_input("Air Density (ρ) [kg/m³]", value=1.225, min_value=0.1, step=0.001, format="%.3f", key="rho_upload")
    
    uploaded_file = st.file_uploader("Upload Height Data (CSV/TXT)", type=["csv", "txt"])
    
    if uploaded_file is not None:
        heights = read_height_file(uploaded_file)
        
        if heights is not None and len(heights) > 0:
            st.success(f"✅ Loaded {len(heights)} height values")
            st.write(f"Height range: {heights.min():.2f} m to {heights.max():.2f} m")
            
            # Preview data
            st.write("### Data Preview")
            st.write(pd.DataFrame({'Height [m]': heights[:10]}))
            
            if st.checkbox("Show All Heights", value=False):
                st.dataframe(pd.DataFrame({'Height [m]': heights}))
            
            if st.button("Calculate Wind Load from Dataset"):
                with st.spinner("Calculating wind loads..."):
                    H_max = int(np.ceil(heights.max()))
                    results = calculate_wind_load(H_max, omega, g, rho_air)
                    
                    # Interpolate results for uploaded heights
                    Fwx_interp = np.interp(heights, results['z'], results['Fwx'])
                    Fwy_interp = np.interp(heights, results['z'], results['Fwy'])
                    vm_interp = np.interp(heights, results['z'], results['vm'])
                    
                    st.success("✅ Calculation Complete!")
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Max Velocity", f"{vm_interp.max():.2f} m/s")
                    with col2:
                        st.metric("Max Load (X)", f"{Fwx_interp.max():.2f} kN")
                    with col3:
                        st.metric("Max Load (Y)", f"{Fwy_interp.max():.2f} kN")
                    with col4:
                        st.metric("ω (rad/s)", f"{omega:.4f}")
                    
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.patch.set_facecolor('#121212')
                    
                    # Wind Load Plot
                    ax1.scatter(Fwy_interp, heights, label='Y-direction', color='cyan', s=20, alpha=0.6)
                    ax1.scatter(Fwx_interp, heights, label='X-direction', color='orange', s=20, alpha=0.6)
                    ax1.set_xlabel('Wind Load [kN]', color='white', fontsize=11)
                    ax1.set_ylabel('Height [m]', color='white', fontsize=11)
                    ax1.set_title('Wind Load Distribution (Uploaded Data)', color='white', fontsize=13, fontweight='bold')
                    ax1.legend(loc='best')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_facecolor('#1e1e1e')
                    ax1.tick_params(colors='white')
                    for spine in ax1.spines.values():
                        spine.set_color('white')
                    
                    # Wind Velocity Plot
                    ax2.scatter(vm_interp, heights, label='Mean Wind Velocity', color='green', s=20, alpha=0.6)
                    ax2.set_xlabel('Wind Velocity [m/s]', color='white', fontsize=11)
                    ax2.set_ylabel('Height [m]', color='white', fontsize=11)
                    ax2.set_title('Wind Velocity Profile (Uploaded Data)', color='white', fontsize=13, fontweight='bold')
                    ax2.legend(loc='best')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_facecolor('#1e1e1e')
                    ax2.tick_params(colors='white')
                    for spine in ax2.spines.values():
                        spine.set_color('white')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Results table
                    if st.checkbox("Show Results Table", value=False):
                        df_output = pd.DataFrame({
                            'Height [m]': heights,
                            'Wind Velocity [m/s]': vm_interp,
                            'Wind Load X [kN]': Fwx_interp,
                            'Wind Load Y [kN]': Fwy_interp
                        })
                        st.write("### Calculation Results")
                        st.dataframe(df_output)

# --- Information Section ---
with st.expander("ℹ️ About Input Parameters"):
    st.write("""
    **Input Parameters:**
    - **H**: Total height of the structure [m]
    - **ω (omega)**: Angular velocity in RPM (converted to rad/s)
    - **g**: Gravitational acceleration [m/s²] (standard: 9.81)
    - **ρ (rho_air)**: Air density [kg/m³] (standard at sea level: 1.225)
    
    **About BS EN 1991.1.4:**
    This standard provides methods for calculating wind loads on buildings and structures.
    
    **Terrain Categories (typical z0 values):**
    - Open sea, lakes: z0 = 0.003 m
    - Flat terrain with obstacles: z0 = 0.01 m
    - Suburban/industrial: z0 = 0.3 m
    - Urban areas: z0 = 1.0 m
    """)