import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --- CSS Styling (Light Theme) ---
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1 {
        color: #1976D2 !important;
        font-size: 40px !important;
        text-align: center;
        font-weight: bold;
    }
    .stSelectbox label, .stNumberInput label {
        color: #000000 !important;
        font-weight: bold;
    }   
    div.stButton > button:first-child {
        background-color: #2196F3;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #1976D2;
        color: white;
    }
    .stSuccess {
        background-color: #C8E6C9;
        color: #1B5E20;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    .stMarkdown p, .css-16huue1, .css-10trblm, .css-1offfwp {
        color: #000000 !important;
    }
    .stFileUploader label {
        color: #000000 !important;
        font-weight: bold;
    }
    .stRadio label {
        color: #000000 !important;
        font-weight: bold;
    }
    .stFileUploader {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 8px;
    }
    div[data-testid="stCheckbox"] label p {
        color: #000000 !important;
        font-weight: bold !important;
    }
    .stMetric {
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Wind Load Calculation Function ---
def calculate_wind_load(H, omega, g, rho_air, Ax=303.3, Ay=592.5, z0=0.01, c_dir=1, c_season=1, c0=1, cp=1.2):
    z = np.arange(1, H + 1)
    v_b0 = 100 / 3.6
    vb = c_dir * c_season * v_b0
    kr = 0.19 * (z0 / 0.05) ** 0.07
    cr = kr * np.log(z / z0)
    vm = cr * c0 * vb
    vm_max = vm[-1]
    kl = 1
    Iv = kl / (c0 * np.log(z / z0))
    q_p = 0.5 * rho_air * (vm ** 2) * (1 + 7 * Iv)
    Fwy = q_p * cp * Ay / 1e3
    Fwx = q_p * cp * Ax / 1e3
    return {'z': z, 'vm': vm, 'vm_max': vm_max, 'Fwy': Fwy, 'Fwx': Fwx, 'q_p': q_p, 'Iv': Iv}


# --- File Reader (H, g, rho_air, omega) ---
def read_parameter_file(uploaded_file):
    """
    Reads a file with 4 comma-separated values:
    H, g, rho_air, omega_rpm
    Example: 66.7,9.81,1.225,2
    """
    try:
        data = np.loadtxt(uploaded_file, delimiter=",")
        if len(data) != 4:
            st.error("❌ File must contain exactly 4 numeric values: H, g, rho_air, omega_rpm")
            return None
        H, g, rho_air, omega_rpm = data
        return float(H), float(g), float(rho_air), float(omega_rpm)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# --- Interactive Plotly Charts ---
def create_interactive_plots(results):
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Wind Load Distribution', 'Wind Velocity Profile'),
        horizontal_spacing=0.12
    )
    
    # Plot 1: Wind Loads
    fig.add_trace(
        go.Scatter(
            x=results['Fwy'], 
            y=results['z'], 
            mode='lines',
            name='Y-direction',
            line=dict(color='#00BCD4', width=3),
            hovertemplate='Load: %{x:.2f} kN<br>Height: %{y} m<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['Fwx'], 
            y=results['z'], 
            mode='lines',
            name='X-direction',
            line=dict(color='#FF9800', width=3),
            hovertemplate='Load: %{x:.2f} kN<br>Height: %{y} m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Wind Velocity
    fig.add_trace(
        go.Scatter(
            x=results['vm'], 
            y=results['z'], 
            mode='lines',
            name='Mean Wind Velocity',
            line=dict(color='#4CAF50', width=3),
            hovertemplate='Velocity: %{x:.2f} m/s<br>Height: %{y} m<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Wind Load [kN]", row=1, col=1, gridcolor='#E0E0E0')
    fig.update_xaxes(title_text="Wind Velocity [m/s]", row=1, col=2, gridcolor='#E0E0E0')
    fig.update_yaxes(title_text="Height [m]", row=1, col=1, gridcolor='#E0E0E0')
    fig.update_yaxes(title_text="Height [m]", row=1, col=2, gridcolor='#E0E0E0')
    
    fig.update_layout(
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', size=12),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#BDBDBD',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig


# --- App UI ---
st.title("Wind Load Calculator (BS EN 1991.1.4) 🌬️")

mode = st.radio("Select Input Mode", ["Manual Input", "Upload Parameters File"])

# --- Manual Input ---
if mode == "Manual Input":
    st.subheader("📊 Manual Parameter Input")

    col1, col2 = st.columns(2)
    with col1:
        H = st.number_input("Total Height (H) [m]", value=66.7, min_value=1.0)
        omega_rpm = st.number_input("Angular Velocity (ω) [RPM]", value=2.0, min_value=0.0)
        omega = omega_rpm * 2 * np.pi / 60
        st.info(f"ω = {omega:.4f} rad/s")
    with col2:
        g = st.number_input("Gravity (g) [m/s²]", value=9.81)
        rho_air = st.number_input("Air Density (ρ) [kg/m³]", value=1.225, format="%.3f")

    if st.button("Calculate Wind Load"):
        with st.spinner("Calculating..."):
            results = calculate_wind_load(int(H), omega, g, rho_air)
            st.success("✅ Calculation Complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
            col2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
            col3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")

            # Interactive Plotly Charts
            fig = create_interactive_plots(results)
            st.plotly_chart(fig, use_container_width=True)

# --- Upload Parameters File ---
else:
    st.subheader("📁 Upload Parameter File")
    st.write("File format: `H, g, rho_air, omega_rpm` (comma-separated, one line)")

    uploaded_file = st.file_uploader("Upload your .csv or .txt file", type=["csv", "txt"])
    if uploaded_file:
        params = read_parameter_file(uploaded_file)
        if params:
            H, g, rho_air, omega_rpm = params
            omega = omega_rpm * 2 * np.pi / 60

            st.info(f"✅ Loaded parameters: H={H} m, g={g}, ρ={rho_air}, ω={omega_rpm} RPM")
            if st.button("Calculate Wind Load"):
                with st.spinner("Calculating..."):
                    results = calculate_wind_load(int(H), omega, g, rho_air)
                    st.success("✅ Calculation Complete!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
                    col2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
                    col3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")

                    # Interactive Plotly Charts
                    fig = create_interactive_plots(results)
                    st.plotly_chart(fig, use_container_width=True)

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
