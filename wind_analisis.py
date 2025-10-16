import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Wind Load Calculator (BS EN 1991.1.4)", layout="wide", page_icon="🌬️")

# --- Light CSS Styling (white background, black text) ---
st.markdown("""
<style>
    /* App background and text */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    /* Header */
    h1 {
        color: #0B57A4 !important;
        font-size: 40px !important;
        text-align: center;
        font-weight: bold;
    }

    /* Form labels and text */
    label, .stSelectbox label, .stNumberInput label, .stMarkdown p, .stFileUploader label, .stRadio label {
        color: #000000 !important;
        font-weight: bold;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #0B57A4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #083e6a;
        color: white;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #f7f7f7;
        border-radius: 8px;
        padding: 8px;
    }

    /* Make Plotly charts contained and scrollable when tall */
    .chart-container {
        height: 520px; /* fixed height so user can scroll inside */
        overflow: auto;
        padding: 4px;
        border-radius: 6px;
        background-color: #ffffff;
    }

    /* Ensure metric text is readable */
    .stMetric { color: #000000; }
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

# --- App UI ---
st.title("Wind Load Calculator (BS EN 1991.1.4) 🌬️")

# Default the radio to "Manual Input" (index=0) so the app opens in manual mode.
mode = st.radio("Select Input Mode", ["Manual Input", "Upload Parameters File"], index=0)

# Helper: build interactive Plotly charts (ensure template 'plotly_white' for bright theme)
def build_plots(results):
    z = results['z'].tolist()
    Fwy = results['Fwy'].tolist()
    Fwx = results['Fwx'].tolist()
    vm = results['vm'].tolist()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=Fwy, y=z, mode='lines+markers', name='Y-direction', hovertemplate='Load: %{x:.3f} kN<br>Height: %{y} m'))
    fig1.add_trace(go.Scatter(x=Fwx, y=z, mode='lines+markers', name='X-direction', hovertemplate='Load: %{x:.3f} kN<br>Height: %{y} m'))
    fig1.update_layout(
        title='Wind Load Distribution',
        xaxis_title='Wind Load [kN]',
        yaxis_title='Height [m]',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        hovermode='closest',
        dragmode='pan',
        margin=dict(l=40,r=20,t=60,b=40)
    )
    fig1.update_yaxes(autorange=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vm, y=z, mode='lines+markers', name='Mean Wind Velocity', hovertemplate='Velocity: %{x:.3f} m/s<br>Height: %{y} m'))
    fig2.update_layout(
        title='Wind Velocity Profile',
        xaxis_title='Wind Velocity [m/s]',
        yaxis_title='Height [m]',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        hovermode='closest',
        dragmode='pan',
        margin=dict(l=40,r=20,t=60,b=40)
    )
    fig2.update_yaxes(autorange=True)

    return fig1, fig2

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

            m1, m2, m3 = st.columns(3)
            m1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
            m2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
            m3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")

            fig1, fig2 = build_plots(results)

            # Display plots side-by-side in scrollable containers so the charts remain interactive
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom": True, "responsive": True})
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True, "responsive": True})
                st.markdown('</div>', unsafe_allow_html=True)

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

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
                    m2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
                    m3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")

                    fig1, fig2 = build_plots(results)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom": True, "responsive": True})
                        st.markdown('</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True, "responsive": True})
                        st.markdown('</div>', unsafe_allow_html=True)

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

