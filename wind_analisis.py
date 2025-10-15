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

            # === Plots ===
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#121212')

            # Plot 1: Wind Loads
            ax1.plot(results['Fwy'], results['z'], color='cyan', label='Y-direction', linewidth=2)
            ax1.plot(results['Fwx'], results['z'], color='orange', label='X-direction', linewidth=2)
            ax1.set_xlabel('Wind Load [kN]', color='white')
            ax1.set_ylabel('Height [m]', color='white')
            ax1.set_title('Wind Load Distribution', color='white', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#1e1e1e')
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_color('white')

            # Plot 2: Wind Velocity
            ax2.plot(results['vm'], results['z'], color='lime', label='Mean Wind Velocity', linewidth=2)
            ax2.set_xlabel('Wind Velocity [m/s]', color='white')
            ax2.set_ylabel('Height [m]', color='white')
            ax2.set_title('Wind Velocity Profile', color='white', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#1e1e1e')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_color('white')

            # ✅ This line fixes black axis numbers:
            for ax in [ax1, ax2]:
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_color("white")

            plt.tight_layout()
            st.pyplot(fig)

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

                    # === Plots ===
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.patch.set_facecolor('#121212')

                    ax1.plot(results['Fwy'], results['z'], color='cyan', label='Y-direction', linewidth=2)
                    ax1.plot(results['Fwx'], results['z'], color='orange', label='X-direction', linewidth=2)
                    ax1.set_xlabel('Wind Load [kN]', color='white')
                    ax1.set_ylabel('Height [m]', color='white')
                    ax1.set_title('Wind Load Distribution', color='white', fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_facecolor('#1e1e1e')
                    ax1.tick_params(colors='white')
                    for spine in ax1.spines.values():
                        spine.set_color('white')

                    ax2.plot(results['vm'], results['z'], color='lime', label='Mean Wind Velocity', linewidth=2)
                    ax2.set_xlabel('Wind Velocity [m/s]', color='white')
                    ax2.set_ylabel('Height [m]', color='white')
                    ax2.set_title('Wind Velocity Profile', color='white', fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_facecolor('#1e1e1e')
                    ax2.tick_params(colors='white')
                    for spine in ax2.spines.values():
                        spine.set_color('white')

                    # ✅ Fix axis labels color again:
                    for ax in [ax1, ax2]:
                        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                            label.set_color("white")

                    plt.tight_layout()
                    st.pyplot(fig)
