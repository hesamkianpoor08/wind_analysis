import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CSS Styling (dark theme) ---
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    h1 { color: #2196F3 !important; font-size: 40px !important; text-align: center; font-weight: bold; }
    .stSelectbox label, .stNumberInput label { color: #E0E0E0 !important; font-weight: bold; }
    div.stButton > button:first-child { background-color: #1565C0; color: white; border-radius: 8px; height: 3em; width: 100%; font-size: 16px; border: none; }
    div.stButton > button:first-child:hover { background-color: #0D47A1; color: white; }
    .stSuccess { background-color: #1B5E20; color: white; border-radius: 8px; padding: 10px; font-weight: bold; }
    .stMarkdown p, .css-16huue1, .css-10trblm, .css-1offfwp { color: #E0E0E0 !important; }
    .stFileUploader label { color: #E0E0E0 !important; font-weight: bold; }
    .stRadio label { color: #E0E0E0 !important; font-weight: bold; }
    .stRadio div[role="radiogroup"] > label, .stRadio div[role="radiogroup"] > div > label, .stRadio div[role="radiogroup"] p { color: #E0E0E0 !important; }
    .stFileUploader { background-color: #1e1e1e; border-radius: 8px; padding: 8px; }
    div[data-testid="stCheckbox"] label p { color: #E0E0E0 !important; font-weight: bold !important; }
            /* ---------- Make input numbers/text visible on dark background ---------- */
/* Number / text inputs, textareas and selects */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[role="button"],
div[data-testid="stMultiSelect"] div[role="listbox"] input,
input[type="number"],
input[type="text"],
textarea,
select {
    color: #E0E0E0 !important;           /* text color (light) */
    background-color: #1e1e1e !important;/* input background (dark) */
    border-color: rgba(255,255,255,0.08) !important;
}

/* Metric values (big numbers) and deltas */
.stMetricValue, .stMetricDelta, div[data-testid="metric-container"] {
    color: #E0E0E0 !important;
}

/* Small extra: ensure placeholder text also visible */
input::placeholder, textarea::placeholder {
    color: rgba(224,224,224,0.6) !important;
}

/* Ensure dropdown options text is visible */
div[role="option"] {
    color: #E0E0E0 !important;
    background-color: #1e1e1e !important;
}

</style>
""", unsafe_allow_html=True)


# --- Wind load calculation function ---
def calculate_wind_load(H, omega_rad_s, g, rho_air, Ax=303.3, Ay=592.5, z0=0.01, c_dir=1, c_season=1, c0=1, cp=1.2):
    """
    Compute wind load distribution (simple model inspired by EN1991-1-4).
    Parameters:
      H           : total height in meters (int or float)
      omega_rad_s : angular velocity in rad/s (for record; not strongly used in this simplified model)
      g           : gravity acceleration (m/s^2) (kept for record)
      rho_air     : air density (kg/m^3)
      Ax, Ay      : reference areas in m^2
      z0          : roughness length (m)
      c_dir,c_season,c0,cp : multiplicative factors (unitless)
    Returns:
      dict with arrays and scalars
    """
    H_int = max(1, int(np.ceil(H)))
    z = np.arange(1, H_int + 1)

    # Basic wind velocity (default reference value used when not provided)
    v_b0 = 100.0 / 3.6  # 100 km/h -> m/s, based on original code baseline

    vb = c_dir * c_season * v_b0

    # Terrain/roughness scaling
    kr = 0.19 * (z0 / 0.05) ** 0.07
    z_safe = np.maximum(z, z0 * 1.0001)
    cr = kr * np.log(z_safe / z0)

    # Mean wind velocity profile
    vm = cr * c0 * vb
    vm_max = vm.max() if vm.size > 0 else 0.0

    # Turbulence intensity (simple treatment)
    kl = 1.0  # baseline (you can modify kl with omega if you have a specific model)
    Iv = kl / (c0 * np.log(z_safe / z0))

    # Peak velocity pressure using provided air density
    q_p = 0.5 * float(rho_air) * (vm ** 2) * (1 + 7 * Iv)

    # Wind loads in kN
    Fwy = q_p * cp * Ay / 1e3
    Fwx = q_p * cp * Ax / 1e3

    return {
        'z': z,
        'vm': vm,
        'vm_max': vm_max,
        'Fwy': Fwy,
        'Fwx': Fwx,
        'q_p': q_p,
        'Iv': Iv,
        'omega_rad_s': omega_rad_s,
        'g': g
    }


# --- Robust uploader reader for headerless or headered files ---
def read_upload_file(uploaded_file):
    """
    Read uploaded CSV/TXT and return (params, heights).

    Supported headerless single-line formats:
      1) Four values in one row: H, g, rho_air, omega_rpm
         e.g. 66.7, 9.81, 1.225, 2

    Supported other cases:
      - Single-column headerless list of heights (one per line)
      - Headered CSV with columns including H, omega_rpm, g, rho_air, height, etc.

    Returns:
      params: dict with keys H, omega_rpm, g, rho_air, Ax, Ay, v_b0_kmh, z0, cp
      heights: numpy array of heights or None
    """
    defaults = {
        'H': 66.7,
        'omega_rpm': 1.0,
        'g': 9.81,
        'rho_air': 1.225,
        'Ax': 303.3,
        'Ay': 592.5,
        'v_b0_kmh': 100.0,
        'z0': 0.01,
        'cp': 1.2
    }

    params = defaults.copy()
    heights = None

    # Try headerless read first (works for single-row headerless or single-column heights)
    try:
        df0 = pd.read_csv(uploaded_file, header=None)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

    # Case: headerless single-row with >=4 numeric columns -> interpret as H, g, rho_air, omega_rpm
    if df0.shape[0] == 1 and df0.shape[1] >= 4:
        try:
            row = df0.iloc[0].astype(float).values
            params['H'] = float(row[0])
            params['g'] = float(row[1])
            params['rho_air'] = float(row[2])
            params['omega_rpm'] = float(row[3])
            # optional extras if present (Ax, Ay, v_b0_kmh)
            if df0.shape[1] >= 6:
                try:
                    params['Ax'] = float(row[4])
                    params['Ay'] = float(row[5])
                except Exception:
                    pass
        except Exception:
            # fallback to later parsing
            pass
        return params, None

    # Case: headerless single-column -> treat as heights
    if df0.shape[1] == 1:
        try:
            heights = pd.to_numeric(df0.iloc[:, 0], errors='coerce')
            heights = heights.dropna().values
            return params, heights
        except Exception:
            pass

    # Otherwise try reading with header (standard CSV)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        # If we can't parse headered CSV, return defaults/no heights
        return params, None

    cols_lower = [c.lower() for c in df.columns.astype(str)]

    # If there's a 'height' column (case-insensitive), use it
    if 'height' in cols_lower:
        height_col = df.columns[cols_lower.index('height')]
        heights = pd.to_numeric(df[height_col], errors='coerce').dropna().values

    # Try reading parameters from headered CSV (first row)
    first_row = df.iloc[0]
    for key in ['H', 'omega_rpm', 'g', 'rho_air', 'Ax', 'Ay', 'v_b0_kmh', 'z0', 'cp']:
        # check exact name or lowercase match
        if key in df.columns:
            try:
                params[key] = float(df[key].iloc[0])
            except Exception:
                pass
        elif key.lower() in cols_lower:
            colname = df.columns[cols_lower.index(key.lower())]
            try:
                params[key] = float(df[colname].iloc[0])
            except Exception:
                pass

    return params, heights


# --- App UI ---
st.title("Wind Load Calculator (BS EN 1991.1.4) 🌬️")
st.write("This app calculates wind loads. You can input parameters manually or upload a file. Headerless single-line upload format is supported: `H, g, rho_air, omega_rpm`.")

mode = st.radio("Select Input Mode", ["Manual Input", "Upload Dataset"])

# Manual mode
if mode == "Manual Input":
    st.subheader("Manual Parameter Input")

    col1, col2 = st.columns(2)
    with col1:
        H = st.number_input("Total Height (H) [m]", value=66.7, min_value=1.0, step=1.0)
        omega_rpm = st.number_input("Angular Velocity (ω) [RPM]", value=2.0, min_value=0.0, step=0.1)
        omega_rad_s = omega_rpm * 2 * np.pi / 60.0
        st.info(f"ω = {omega_rad_s:.6f} rad/s")
    with col2:
        g = st.number_input("Gravity (g) [m/s²]", value=9.81, min_value=0.0, step=0.01)
        rho_air = st.number_input("Air Density (ρ) [kg/m³]", value=1.225, min_value=0.1, step=0.001, format="%.3f")

    # Advanced options in expander
    with st.expander("Advanced / Optional parameters"):
        Ax = st.number_input("Cross-sectional Area X (Ax) [m²]", value=303.3, min_value=0.01, step=1.0)
        Ay = st.number_input("Cross-sectional Area Y (Ay) [m²]", value=592.5, min_value=0.01, step=1.0)
        v_b0_kmh = st.number_input("Basic Wind Velocity (v_b0) [km/h]", value=100.0, min_value=1.0, step=1.0)
        z0 = st.number_input("Roughness length (z0) [m]", value=0.01, min_value=0.0001, step=0.01, format="%.4f")
        cp = st.number_input("Pressure coefficient (cp)", value=1.2, min_value=0.1, step=0.1)

    v_b0 = v_b0_kmh / 3.6

    if st.button("Calculate Wind Load"):
        with st.spinner("Calculating..."):
            results = calculate_wind_load(H, omega_rad_s, g, rho_air, Ax=Ax, Ay=Ay, z0=z0, cp=cp)
            st.success("✅ Calculation Complete!")

            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Velocity", f"{results['vm_max']:.2f} m/s")
            c2.metric("Max Load (X)", f"{results['Fwx'][-1]:.2f} kN")
            c3.metric("Max Load (Y)", f"{results['Fwy'][-1]:.2f} kN")
            c4.metric("ω (rad/s)", f"{results['omega_rad_s']:.6f}")

            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#121212')

            ax1.plot(results['Fwy'], results['z'], label='Y-direction', color='cyan', linewidth=2)
            ax1.plot(results['Fwx'], results['z'], label='X-direction', color='orange', linewidth=2)
            ax1.set_xlabel('Wind Load [kN]')
            ax1.set_ylabel('Height [m]')
            ax1.set_title('Wind Load Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#1e1e1e')

            ax2.plot(results['vm'], results['z'], label='Mean Wind Velocity', color='green', linewidth=2)
            ax2.set_xlabel('Wind Velocity [m/s]')
            ax2.set_ylabel('Height [m]')
            ax2.set_title('Wind Velocity Profile')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#1e1e1e')

            plt.tight_layout()
            st.pyplot(fig)

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


# Dataset / upload mode
else:
    st.subheader("Upload Parameters or Height List")
    st.write("Supported uploads:\n- Headerless single-line: `H, g, rho_air, omega_rpm`\n- Single-column heights (one per line)\n- Headered CSV with columns like H, omega_rpm, g, rho_air, height, ...")

    uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if uploaded_file is not None:
        params, heights = read_upload_file(uploaded_file)

        if params is None and heights is None:
            st.error("Could not parse the uploaded file.")
        else:
            st.success("File parsed (defaults used where missing).")
            st.write({
                'H': params['H'],
                'omega_rpm': params['omega_rpm'],
                'g': params['g'],
                'rho_air': params['rho_air'],
                'Ax': params['Ax'],
                'Ay': params['Ay'],
                'v_b0_kmh': params['v_b0_kmh'],
                'z0': params['z0'],
                'cp': params['cp'],
            })

            # Convert RPM to rad/s
            omega_rad_s = float(params['omega_rpm']) * 2 * np.pi / 60.0

            if heights is None:
                # Build integer grid from H
                H_max = int(np.ceil(float(params['H'])))
                st.info(f"No explicit height list found — using integer heights 1..{H_max} from H")

                results = calculate_wind_load(H_max, omega_rad_s, float(params['g']), float(params['rho_air']),
                                              Ax=float(params['Ax']), Ay=float(params['Ay']),
                                              z0=float(params['z0']), cp=float(params['cp']))

                df_output = pd.DataFrame({
                    'Height [m]': results['z'],
                    'Wind Velocity [m/s]': results['vm'],
                    'Wind Load X [kN]': results['Fwx'],
                    'Wind Load Y [kN]': results['Fwy'],
                })
                st.dataframe(df_output)

                # plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.patch.set_facecolor('#121212')
                ax1.scatter(results['Fwy'], results['z'], label='Y-direction', s=20, alpha=0.6)
                ax1.scatter(results['Fwx'], results['z'], label='X-direction', s=20, alpha=0.6)
                ax1.set_xlabel('Wind Load [kN]'); ax1.set_ylabel('Height [m]'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_facecolor('#1e1e1e')
                ax2.scatter(results['vm'], results['z'], label='Mean Wind Velocity', s=20, alpha=0.6)
                ax2.set_xlabel('Wind Velocity [m/s]'); ax2.set_ylabel('Height [m]'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_facecolor('#1e1e1e')
                plt.tight_layout()
                st.pyplot(fig)

            else:
                # Interpolate results to uploaded heights
                H_max = int(np.ceil(float(max(heights.max(), params['H']))))
                results = calculate_wind_load(H_max, omega_rad_s, float(params['g']), float(params['rho_air']),
                                              Ax=float(params['Ax']), Ay=float(params['Ay']),
                                              z0=float(params['z0']), cp=float(params['cp']))

                Fwx_interp = np.interp(heights, results['z'], results['Fwx'])
                Fwy_interp = np.interp(heights, results['z'], results['Fwy'])
                vm_interp = np.interp(heights, results['z'], results['vm'])

                st.metric("Max Velocity (interp)", f"{vm_interp.max():.2f} m/s")
                st.metric("Max Load (X) (interp)", f"{Fwx_interp.max():.2f} kN")
                st.metric("Max Load (Y) (interp)", f"{Fwy_interp.max():.2f} kN")
                st.metric("ω (rad/s)", f"{omega_rad_s:.6f}")

                df_output = pd.DataFrame({
                    'Height [m]': heights,
                    'Wind Velocity [m/s]': vm_interp,
                    'Wind Load X [kN]': Fwx_interp,
                    'Wind Load Y [kN]': Fwy_interp
                })
                st.dataframe(df_output)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.patch.set_facecolor('#121212')
                ax1.scatter(Fwy_interp, heights, label='Y-direction', s=20, alpha=0.6)
                ax1.scatter(Fwx_interp, heights, label='X-direction', s=20, alpha=0.6)
                ax1.set_xlabel('Wind Load [kN]'); ax1.set_ylabel('Height [m]'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_facecolor('#1e1e1e')
                ax2.scatter(vm_interp, heights, label='Mean Wind Velocity', s=20, alpha=0.6)
                ax2.set_xlabel('Wind Velocity [m/s]'); ax2.set_ylabel('Height [m]'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_facecolor('#1e1e1e')
                plt.tight_layout()
                st.pyplot(fig)


# --- Info ---
with st.expander("About accepted file formats & mapping"):
    st.write("""
    - Headerless single-line with 4 values: `H, g, rho_air, omega_rpm` (e.g. `66.7, 9.81, 1.225, 2`)
      -> parsed as H (m), g (m/s^2), rho_air (kg/m^3), omega (RPM)
    - Single-column file (no header): interpreted as heights (one per line)
    - Headered CSV: will attempt to read columns named H, omega_rpm, g, rho_air, height, Ax, Ay, v_b0_kmh, z0, cp
    """)
