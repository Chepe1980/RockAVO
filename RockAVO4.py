import streamlit as st
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from pyavo.avotools import log_crossplot as log
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet
import math

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("🎯 Seismic Well Log Modeling App")

# Safe implementation of Zoeppritz equations with range checking
def safe_rc_zoep(vp1, vs1, vp2, vs2, rho1, rho2, theta1):
    """Calculate reflection coefficient using Zoeppritz equations with safety checks"""
    try:
        theta1_rad = math.radians(theta1)
        p = math.sin(theta1_rad) / vp1  # Ray parameter
        
        # Safe calculation of transmission angles with clamping
        sin_theta2 = min(1.0, max(-1.0, p * vp2))
        sin_phi1 = min(1.0, max(-1.0, p * vs1))
        sin_phi2 = min(1.0, max(-1.0, p * vs2))
        
        theta2 = math.asin(sin_theta2)
        phi1 = math.asin(sin_phi1)
        phi2 = math.asin(sin_phi2)
        
        # Coefficients for Zoeppritz equations
        a = rho2 * (1 - 2 * math.sin(phi2)**2) - rho1 * (1 - 2 * math.sin(phi1)**2)
        b = rho2 * (1 - 2 * math.sin(phi2)**2) + 2 * rho1 * math.sin(phi1)**2
        c = rho1 * (1 - 2 * math.sin(phi1)**2) + 2 * rho2 * math.sin(phi2)**2
        d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
        
        E = (b * math.cos(theta1_rad) / vp1) + (c * math.cos(theta2) / vp2)
        F = (b * math.cos(phi1) / vs1) + (c * math.cos(phi2) / vs2)
        G = a - d * math.cos(theta1_rad)/vp1 * math.cos(phi2)/vs2
        H = a - d * math.cos(theta2)/vp2 * math.cos(phi1)/vs1
        
        D = E * F + G * H * p**2
        
        # PP reflection coefficient - CORRECTED with proper parentheses
        Rpp = (1/D) * (F*(b*math.cos(theta1_rad)/vp1 - H*(a + d*math.cos(theta1_rad)/vp1 * math.cos(phi2)/vs2))) - 1
        
        return Rpp
        
    except Exception as e:
        st.warning(f"Calculation failed for angle {theta1}°: {str(e)}")
        return 0.0  # Return zero reflection coefficient if error occurs

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Parameters")
    uploaded_file = st.file_uploader("Upload LAS File", type=["las", "LAS"])
    
    st.subheader("Rock Physics")
    rho_qtz = st.slider("Quartz Density (g/cc)", 2.0, 3.0, 2.65, 0.01)
    rho_fl = st.slider("Fluid Density (g/cc)", 0.8, 1.2, 1.05, 0.01)
    
    st.subheader("Synthetic Model")
    thickness = st.slider("Layer Thickness (m)", 5, 50, 10, 1)
    freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 50, 5)
    max_angle = st.slider("Maximum Angle (deg)", 10, 60, 45, 1)

# Process LAS file or use default data
if uploaded_file:
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        las = lasio.read(stringio)
        
        # Get curve names (case insensitive)
        curve_names = {curve.mnemonic.lower(): curve.mnemonic for curve in las.curves}
        
        # Extract data with fallbacks
        def get_curve(name, alternatives, default=None):
            for alt in [name] + alternatives:
                if alt.lower() in curve_names:
                    return np.array(las[curve_names[alt.lower()]].data)
            return default if default is not None else np.zeros(len(las.index))
        
        z_well = get_curve('depth', ['dept', 'depths'])
        rho_well = get_curve('rhob', ['density', 'den'], np.linspace(1.9, 2.3, len(z_well)))
        vp_well = get_curve('vp', ['dtp', 'pwave'], np.linspace(2400, 2800, len(z_well)))
        vs_well = get_curve('vs', ['dts', 'swave'], np.divide(vp_well, 1.7))
        gr_well = get_curve('gr', ['gamma', 'gammaray'], np.random.normal(50, 10, len(z_well)))
        
        st.success(f"✅ LAS file loaded with {len(z_well)} data points")
        
    except Exception as e:
        st.error(f"❌ Error reading LAS file: {str(e)}")
        st.stop()
else:
    st.warning("⚠️ Using synthetic data - upload LAS file for real analysis")
    z_well = np.linspace(2000, 2500, 100)
    rho_well = np.linspace(1.9, 2.3, 100)
    vp_well = np.linspace(2400, 2800, 100)
    vs_well = np.divide(vp_well, 1.7)
    gr_well = np.random.normal(50, 10, 100)

# Calculate derived properties with safe division
vpvs_well = np.divide(vp_well, vs_well, out=np.zeros_like(vp_well), where=vs_well!=0)
phi_well = np.divide((rho_qtz - rho_well), (rho_qtz - rho_fl), out=np.zeros_like(rho_well), where=(rho_qtz - rho_fl)!=0)

# Create DataFrame
well_df = pd.DataFrame({
    'Depth': z_well,
    'GR': gr_well,
    'Density': rho_well, 
    'Vp': vp_well, 
    'Vs': vs_well,
    'Vp/Vs': vpvs_well,
    'Porosity': phi_well
})

# Calculate impedance
try:
    im = log.plot_imp(vpvs=vpvs_well, vp=vp_well, vs=vs_well, 
                     rho=rho_well, angle=30, h_well=z_well, h_ref=z_well.mean())
except Exception as e:
    st.error(f"Impedance calculation failed: {str(e)}")
    st.stop()

# Create synthetic model from log statistics
vp1, vp2 = np.percentile(vp_well, [25, 75])
vs1, vs2 = np.percentile(vs_well, [25, 75])
rho1, rho2 = np.percentile(rho_well, [25, 75])

vp_data = [vp1, vp2, vp1*0.95]  # 3-layer model
vs_data = [vs1, vs2, vs1*0.95]
rho_data = [rho1, rho2, rho1*0.98]

# Generate synthetic seismic with angle validation
nangles = tp.n_angles(0, max_angle)
rc_values = []
valid_angles = []

for angle in range(nangles):
    try:
        rc = safe_rc_zoep(vp_data[0], vs_data[0], vp_data[1], vs_data[1], 
                         rho_data[0], rho_data[1], angle)
        if not np.isnan(rc):
            rc_values.append(rc)
            valid_angles.append(angle)
    except Exception as e:
        st.warning(f"Skipping angle {angle}°: {str(e)}")
        continue

if not rc_values:
    st.error("No valid reflection coefficients could be calculated")
    st.stop()

# Generate wavelet
wlt_time, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=freq)

# Generate synthetic seismogram
t_samp = np.arange(0, 0.5, 0.001)
syn_seismic = []

for i, rc in enumerate(rc_values):
    try:
        # Create reflectivity series
        rc_series = np.zeros_like(t_samp)
        rc_series[100] = rc  # Place at 100th sample
        
        # Convolve with wavelet
        synthetic = np.convolve(rc_series, wlt_amp, mode='same')
        syn_seismic.append(synthetic)
    except Exception as e:
        st.warning(f"Error generating synthetic for angle {valid_angles[i]}°: {str(e)}")
        continue

syn_seismic = np.array(syn_seismic)

# Main display tabs
tab1, tab2, tab3 = st.tabs(["📊 Well Logs", "📈 Crossplots", "🎚️ Synthetic Seismic"])

with tab1:
    st.subheader("Well Log Display")
    fig, ax = plt.subplots(1, 5, figsize=(20, 8))
    
    logs = [
        ('GR', 'Gamma Ray', 'green'),
        ('Density', 'Density', 'blue'),
        ('Vp', 'P-Wave Velocity', 'red'),
        ('Vs', 'S-Wave Velocity', 'purple'),
        ('Porosity', 'Porosity', 'black')
    ]
    
    for i, (col, title, color) in enumerate(logs):
        ax[i].plot(well_df[col], well_df['Depth'], color)
        ax[i].set_title(title)
        ax[i].invert_yaxis()
    
    st.pyplot(fig)
    st.dataframe(well_df.describe(), use_container_width=True)

with tab2:
    st.subheader("Rock Physics Crossplots")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vp vs Vs colored by Density
    sc1 = ax1.scatter(well_df['Vp'], well_df['Vs'], c=well_df['Density'], cmap='viridis')
    plt.colorbar(sc1, ax=ax1, label='Density (g/cc)')
    ax1.set_xlabel('Vp (m/s)')
    ax1.set_ylabel('Vs (m/s)')
    ax1.set_title('Vp vs Vs')
    
    # AI vs Porosity colored by GR
    ai = well_df['Vp'] * well_df['Density']
    sc2 = ax2.scatter(ai, well_df['Porosity'], c=well_df['GR'], cmap='plasma')
    plt.colorbar(sc2, ax=ax2, label='GR (API)')
    ax2.set_xlabel('Acoustic Impedance')
    ax2.set_ylabel('Porosity')
    ax2.set_title('AI vs Porosity')
    
    st.pyplot(fig)

with tab3:
    st.subheader("Synthetic Seismic Gather")
    
    if len(syn_seismic) > 0:
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(syn_seismic.T, aspect='auto', 
                  extent=[0, len(valid_angles), t_samp[-1], t_samp[0]],
                  cmap='seismic', 
                  vmin=-np.max(np.abs(syn_seismic)), 
                  vmax=np.max(np.abs(syn_seismic)))
        plt.colorbar(label='Amplitude')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Time (s)')
        plt.title(f'Synthetic Gather ({len(valid_angles)} angles)')
        st.pyplot(fig)
        
        st.info(f"Processed angles: {valid_angles}")
    else:
        st.warning("No valid synthetic gather to display")

# Model parameters expander
with st.expander("🔍 Model Parameters"):
    st.subheader("Synthetic Model Properties")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Layer 1 (Top)**")
        st.write(f"- Vp: {vp_data[0]:.0f} m/s")
        st.write(f"- Vs: {vs_data[0]:.0f} m/s")
        st.write(f"- Density: {rho_data[0]:.2f} g/cc")
        
    with col2:
        st.write("**Layer 2 (Middle)**")
        st.write(f"- Vp: {vp_data[1]:.0f} m/s")
        st.write(f"- Vs: {vs_data[1]:.0f} m/s")
        st.write(f"- Density: {rho_data[1]:.2f} g/cc")
    
    st.write("**Wavelet**")
    st.write(f"- Frequency: {freq} Hz")
    st.write(f"- Length: 128 ms")
