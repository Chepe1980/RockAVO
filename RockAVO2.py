import streamlit as st
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from pyavo.avotools import log_crossplot as log
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("Seismic Well Log Modeling App")

# Sidebar for user inputs
with st.sidebar:
    st.header("1. Upload LAS File")
    uploaded_file = st.file_uploader("Choose a LAS file", type=["las", "LAS"])
    
    st.header("2. Rock Physics Parameters")
    rho_qtz = st.slider("Quartz Density (g/cc)", 2.0, 3.0, 2.65, 0.01)
    rho_fl = st.slider("Fluid Density (g/cc)", 0.8, 1.2, 1.05, 0.01)
    
    st.header("3. Synthetic Model Settings")
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
        
        # Extract data with fallbacks and convert to numpy arrays
        def get_curve(name, alternatives, default=None):
            for alt in [name] + alternatives:
                if alt.lower() in curve_names:
                    return np.array(las[curve_names[alt.lower()]].data)  # Convert to numpy array
            return default if default is not None else np.zeros(len(las.index))
        
        z_well = get_curve('depth', ['dept', 'depths'])
        rho_well = get_curve('rhob', ['density', 'den'], np.linspace(1.9, 2.3, len(z_well)))
        vp_well = get_curve('vp', ['dtp', 'pwave'], np.linspace(2400, 2800, len(z_well)))
        
        # Ensure VS calculation works properly
        vs_default = np.divide(vp_well, 1.7) if isinstance(vp_well, np.ndarray) else np.linspace(1400, 1700, len(z_well))
        vs_well = get_curve('vs', ['dts', 'swave'], vs_default)
        
        gr_well = get_curve('gr', ['gamma', 'gammaray'], np.random.normal(50, 10, len(z_well)))
        
        st.success(f"LAS file loaded successfully with {len(z_well)} data points")
        
    except Exception as e:
        st.error(f"Error reading LAS file: {str(e)}")
        st.stop()
else:
    st.warning("Using synthetic data - upload a LAS file for real well analysis")
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
    'Gr': gr_well,
    'Rho': rho_well, 
    'Vp': vp_well, 
    'Vs': vs_well,
    'VpVs': vpvs_well,
    'PHI': phi_well
})

# Calculate impedance
im = log.plot_imp(vpvs=vpvs_well, vp=vp_well, vs=vs_well, 
                  rho=rho_well, angle=30, h_well=z_well, h_ref=z_well.mean())

# Create synthetic model from log statistics
vp1, vp2 = np.percentile(vp_well, [25, 75])
vs1, vs2 = np.percentile(vs_well, [25, 75])
rho1, rho2 = np.percentile(rho_well, [25, 75])

vp_data = [vp1, vp2, vp1*0.95]  # 3-layer model
vs_data = [vs1, vs2, vs1*0.95]
rho_data = [rho1, rho2, rho1*0.98]

# Generate synthetic seismic
nangles = tp.n_angles(0, max_angle)
rc_zoep = []
theta1 = []

for angle in range(0, nangles):
    theta1_samp, rc_1, rc_2 = tp.calc_theta_rc(
        theta1_min=0, theta1_step=1, 
        vp=vp_data, vs=vs_data, rho=rho_data, ang=angle
    )
    theta1.append(theta1_samp)
    rc_zoep.append([rc_1[0, 0], rc_2[0, 0]])

# Generate wavelet
wlt_time, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=freq)

# Generate synthetic seismogram
syn_zoep = []
lyr_times = []
t_samp = np.arange(0, 0.5, 0.001)

for angle in range(0, nangles):
    z_int = tp.int_depth(h_int=[z_well.mean()], thickness=thickness)
    t_int = tp.calc_times(z_int, vp_data)
    lyr_times.append(t_int)
    rc = tp.mod_digitize(rc_zoep[angle], t_int, t_samp)
    s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
    syn_zoep.append(s)

syn_zoep = np.array(syn_zoep)
rc_zoep = np.array(rc_zoep)
lyr_times = np.array(lyr_times)

# Main display tabs
tab1, tab2, tab3, tab4 = st.tabs(["Well Logs", "Crossplots", "Synthetic Seismic", "Model Parameters"])

with tab1:
    st.subheader("Well Log Display")
    fig, ax = plt.subplots(1, 5, figsize=(25, 10))
    
    plots = [
        ('Gr', 'Gamma Ray', 'g'),
        ('Rho', 'Density', 'b'),
        ('Vp', 'P-Wave Velocity', 'r'),
        ('Vs', 'S-Wave Velocity', 'm'),
        ('PHI', 'Porosity', 'k')
    ]
    
    for i, (col, title, color) in enumerate(plots):
        ax[i].plot(well_df[col], well_df['Depth'], color)
        ax[i].set_title(title)
        ax[i].invert_yaxis()
    
    st.pyplot(fig)
    st.dataframe(well_df.describe(), use_container_width=True)

with tab2:
    st.subheader("Rock Physics Crossplots")
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(vp_well, vs_well, c=rho_well, cmap='viridis')
    plt.colorbar(label='Density (g/cc)')
    plt.xlabel('Vp (m/s)')
    plt.ylabel('Vs (m/s)')
    plt.title('Vp vs Vs Colored by Density')
    st.pyplot(fig)

with tab3:
    st.subheader("Synthetic Seismic Gather")
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(syn_zoep.T, aspect='auto', 
              extent=[0, max_angle, t_samp[-1], t_samp[0]],
              cmap='seismic', vmin=-np.max(np.abs(syn_zoep)), vmax=np.max(np.abs(syn_zoep)))
    plt.colorbar(label='Amplitude')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Time (s)')
    plt.title('Angle Gather')
    st.pyplot(fig)

with tab4:
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Well Log Statistics")
        st.write(f"- Mean Vp: {vp_well.mean():.1f} m/s")
        st.write(f"- Mean Vs: {vs_well.mean():.1f} m/s")
        st.write(f"- Mean Density: {rho_well.mean():.2f} g/cc")
        st.write(f"- Mean Porosity: {phi_well.mean():.2%}")
        
    with col2:
        st.write("#### Synthetic Model Layers")
        st.write("| Layer | Vp (m/s) | Vs (m/s) | Density (g/cc) |")
        st.write("|-------|----------|----------|----------------|")
        for i, (vp, vs, rho) in enumerate(zip(vp_data, vs_data, rho_data)):
            st.write(f"| {i+1} | {vp:.0f} | {vs:.0f} | {rho:.2f} |")
    
    st.write("#### Reflection Coefficients")
    st.dataframe(pd.DataFrame(rc_zoep, 
                 columns=['Interface 1', 'Interface 2'],
                 index=[f"{ang}°" for ang in range(nangles)]))
