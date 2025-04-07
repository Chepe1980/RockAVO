import streamlit as st
import lasio
import numpy as np
import pandas as pd
from pyavo.avotools import log_crossplot as log
from pyavo.seismodel import angle_stack as stk
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("Seismic Well Log Analysis & Synthetic Modeling")

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # Well Log Parameters
    st.subheader("Well Log Properties")
    rho_qtz = st.slider("Quartz Density (g/cc)", 2.0, 3.0, 2.65, 0.01)
    rho_fl = st.slider("Fluid Density (g/cc)", 0.8, 1.2, 1.05, 0.01)
    
    # Synthetic Model Parameters
    st.subheader("Synthetic Model Layers")
    vp1 = st.slider("Layer 1 Vp (m/s)", 2000, 3000, 2400, 10)
    vp2 = st.slider("Layer 2 Vp (m/s)", 2000, 3000, 2500, 10)
    vp3 = st.slider("Layer 3 Vp (m/s)", 2000, 3000, 2450, 10)
    
    vs1 = st.slider("Layer 1 Vs (m/s)", 1000, 2000, 1400, 10)
    vs2 = st.slider("Layer 2 Vs (m/s)", 1000, 2000, 1500, 10)
    vs3 = st.slider("Layer 3 Vs (m/s)", 1000, 2000, 1400, 10)
    
    rho1 = st.slider("Layer 1 Density (g/cc)", 1.5, 2.5, 1.97, 0.01)
    rho2 = st.slider("Layer 2 Density (g/cc)", 1.5, 2.5, 2.00, 0.01)
    rho3 = st.slider("Layer 3 Density (g/cc)", 1.5, 2.5, 1.98, 0.01)
    
    # Seismic Parameters
    st.subheader("Seismic Settings")
    thickness = st.slider("Layer Thickness (m)", 5, 50, 10, 1)
    freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 50, 5)
    max_angle = st.slider("Maximum Angle (deg)", 10, 60, 45, 1)

# Load well data (placeholder - replace with actual LAS loading)
try:
    well = lasio.read('./data/well_5.las')
    z_well = well['DEPT'].data
    rho_well = well['RHOB'].data
    vp_well = well['VP'].data
    vs_well = well['VS'].data
    gr_well = well['GR'].data
    vpvs_well = vp_well / vs_well
    phi_well = (rho_qtz - rho_well)/(rho_qtz - rho_fl)
except:
    st.warning("Well log file not found. Using synthetic data.")
    z_well = np.linspace(2000, 2500, 100)
    rho_well = np.linspace(1.9, 2.3, 100)
    vp_well = np.linspace(2400, 2800, 100)
    vs_well = np.linspace(1400, 1700, 100)
    gr_well = np.random.normal(50, 10, 100)
    vpvs_well = vp_well / vs_well
    phi_well = (rho_qtz - rho_well)/(rho_qtz - rho_fl)

# Create DataFrames
well_dict = {
    'Depth': z_well,
    'Gr': gr_well,
    'Rho': rho_well, 
    'Vp': vp_well, 
    'Vs': vs_well,
    'VpVs': vpvs_well,
    'PHI': phi_well
}
well_df = pd.DataFrame(well_dict)

# Calculate impedance
im = log.plot_imp(vpvs=vpvs_well, vp=vp_well, vs=vs_well, 
                  rho=rho_well, angle=30, h_well=z_well, h_ref=2170)

# Create synthetic model
vp_data = [vp1, vp2, vp3]
vs_data = [vs1, vs2, vs3]
rho_data = [rho1, rho2, rho3]

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
    z_int = tp.int_depth(h_int=[500.0], thickness=thickness)
    t_int = tp.calc_times(z_int, vp_data)
    lyr_times.append(t_int)
    rc = tp.mod_digitize(rc_zoep[angle], t_int, t_samp)
    s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
    syn_zoep.append(s)

syn_zoep = np.array(syn_zoep)
rc_zoep = np.array(rc_zoep)
lyr_times = np.array(lyr_times)

# Visualization
tab1, tab2, tab3 = st.tabs(["Well Logs", "Crossplots", "Synthetic Seismic"])

with tab1:
    st.subheader("Well Log Display")
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    
    ax[0].plot(well_df['Gr'], well_df['Depth'], 'g')
    ax[0].set_title("Gamma Ray")
    ax[0].invert_yaxis()
    
    ax[1].plot(well_df['Rho'], well_df['Depth'], 'b')
    ax[1].set_title("Density")
    ax[1].invert_yaxis()
    
    ax[2].plot(well_df['Vp'], well_df['Depth'], 'r')
    ax[2].set_title("P-Wave Velocity")
    ax[2].invert_yaxis()
    
    ax[3].plot(well_df['PHI'], well_df['Depth'], 'k')
    ax[3].set_title("Porosity")
    ax[3].invert_yaxis()
    
    st.pyplot(fig)

with tab2:
    st.subheader("Rock Physics Crossplots")
    fig = log.crossplot(
        vp=vp_well, vs=vs_well, vpvs=vpvs_well, rho=rho_well, 
        phi=phi_well, GR=gr_well, AI=im['AI'], NEI=im['NEI'], 
        lambda_rho=im['lambda_rho'], mu_rho=im['mu_rho']
    )
    st.pyplot(fig)

with tab3:
    st.subheader("Synthetic Seismic Gather")
    fig = tp.syn_angle_gather(
        min_plot_time=0.15, 
        max_plot_time=0.3, 
        lyr_times=lyr_times,
        thickness=thickness, 
        top_layer=syn_zoep[:, lyr_times[:,0].astype(int)], 
        bottom_layer=syn_zoep[:, lyr_times[:,1].astype(int)],
        vp_dig=vp_data[0], 
        vs_dig=vs_data[0], 
        rho_dig=rho_data[0], 
        syn_seis=syn_zoep, 
        rc_zoep=rc_zoep, 
        t=t_samp, 
        excursion=2
    )
    st.pyplot(fig)
