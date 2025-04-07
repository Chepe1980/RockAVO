import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pyavo.seismodel import tuning_prestack as tp
from pyavo.avotools import log_crossplot as log
from io import StringIO
import lasio
import math

# Streamlit Setup
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
    
    st.subheader("Layer Information (3-Layer Model)")
    layer1_thickness = st.number_input("Layer 1 Thickness (m)", min_value=1, value=10)
    layer1_vp = st.number_input("Layer 1 Vp (m/s)", min_value=1000, value=2500)
    layer1_vs = st.number_input("Layer 1 Vs (m/s)", min_value=500, value=1200)
    layer1_density = st.number_input("Layer 1 Density (g/cc)", min_value=1.5, value=2.5)
    
    layer2_thickness = st.number_input("Layer 2 Thickness (m)", min_value=1, value=10)
    layer2_vp = st.number_input("Layer 2 Vp (m/s)", min_value=1000, value=3000)
    layer2_vs = st.number_input("Layer 2 Vs (m/s)", min_value=500, value=1400)
    layer2_density = st.number_input("Layer 2 Density (g/cc)", min_value=1.5, value=2.7)
    
    layer3_thickness = st.number_input("Layer 3 Thickness (m)", min_value=1, value=10)
    layer3_vp = st.number_input("Layer 3 Vp (m/s)", min_value=1000, value=3500)
    layer3_vs = st.number_input("Layer 3 Vs (m/s)", min_value=500, value=1600)
    layer3_density = st.number_input("Layer 3 Density (g/cc)", min_value=1.5, value=2.8)

# Process LAS file or use synthetic data
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

# Create synthetic model
thickness = [layer1_thickness, layer2_thickness, layer3_thickness]
vp = [layer1_vp, layer2_vp, layer3_vp]
vs = [layer1_vs, layer2_vs, layer3_vs]
rho = [layer1_density, layer2_density, layer3_density]

# Generate synthetic seismic gather using AVO model and Zoeppritz
min_plot_time = 0
max_plot_time = 0.5  # Max time for plotting (e.g., 500 ms)
lyr_times = np.cumsum(thickness)  # Layer times
t = np.linspace(min_plot_time, max_plot_time, 500)  # Time axis
excursion = 0.1  # Excursion factor for modeling AVO effects (example value)

# Generate synthetic reflectivity using Zoeppritz equations for each layer
syn_zoep = np.zeros((len(t), len(thickness)))  # Initialize the synthetic reflectivity matrix

# Generate synthetic reflectivity for each layer
for i in range(len(thickness)):
    angle_range = np.arange(0, 45, 5)  # Simple angle range from 0 to 45 degrees
    for j, angle in enumerate(angle_range):
        # Example values for reflectivity calculation (replace with actual layer properties)
        syn_zoep[j, i] = safe_rc_zoep(vp[i], vs[i], vp[(i+1)%len(vp)], vs[(i+1)%len(vs)], rho[i], rho[(i+1)%len(rho)], angle)

# Using the given function to plot the synthetic gather
tp.syn_angle_gather(min_plot_time, max_plot_time, lyr_times, 
                    thickness, 0, len(thickness)-1,  # Define top and bottom layer indices
                    vp, vs, rho, syn_zoep,  # Pass the syn_zoep reflectivity
                    None, t, excursion)

# Main display tabs
tab1, tab2, tab3 = st.tabs(["📊 Scatter Plots", "🎚️ Wavelet", "🔍 AVO Synthetic & Curves"])

with tab1:
    st.subheader("Scatter Plots")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vp vs Vs colored by Density
    sc1 = ax1.scatter(vp_well, vs_well, c=rho_well, cmap='viridis')
    plt.colorbar(sc1, ax=ax1, label='Density (g/cc)')
    ax1.set_xlabel('Vp (m/s)')
    ax1.set_ylabel('Vs (m/s)')
    ax1.set_title('Vp vs Vs')
    
    # AI vs Porosity colored by GR
    ai = vp_well * rho_well
    sc2 = ax2.scatter(ai, gr_well, c=gr_well, cmap='plasma')
    plt.colorbar(sc2, ax=ax2, label='Gamma Ray')
    ax2.set_xlabel('Acoustic Impedance (AI)')
    ax2.set_ylabel('Gamma Ray')
    ax2.set_title('AI vs Gamma Ray')
    
    st.pyplot(fig)

with tab2:
    st.subheader("Wavelet")
    st.write("Display a synthetic or loaded wavelet.")
    # You can add wavelet generation code here if needed.

with tab3:
    st.subheader("AVO Synthetic & Curves")
    # Add AVO curve plotting here based on Zoeppritz reflectivity
    # Use the previous `safe_rc_zoep` function to generate reflection coefficients
    # Plot AVO curves across different angles
    angles = np.linspace(0, 40, 9)
    rpp = [safe_rc_zoep(vp[0], vs[0], vp[1], vs[1], rho[0], rho[1], angle) for angle in angles]
    
    st.line_chart(pd.DataFrame(rpp, columns=['Reflection Coefficient'], index=angles))



