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
    
    st.subheader("Layer Information")
    num_layers = st.number_input("Number of Layers", min_value=1, max_value=10, value=3, step=1)
    
    # Define text inputs for each layer's properties
    layer_info = []
    for i in range(num_layers):
        st.subheader(f"Layer {i+1}")
        thickness = st.number_input(f"Layer {i+1} Thickness (m)", min_value=1, value=10)
        vp = st.number_input(f"Layer {i+1} Vp (m/s)", min_value=1000, value=2500)
        vs = st.number_input(f"Layer {i+1} Vs (m/s)", min_value=500, value=1200)
        density = st.number_input(f"Layer {i+1} Density (g/cc)", min_value=1.5, value=2.5)
        
        layer_info.append((thickness, vp, vs, density))

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

# Now, let's modify the seismic gather plotting
# We'll collect the values for each layer to build the synthetic gather model
thickness, vp, vs, rho = zip(*layer_info)

# Create synthetic model (Adjust based on user input)
# Use predefined minimum and maximum values for angle (for simplicity)
min_plot_time = 0
max_plot_time = 0.5  # Max time for plotting (e.g., 500 ms)

# Generate synthetic seismic gather using AVO model and Zoeppritz
lyr_times = np.cumsum(thickness)  # Calculate the layer time (this is a simple cumulative sum)
t = np.linspace(min_plot_time, max_plot_time, 500)  # Time axis
excursion = 0.1  # Excursion factor for modeling AVO effects (example value)

# Using the given function to plot the synthetic gather
tp.syn_angle_gather(min_plot_time, max_plot_time, lyr_times, 
                    thickness, 0, len(thickness)-1,  # Define top and bottom layer indices
                    vp, vs, rho, None,  # Assuming syn_zoep and rc_zoep are not required in this context
                    None, t, excursion)

# Plotting synthetic gather using the defined function
st.subheader("Synthetic Seismic Gather Model")
st.pyplot(plt)

# Next part: Add the AVO (Amplitude Versus Offset) Economic Curves
st.subheader("AVO Economic Curves")

# Calculate and plot AVO curves
# AVO economic curves usually involve plotting reflectivity (Rpp) vs. angle or offset.
# Let's plot synthetic AVO curves using the Zoeppritz reflectivity model for different layers.

# Let's use the Zoeppritz reflectivity calculation model to plot the AVO curves
def plot_avo_curves(vp1, vs1, vp2, vs2, rho1, rho2, angles):
    rc_values = []
    for angle in angles:
        rc = safe_rc_zoep(vp1, vs1, vp2, vs2, rho1, rho2, angle)
        rc_values.append(rc)
    return rc_values

# Example: Calculate and plot AVO for the first two layers (user-defined)
angles = np.arange(0, 45, 5)  # Angles from 0 to 45 degrees
rc_layer1 = plot_avo_curves(vp[0], vs[0], vp[1], vs[1], rho[0], rho[1], angles)

# Plot AVO curves
plt.figure(figsize=(8, 6))
plt.plot(angles, rc_layer1, label="Layer 1 to Layer 2", color="blue")
plt.xlabel("Angle (degrees)")
plt.ylabel("Reflection Coefficient (Rpp)")
plt.title("AVO Curves")
plt.legend()
st.pyplot(plt)


