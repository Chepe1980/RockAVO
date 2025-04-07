import streamlit as st
import numpy as np
import math

# Sidebar for Layer Properties Input
with st.sidebar:
    st.header("🔧 Layer Properties")
    
    # Layer 1 (Top Layer)
    st.subheader("Layer 1 Properties")
    vp1 = st.text_input("P-Wave Velocity (Layer 1) [m/s]", "2500")  # Default value
    vs1 = st.text_input("S-Wave Velocity (Layer 1) [m/s]", "1500")  # Default value
    rho1 = st.text_input("Density (Layer 1) [g/cc]", "2.65")  # Default value
    thickness1 = st.text_input("Layer 1 Thickness [m]", "10")  # Default value

    # Layer 2 (Middle Layer)
    st.subheader("Layer 2 Properties")
    vp2 = st.text_input("P-Wave Velocity (Layer 2) [m/s]", "3000")  # Default value
    vs2 = st.text_input("S-Wave Velocity (Layer 2) [m/s]", "1700")  # Default value
    rho2 = st.text_input("Density (Layer 2) [g/cc]", "2.75")  # Default value
    thickness2 = st.text_input("Layer 2 Thickness [m]", "15")  # Default value
    
    # Layer 3 (Bottom Layer)
    st.subheader("Layer 3 Properties")
    vp3 = st.text_input("P-Wave Velocity (Layer 3) [m/s]", "3500")  # Default value
    vs3 = st.text_input("S-Wave Velocity (Layer 3) [m/s]", "2000")  # Default value
    rho3 = st.text_input("Density (Layer 3) [g/cc]", "2.85")  # Default value
    thickness3 = st.text_input("Layer 3 Thickness [m]", "20")  # Default value

    # Max Angle for AVO
    max_angle = st.slider("Maximum Angle for AVO (degrees)", 10, 60, 45, 1)

# Convert input to float or int
vp1, vs1, rho1, thickness1 = float(vp1), float(vs1), float(rho1), float(thickness1)
vp2, vs2, rho2, thickness2 = float(vp2), float(vs2), float(rho2), float(thickness2)
vp3, vs3, rho3, thickness3 = float(vp3), float(vs3), float(rho3), float(thickness3)

# Generate synthetic data using user-defined layer properties
layer_info = [
    {'vp': vp1, 'vs': vs1, 'rho': rho1, 'thickness': thickness1},
    {'vp': vp2, 'vs': vs2, 'rho': rho2, 'thickness': thickness2},
    {'vp': vp3, 'vs': vs3, 'rho': rho3, 'thickness': thickness3}
]

# Define the AVO curve calculation (Reflection Coefficient) using the Zoeppritz equation
def safe_rc_zoep(vp1, vs1, vp2, vs2, rho1, rho2, angle):
    """Calculate reflection coefficient using Zoeppritz equations with safety checks."""
    try:
        theta1_rad = math.radians(angle)
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
        st.warning(f"Calculation failed for angle {angle}°: {str(e)}")
        return 0.0  # Return zero reflection coefficient if error occurs

# Define time axis and calculate arrival times for each layer (simple model)
t = np.arange(0, 2.0, 0.001)  # Time array (0 to 2 seconds)
lyr_times = np.cumsum([0] + [layer['thickness'] / layer['vp'] for layer in layer_info])

# Now, let's calculate reflection coefficients (RC) for different angles and model the seismic data.
angles = np.arange(0, max_angle + 1, 5)  # Angles from 0 to max_angle with step of 5 degrees
rc_zoep = []
for angle in angles:
    rc = safe_rc_zoep(vp1, vs1, vp2, vs2, rho1, rho2, angle)
    rc_zoep.append(rc)

# Plot gathers with AVO curves
min_plot_time = 0.0  # Start time for plotting
max_plot_time = 2.0  # End time for plotting
excursion = 0.2  # Excursion of the plot (amplitude scale)

# Assuming tp.syn_angle_gather is a function to plot the synthetic gathers with AVO curves
try:
    import pyavo.seismodel as tp
    tp.syn_angle_gather(min_plot_time, max_plot_time, lyr_times, 
                        [layer['thickness'] for layer in layer_info],  # Thickness of each layer
                        [0, lyr_times[1], lyr_times[2]],  # Top layer, middle layer, bottom layer
                        [layer['vp'] for layer in layer_info],  # P-Wave velocities
                        [layer['vs'] for layer in layer_info],  # S-Wave velocities
                        [layer['rho'] for layer in layer_info],  # Densities
                        lyr_times,  # Reflection coefficient times (using layer times as reference)
                        rc_zoep,  # AVO curve (reflection coefficients for each angle)
                        t,  # Time array
                        excursion)  # Excursion for amplitude scaling
except ImportError:
    st.error("pyavo library is required for plotting synthetic gathers and AVO curves.")

st.success("Synthetic gather and AVO curves generated successfully!")

