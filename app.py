# Save as app.py and run with: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ADM1 import ADM1Simulator

# Sidebar sliders
st.sidebar.header("Feedstock Mix (%)")
maize_silage = st.sidebar.slider("Maize Silage", 0, 100, 50, 5)
grass_silage = st.sidebar.slider("Grass Silage", 0, 100, 30, 10)
food_waste = st.sidebar.slider("Food Waste", 0, 100, 10, 10)
cattle_slurry = st.sidebar.slider("Cattle Slurry", 0, 100, 10, 10)

st.sidebar.header("Process Parameters")
V = st.sidebar.slider("Volume (m³)", 1000, 10000, 7000, 500)
Q = st.sidebar.slider("Flow Q (m³/d)", 50, 500, 136, 10)
T = st.sidebar.slider("Temperature (°C)", 25, 65, 45, 1)
sim_days = st.sidebar.slider("Simulation Days", 50, 100, 70, 5)

def get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry):
    ratios = {
        'Maize Silage': maize_silage/100,
        'Grass Silage': grass_silage/100,
        'Food Waste': food_waste/100,
        'Cattle Slurry': cattle_slurry/100
    }
    total = sum(ratios.values())
    return {k: v/total for k,v in ratios.items()} if total else ratios

if st.button("Run Simulation"):
    ratios = get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry)
    sim = ADM1Simulator(ratios, days=int(sim_days), Q=Q, V=V, T=T)
    sim.run()
    output_data = sim.get_results()

    if output_data.empty:
        st.error("Simulation returned no data.")
    else:
        # Summary table
        summary_data = {
            'pH': output_data['pH'].iloc[-1],
            'FOS': output_data['FOS'].iloc[-1],
            'TAC': output_data['TAC'].iloc[-1],
            'FOS/TAC': output_data['FOS/TAC'].iloc[-1],
            'Gas Pressure': output_data['p_gas'].iloc[-1],
            'HRT (V/Q)': output_data['HRT'].iloc[-1]
        }
        st.subheader("📊 Steady-State Process Indicators")
        st.dataframe(pd.DataFrame([summary_data]))

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=output_data['time'], y=output_data['q_ch4'], mode='lines', name="Methane Flow"))
        fig.add_trace(go.Scatter(x=output_data['time'], y=output_data['q_gas'], mode='lines', name="Biogas Flow"))
        fig.update_layout(title="Biogas and Methane Flow Rates Over Time",
                          xaxis_title="Time (days)",
                          yaxis_title="Flow Rate (m³/d)")
        st.plotly_chart(fig, use_container_width=True)

        # CSV export
        csv = output_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Export Results to CSV", csv, "ADM1_results.csv", "text/csv")
