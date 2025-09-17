# --- ADM1 Interactive Simulation (Accordion UI, Cleaned Style) ---

from ADM1 import ADM1Simulator
import plotly.graph_objects as go
import pandas as pd
from ipywidgets import FloatSlider, Button, VBox, HBox, Output, Accordion
from IPython.display import display, FileLink, clear_output
from tqdm import tqdm

# --- Helper functions ---

def get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry):
    ratios = {
        'Maize Silage': maize_silage/100,
        'Grass Silage': grass_silage/100,
        'Food Waste': food_waste/100,
        'Cattle Slurry': cattle_slurry/100
    }
    total = sum(ratios.values())
    return {k: (v/total if total > 0 else 0) for k,v in ratios.items()}

def run_adm1(maize_silage, grass_silage, food_waste, cattle_slurry, V, Q, T, sim_days, output):
    with output:
        clear_output(wait=True)
        ratios = get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry)
        days = int(sim_days)

        simulator = ADM1Simulator(ratios, days=days, Q=Q, V=V, T=T)
        with tqdm(total=days, desc='Simulating Days', disable=True) as pbar:
            simulator.run()
            pbar.update(days)

        output_data = simulator.get_results()
        if output_data.empty:
            print("❌ Simulation returned no data.")
            return

        # --- Show steady-state summary ---
        summary_data = {
            'pH': [output_data['pH'].iloc[-1]],
            'FOS': [output_data['FOS'].iloc[-1]],
            'TAC': [output_data['TAC'].iloc[-1]],
            'FOS/TAC': [output_data['FOS/TAC'].iloc[-1]],
            'Gas Pressure': [output_data['p_gas'].iloc[-1]],
            'HRT (V/Q)': [output_data['HRT'].iloc[-1]]
        }
        summary_df = pd.DataFrame(summary_data, index=['Steady-State Value'])
        display(summary_df.style.format('{:.2f}'))

        # --- Plot outputs ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=output_data['time'], y=output_data['q_ch4'],
                                 mode='lines', name='Methane Flow (m³/d)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=output_data['time'], y=output_data['q_gas'],
                                 mode='lines', name='Biogas Flow (m³/d)', line=dict(color='green')))
        fig.update_layout(
            title="ADM1 Simulation Outputs",
            xaxis_title="Time (days)",
            yaxis_title="Flow Rate (m³/d)",
            hovermode="x unified"
        )
        fig.show()

        # --- Export button ---
        export_button = Button(description="Export Results to CSV")
        def on_export_clicked(b):
            filename = "ADM1_results.csv"
            output_data.to_csv(filename, index=False)
            display(FileLink(filename))
            print(f"✅ CSV exported: {filename}")
        export_button.on_click(on_export_clicked)
        display(export_button)

        print("✅ Simulation Complete.")


# --- Output widget ---
output = Output()

# --- Sliders (clean & simple) ---
maize_slider  = FloatSlider(min=0, max=100, step=5, value=50, description="Maize (%)")
grass_slider  = FloatSlider(min=0, max=100, step=5, value=30, description="Grass (%)")
food_slider   = FloatSlider(min=0, max=100, step=5, value=10, description="Food (%)")
cattle_slider = FloatSlider(min=0, max=100, step=5, value=10, description="Slurry (%)")

v_slider  = FloatSlider(min=1000, max=10000, step=500, value=7000, description="Volume (m³)")
q_slider  = FloatSlider(min=50, max=500, step=10, value=140, description="Flow (m³/d)")
t_slider  = FloatSlider(min=25, max=65, step=1, value=45, description="Temp (°C)")
sim_slider = FloatSlider(min=50, max=200, step=10, value=150, description="Sim Days")

# --- Group into accordions (kept as before) ---
feedstock_box = VBox([maize_slider, grass_slider, food_slider, cattle_slider])
feedstock_acc = Accordion(children=[feedstock_box])
feedstock_acc.set_title(0, "Feedstock Mix (%)")
feedstock_acc.selected_index = None  # collapsed by default

process_box = VBox([v_slider, q_slider, t_slider, sim_slider])
process_acc = Accordion(children=[process_box])
process_acc.set_title(0, "Process Parameters")
process_acc.selected_index = None  # collapsed by default

# --- Buttons ---
run_button   = Button(description="Run Simulation")
reset_button = Button(description="Reset Sliders")

def on_run_clicked(b):
    run_adm1(maize_slider.value, grass_slider.value, food_slider.value, cattle_slider.value,
             v_slider.value, q_slider.value, t_slider.value, sim_slider.value, output)

def on_reset_clicked(b):
    maize_slider.value, grass_slider.value, food_slider.value, cattle_slider.value = 50, 30, 10, 10
    v_slider.value, q_slider.value, t_slider.value, sim_slider.value = 7000, 140, 45, 150
    with output:
        clear_output(wait=True)
        print("Sliders reset to defaults.")

run_button.on_click(on_run_clicked)
reset_button.on_click(on_reset_clicked)

# --- Final Layout ---
display(HBox([feedstock_acc, process_acc]), HBox([run_button, reset_button]), output)
