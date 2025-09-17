from ADM1 import ADM1Simulator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from ipywidgets import FloatSlider, Button, VBox, HBox, Output, Accordion
from IPython.display import display, FileLink, clear_output
from tqdm import tqdm
import numpy as np

def get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry):
    ratios = {
        'Maize Silage': maize_silage/100,
        'Grass Silage': grass_silage/100,
        'Food Waste': food_waste/100,
        'Cattle Slurry': cattle_slurry/100
    }
    total = sum(ratios.values())
    if total == 0:
        return ratios
    normalized = {k: v / total for k, v in ratios.items()}
    return normalized

def run_adm1(maize_silage, grass_silage, food_waste, cattle_slurry, V, Q, T, sim_days, output):
    with output:
        clear_output(wait=True)
        ratios = get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry)
        
        actual_HRT = V / Q
        days = int(sim_days)
        
        simulator = ADM1Simulator(ratios, days=days, Q=Q, V=V, T=T)

        with tqdm(total=days, desc='Simulating Days', disable=True) as pbar:
            simulator.run()
            pbar.update(days)

        output_data = simulator.get_results()
        if output_data.empty:
            print('❌ Simulation returned no data.')
            return

        # Summary table
        summary_data = {
            'pH': [output_data['pH'].iloc[-1]],
            'FOS': [output_data['FOS'].iloc[-1]],
            'TAC': [output_data['TAC'].iloc[-1]],
            'FOS/TAC': [output_data['FOS/TAC'].iloc[-1]],
            'Gas Pressure': [output_data['p_gas'].iloc[-1]],
            'HRT (V/Q)': [output_data['HRT'].iloc[-1]]
        }
        summary_df = pd.DataFrame(summary_data, index=['Steady-State Value'])
        styled_summary = summary_df.style \
            .format('{:.2f}') \
            .background_gradient(cmap='viridis', axis=1) \
            .set_properties(**{'font-size': '10pt', 'border': '1px solid black'}) \
            .set_caption('📊 Steady-State Process Indicators')

        display(styled_summary)

        # Matplotlib plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Flow rates over time
        ax1.plot(output_data['time'], output_data['q_ch4'], 'b-', linewidth=2, label='Methane Flow')
        ax1.plot(output_data['time'], output_data['q_gas'], 'purple', linewidth=2, label='Biogas Flow')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Flow Rate (m³/d)')
        ax1.set_title('ADM1 Simulation: Biogas and Methane Flow Rates Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Process indicators
        ax2_twin = ax2.twinx()
        ax2.plot(output_data['time'], output_data['pH'], 'g-', linewidth=2, label='pH')
        ax2_twin.plot(output_data['time'], output_data['FOS/TAC'], 'r-', linewidth=2, label='FOS/TAC')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('pH', color='g')
        ax2_twin.set_ylabel('FOS/TAC Ratio', color='r')
        ax2.set_title('Process Stability Indicators')
        ax2.grid(True, alpha=0.3)
        
        # Legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()

        # Export functionality
        export_button = Button(description='Export Results to CSV')
        def on_export_clicked(b):
            filename = 'ADM1_results.csv'
            output_data.to_csv(filename, index=False)
            display(FileLink(filename))
            print(f'✅ CSV exported: {filename}')
        export_button.on_click(on_export_clicked)
        display(export_button)

        print('✅ Simulation Complete.')

# Output widget
output = Output()

# Feedstock sliders
maize_slider = FloatSlider(min=0, max=100, step=5, value=50, description='🌽 Maize:', style={'description_width': '120px'})
grass_slider = FloatSlider(min=0, max=100, step=10, value=30, description='🌿 Grass:', style={'description_width': '120px'})
food_slider = FloatSlider(min=0, max=100, step=10, value=10, description='🍎 Food Waste:', style={'description_width': '120px'})
cattle_slider = FloatSlider(min=0, max=100, step=10, value=10, description='🐄 Cattle Slurry:', style={'description_width': '120px'})

# Process parameter sliders
v_slider = FloatSlider(min=1000, max=10000, step=500, value=7000, description='Volume:', style={'description_width': '100px'})
q_slider = FloatSlider(min=50, max=500, step=10, value=136.63, description='Flow Rate:', style={'description_width': '100px'})
t_slider = FloatSlider(min=25, max=65, step=1, value=45, description='Temperature:', style={'description_width': '100px'})
sim_period_slider = FloatSlider(min=50, max=100, step=5, value=70, description='Sim Period:', style={'description_width': '100px'})

# Simple layout without accordions
feedstock_title = Output()
with feedstock_title:
    print('📊 Feedstock Composition (%)')

process_title = Output()  
with process_title:
    print('⚙️ Process Parameters')

feedstock_box = VBox([feedstock_title, maize_slider, grass_slider, food_slider, cattle_slider])
process_box = VBox([process_title, v_slider, q_slider, t_slider, sim_period_slider])

# Buttons
run_button = Button(description='Run Simulation')
reset_button = Button(description='Reset Sliders')

def on_run_clicked(b):
    run_adm1(maize_slider.value, grass_slider.value, food_slider.value, cattle_slider.value,
             v_slider.value, q_slider.value, t_slider.value, sim_period_slider.value, output)

def on_reset_clicked(b):
    maize_slider.value = 50
    grass_slider.value = 30
    food_slider.value = 10
    cattle_slider.value = 10
    v_slider.value = 7000
    q_slider.value = 136.63
    t_slider.value = 45
    sim_period_slider.value = 70
    with output:
        clear_output(wait=True)
        print('Sliders reset to defaults.')

run_button.on_click(on_run_clicked)
reset_button.on_click(on_reset_clicked)

# Display interface
display(VBox([
    HBox([feedstock_box, process_box]), 
    HBox([run_button, reset_button]), 
    output
]))
