# !git clone https://github.com/benmola/AI4AD-ADM1-Workshop..git
# %cd AI4AD-ADM1-Workshop.

# !pip install -r requirements.txt -q

from ADM1 import ADM1Simulator
import plotly.graph_objects as go
import plotly.io as pio
import kaleido
import pandas as pd
from ipywidgets import FloatSlider, Button, VBox, HBox, Output, Accordion
from IPython.display import display, FileLink, clear_output
from tqdm import tqdm
import os

# # Set Plotly renderer for better compatibility in Colab/Colab
# pio.renderers.default = 'colab'

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
    if abs(sum(normalized.values()) - 1) > 0.01:
        print('⚠️ Warning: Normalized ratios do not sum to exactly 1 (possible rounding issue).')
    return normalized

def run_adm1(maize_silage, grass_silage, food_waste, cattle_slurry, V, Q, T, sim_days, output):
    with output:
        clear_output(wait=True)
        ratios = get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry)
        
        # Calculate actual HRT from V and Q
        actual_HRT = V / Q
        days = int(sim_days)  # Use simulation period instead of HRT
        
        
        simulator = ADM1Simulator(ratios, days=days, Q=Q, V=V, T=T)

        # Simulate with progress bar
        with tqdm(total=days, desc='Simulating Days', disable=True) as pbar:
            simulator.run()
            pbar.update(days)

        output_data = simulator.get_results()
        if output_data.empty:
            print('❌ Simulation returned no data.')
            return

        # Steady-state summary - TRANSPOSED
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
            .set_caption('📊 Steady-State Process Indicators') \
            .set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td',
                 'props': [('text-align', 'center')]},
                {'selector': 'caption',
                 'props': [('caption-side', 'top'), ('font-size', '1.2em'), ('font-weight', 'bold')]}
            ])

        display(styled_summary)

        # Animated Plotly plots
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Methane Flow (Output)', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Biogas Flow (Output)', line=dict(color='purple', width=2)))

        max_time = output_data['time'].max()
        max_flow = max(output_data['q_gas'].max(), output_data['q_ch4'].max()) * 1.1
        fig.update_layout(
            title='ADM1 Simulation Outputs: Biogas and Methane Flow Rates Over Time',
            xaxis_title='Time (days)',
            yaxis_title='Flow Rate (m³/d)',
            showlegend=True,
            xaxis=dict(range=[0, max_time]),
            yaxis=dict(range=[0, max_flow]),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}],
                        'label': '▶️ Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'label': '⏸️ Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': True,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            hovermode='x unified'
        )

        frames = [go.Frame(
            data=[
                go.Scatter(x=output_data['time'][:k], y=output_data['q_ch4'][:k], mode='lines', name='Methane Flow (Output)'),
                go.Scatter(x=output_data['time'][:k], y=output_data['q_gas'][:k], mode='lines', name='Biogas Flow (Output)')
            ],
            name=f'Frame {k}'
        ) for k in range(1, len(output_data) + 1)]
        fig.frames = frames

        fig.show()

        # Export button
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

# Updated sliders -  added simulation period
maize_slider = FloatSlider(min=0, max=100, step=5, value=50, description='Maize Silage')
grass_slider = FloatSlider(min=0, max=100, step=10, value=30, description='Grass Silage')
food_slider = FloatSlider(min=0, max=100, step=10, value=10, description='Food Waste')
cattle_slider = FloatSlider(min=0, max=100, step=10, value=10, description='Cattle Slurry')
v_slider = FloatSlider(min=1000, max=10000, step=500, value=7000, description='Volume (m³)')
q_slider = FloatSlider(min=50, max=500, step=10, value=136.63, description='Flow Q (m³/d)')
t_slider = FloatSlider(min=25, max=65, step=1, value=45, description='Temp (°C)')
sim_period_slider = FloatSlider(min=50, max=100, step=5, value=70, description='Sim Days')

# # Group into accordions
feedstock_box = VBox([maize_slider, grass_slider, food_slider, cattle_slider])
feedstock_acc = Accordion(children=[feedstock_box])
feedstock_acc.set_title(0, 'Feedstock Mix (%)')
feedstock_acc.selected_index = None  # <-- keeps it collapsed by default (title always visible)

process_box = VBox([v_slider, q_slider, t_slider, sim_period_slider])
process_acc = Accordion(children=[process_box])
process_acc.set_title(0, 'Process Parameters')
process_acc.selected_index = None




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
    sim_period_slider.value = 150
    with output:
        clear_output(wait=True)
        print('Sliders reset to defaults.')

run_button.on_click(on_run_clicked)
reset_button.on_click(on_reset_clicked)

# # Display interface
# #display(VBox([HBox([feedstock_acc, process_acc]), HBox([run_button, reset_button]), output]))
display(HBox([feedstock_acc, process_acc]), HBox([run_button, reset_button]), output)



