!git clone https://github.com/benmola/AI4AD-ADM1-Workshop..git
%cd AI4AD-ADM1-Workshop.

!pip install -r requirements.txt -q

from ADM1 import ADM1Simulator
import plotly.graph_objects as go
import plotly.io as pio
import kaleido
import pandas as pd
from ipywidgets import FloatSlider, Button, VBox, HBox, Output, Accordion
from IPython.display import display, FileLink, clear_output
from tqdm import tqdm
import os

# Set Plotly renderer for better compatibility in Colab/Colab
pio.renderers.default = 'colab'  # Use 'notebook' for classic Jupyter if needed

def get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry):
    ratios = {
        'Maize Silage': maize_silage,
        'Grass Silage': grass_silage,
        'Food Waste': food_waste,
        'Cattle Slurry': cattle_slurry
    }
    total = sum(ratios.values())
    if total == 0:
        return ratios
    normalized = {k: v / total for k, v in ratios.items()}
    if abs(sum(normalized.values()) - 1) > 0.01:
        print('⚠️ Warning: Normalized ratios do not sum to exactly 1 (possible rounding issue).')
    return normalized
def run_adm1(maize_silage, grass_silage, food_waste, cattle_slurry, HRT, V, Q, T, output):
    with output:
        clear_output(wait=True)
        ratios = get_feedstock_ratios(maize_silage, grass_silage, food_waste, cattle_slurry)
        days = int(HRT)
        simulator = ADM1Simulator(ratios, days=days, Q=Q, V=V, T=T)

        # Simulate with progress bar
        with tqdm(total=days, desc='Simulating Days', disable=True) as pbar:
            simulator.run()
            pbar.update(days)

        output_data = simulator.get_results()
        if output_data.empty:
            print('❌ Simulation returned no data.')
            return

        # Patch for compatibility
        if 'V' not in output_data.columns:
            output_data['V'] = V

        # Steady-state summary as styled table (Process Indicators only for non-time-series)
        summary_data = {
            'Process Indicator': ['pH', 'FOS', 'TAC', 'FOS/TAC', 'Gas Pressure'],
            'Steady-State Value': [
                output_data['pH'].iloc[-1],
                output_data['FOS'].iloc[-1],
                output_data['TAC'].iloc[-1],
                output_data['FOS/TAC'].iloc[-1],
                output_data['p_gas'].iloc[-1]
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        # Apply enhanced styling
        styled_summary = summary_df.style \
            .format({'Steady-State Value': '{:.2f}'}) \
            .background_gradient(cmap='viridis', subset=['Steady-State Value']) \
            .set_properties(**{'font-size': '10pt', 'border': '1px solid black'}) \
            .set_caption('📊 Steady-State Process Indicators (Biogas Plant Style)') \
            .set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td',
                 'props': [('text-align', 'center')]},
                {'selector': 'caption',
                 'props': [('caption-side', 'top'), ('font-size', '1.2em'), ('font-weight', 'bold')]}
            ])


        display(styled_summary)

        # Animated Plotly plots for Biogas and Methane only (Outputs)
        fig = go.Figure()

        # Initial traces (empty for animation start)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Methane Flow (Output)', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Biogas Flow (Output)', line=dict(color='purple', width=2)))

        # Layout with animation controls
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
            hovermode='x unified'  # Better interactivity
        )

        # Animation frames (build incrementally for real-time feel)
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

        print('✅ Simulation Complete. Explore the interactive plot above!')

# Output widget
output = Output()

# Sliders
maize_slider = FloatSlider(min=0, max=1, step=0.1, value=0.5, description='Maize Silage')
grass_slider = FloatSlider(min=0, max=1, step=0.1, value=0.3, description='Grass Silage')
food_slider = FloatSlider(min=0, max=1, step=0.1, value=0.1, description='Food Waste')
cattle_slider = FloatSlider(min=0, max=1, step=0.1, value=0.1, description='Cattle Slurry')
hrt_slider = FloatSlider(min=10, max=80, step=2, value=48, description='HRT (days)')
v_slider = FloatSlider(min=1000, max=10000, step=500, value=6520, description='V (m³)')
q_slider = FloatSlider(min=50, max=500, step=10, value=136.63, description='Q (m³/d)')
t_slider = FloatSlider(min=15, max=60, step=1, value=45, description='T (°C)')

# Group into accordions
feedstock_acc = Accordion(children=[VBox([maize_slider, grass_slider, food_slider, cattle_slider])])
feedstock_acc.set_title(0, 'Feedstock Ratios')
process_acc = Accordion(children=[VBox([hrt_slider, v_slider, q_slider, t_slider])])
process_acc.set_title(0, 'Process Parameters')

# Buttons
run_button = Button(description='Run Simulation')
reset_button = Button(description='Reset Sliders')

def on_run_clicked(b):
    run_adm1(maize_slider.value, grass_slider.value, food_slider.value, cattle_slider.value,
             hrt_slider.value, v_slider.value, q_slider.value, t_slider.value, output)

def on_reset_clicked(b):
    maize_slider.value = 0.5
    grass_slider.value = 0.3
    food_slider.value = 0.1
    cattle_slider.value = 0.1
    hrt_slider.value = 48
    v_slider.value = 6520
    q_slider.value = 136.63
    t_slider.value = 45
    with output:
        clear_output(wait=True)
        print('Sliders reset to defaults.')

run_button.on_click(on_run_clicked)
reset_button.on_click(on_reset_clicked)

# Display interface
display(VBox([HBox([feedstock_acc, process_acc]), HBox([run_button, reset_button]), output]))
