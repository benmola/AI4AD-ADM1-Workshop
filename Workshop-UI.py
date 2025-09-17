# !git clone https://github.com/benmola/AI4AD-ADM1-Workshop..git
# %cd AI4AD-ADM1-Workshop.

# !pip install -r requirements.txt -q

from ADM1 import ADM1Simulator
import plotly.graph_objects as go
import plotly.io as pio
import kaleido
import pandas as pd
from ipywidgets import FloatSlider, Button, VBox, HBox, Output, HTML, Layout
from IPython.display import display, FileLink, clear_output
from tqdm import tqdm
import os

# Set Plotly renderer for Colab compatibility
pio.renderers.default = 'colab'

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
        with tqdm(total=days, desc='Simulating Days') as pbar:
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
            .set_properties(**{'font-size': '12pt', 'border': '1px solid lightgrey', 'text-align': 'center'}) \
            .set_caption('📊 Steady-State Process Indicators') \
            .set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#2E7D32'), ('color', 'white'), ('font-weight', 'bold'), ('padding': '8px')]},
                {'selector': 'td',
                 'props': [('padding': '8px')]},
                {'selector': 'caption',
                 'props': [('caption-side', 'top'), ('font-size': '1.3em'), ('font-weight', 'bold'), ('padding': '10px')]}
            ])

        display(styled_summary)

        # Animated Plotly plots
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Methane Flow (m³/d)', line=dict(color='royalblue', width=3)))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Biogas Flow (m³/d)', line=dict(color='mediumorchid', width=3)))

        max_time = output_data['time'].max()
        max_flow = max(output_data['q_gas'].max(), output_data['q_ch4'].max()) * 1.1
        fig.update_layout(
            title=dict(text='ADM1 Simulation: Biogas and Methane Flow Rates', font=dict(size=20)),
            xaxis_title='Time (days)',
            yaxis_title='Flow Rate (m³/d)',
            showlegend=True,
            xaxis=dict(range=[0, max_time], title_font=dict(size=16), tickfont=dict(size=14)),
            yaxis=dict(range=[0, max_flow], title_font=dict(size=16), tickfont=dict(size=14)),
            legend=dict(font=dict(size=14)),
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
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        frames = [go.Frame(
            data=[
                go.Scatter(x=output_data['time'][:k], y=output_data['q_ch4'][:k], mode='lines', name='Methane Flow (m³/d)'),
                go.Scatter(x=output_data['time'][:k], y=output_data['q_gas'][:k], mode='lines', name='Biogas Flow (m³/d)')
            ],
            name=f'Frame {k}'
        ) for k in range(1, len(output_data) + 1)]
        fig.frames = frames

        fig.show()

        # Export button
        export_button = Button(description='Export Results to CSV', style=dict(button_color='#4CAF50', font_weight='bold'))
        def on_export_clicked(b):
            filename = 'ADM1_simulation_results.csv'
            output_data.to_csv(filename, index=False)
            display(FileLink(filename))
            print(f'✅ CSV exported: {filename}')
        export_button.on_click(on_export_clicked)
        display(export_button)

        print('✅ Simulation Complete.')

# Output widget
output = Output(layout=Layout(border='1px solid lightgrey', padding='10px', margin='10px 0px'))

# Slider layout for consistency
slider_layout = Layout(width='400px')
value_style = {'description_width': '120px'}

# Updated sliders with professional labels
maize_slider = FloatSlider(min=0, max=100, step=5, value=0, description='Maize Silage (%):', layout=slider_layout, style=value_style)
grass_slider = FloatSlider(min=0, max=100, step=10, value=0, description='Grass Silage (%):', layout=slider_layout, style=value_style)
food_slider = FloatSlider(min=0, max=100, step=10, value=90, description='Food Waste (%):', layout=slider_layout, style=value_style)
cattle_slider = FloatSlider(min=0, max=100, step=10, value=10, description='Cattle Slurry (%):', layout=slider_layout, style=value_style)
v_slider = FloatSlider(min=1000, max=10000, step=500, value=7500, description='Reactor Volume (m³):', layout=slider_layout, style=value_style)
q_slider = FloatSlider(min=50, max=500, step=10, value=136.63, description='Influent Flow (m³/d):', layout=slider_layout, style=value_style)
t_slider = FloatSlider(min=25, max=65, step=1, value=35, description='Temperature (°C):', layout=slider_layout, style=value_style)
sim_period_slider = FloatSlider(min=50, max=365, step=5, value=100, description='Simulation Days:', layout=slider_layout, style=value_style)

# Buttons with styling
run_button = Button(description='Run Simulation', style=dict(button_color='#2196F3', font_weight='bold', description_width='initial'), layout=Layout(width='200px', margin='10px'))
reset_button = Button(description='Reset Sliders', style=dict(button_color='#FF5722', font_weight='bold', description_width='initial'), layout=Layout(width='200px', margin='10px'))

def on_run_clicked(b):
    total_ratio = maize_slider.value + grass_slider.value + food_slider.value + cattle_slider.value
    if abs(total_ratio - 100) > 0.01:
        with output:
            clear_output(wait=True)
            print('⚠️ Error: Feedstock ratios must sum to 100%. Current sum: {:.2f}%'.format(total_ratio))
        return
    run_adm1(maize_slider.value, grass_slider.value, food_slider.value, cattle_slider.value,
             v_slider.value, q_slider.value, t_slider.value, sim_period_slider.value, output)

def on_reset_clicked(b):
    maize_slider.value = 0
    grass_slider.value = 0
    food_slider.value = 90
    cattle_slider.value = 10
    v_slider.value = 7500
    q_slider.value = 136.63
    t_slider.value = 35
    sim_period_slider.value = 100
    with output:
        clear_output(wait=True)
        print('✅ Sliders reset to default values.')

run_button.on_click(on_run_clicked)
reset_button.on_click(on_reset_clicked)

# Display interface with title
display(HTML('<h2 style="text-align: center; color: #333; margin-bottom: 20px;">ADM1 Anaerobic Digestion Simulator</h2>'))
display(HBox([
    VBox([HTML('<b style="font-size: 16px; color: #4CAF50;">Feedstock Composition</b>'), maize_slider, grass_slider, food_slider, cattle_slider], layout=Layout(border='1px solid lightgrey', padding='15px', margin='10px', width='50%')),
    VBox([HTML('<b style="font-size: 16px; color: #4CAF50;">Reactor Parameters</b>'), v_slider, q_slider, t_slider, sim_period_slider], layout=Layout(border='1px solid lightgrey', padding='15px', margin='10px', width='50%'))
], layout=Layout(justify_content='space-around')))
display(HBox([run_button, reset_button], layout=Layout(justify_content='center')), output)
