# AI4AD ADM1 Workshop: Simulating Anaerobic Digestion with Python

<p align="center">
  <img src="https://raw.githubusercontent.com/benmola/AI4AD-ADM1-Workshop/main/image.png" alt="Anaerobic Digestion Model No. 1 Workshop" width="600">
</p>

## Overview
This repository provides an interactive workshop for biogas researchers, AD plant operators, consultants, and sustainability professionals to explore anaerobic digestion (AD) modeling using the Anaerobic Digestion Model No. 1 (ADM1). The core focus is on mechanistic simulation of biogas production and process stability through Python-based notebooks. No prior coding experience is required—run simulations directly in Jupyter or Google Colab to test feedstock mixes, operating conditions, and visualize outcomes.

The primary notebook (**PyADM1-R4-Workshop.ipynb**) introduces a simplified ADM1 implementation for hands-on experimentation with digester dynamics. An additional interactive UI (**UI-Workshop.ipynb**) offers sliders for quick scenario testing as an extra feature.

## Purpose
- Simulate AD processes to predict methane yield, pH stability, and FOS/TAC ratios.
- Experiment with UK-based feedstocks like maize silage, grass silage, food waste, and cattle slurry.
- Understand how parameters (e.g., temperature, flow rate, reactor volume) impact biogas output and digester health.
- Ideal for educational workshops, research prototyping, or operational optimization in renewable energy.

## Key Features
- **ADM1 Simulator**: Core Python code in `ADM1.py` for running simulations.
- **Notebooks**:
  - **PyADM1-R4-Workshop.ipynb**: Main tutorial on ADM1 mechanics, influent setup, and result analysis.
  - **UI-Workshop.ipynb**: Optional interactive interface with sliders for feedstock ratios and parameters.
- **Outputs**: Interactive plots (via Plotly), summary plots for methane/biogas flows, pH, and stability metrics.
- **No Installation Needed**: Run in Google Colab or locally with Jupyter.

## Quick Start
1. Clone the repo: `git clone https://github.com/benmola/AI4AD-ADM1-Workshop.git`
2. Navigate: `cd AI4AD-ADM1-Workshop`
3. Install dependencies: `pip install -r requirements.txt`
4. Open and run **PyADM1-R4-Workshop.ipynb** (main) or **UI-Workshop.ipynb** (interactive) in Jupyter/Colab.
5. Adjust feedstocks/parameters and simulate—results include animated plots and steady-state summaries.

## Requirements
- Python 3.8+
- Libraries: numpy, pandas, scipy, matplotlib, ipywidgets, plotly, kaleido (listed in `requirements.txt`)

## License
MIT License - See [LICENSE](LICENSE) for details.

For questions or contributions, open an issue or contact the maintainer. Happy simulating! 🌱
