import numpy as np
import pandas as pd
import scipy.integrate

class ADM1Simulator:
    def __init__(self, feedstock_ratios, days=100, Q=100.0, V=1000.0, T=35):
        """
        Initialize the simulator with feedstock ratios and simulation parameters.
        feedstock_ratios: dict with keys as feedstock names and values as ratios (sum to 1)
        days: number of simulation days
        Q: influent flow rate (default 100.0)
        V: reactor volume (default 1000.0)
        T: temperature in Celsius (default 35)
        """
        self.feedstock_ratios = feedstock_ratios
        self.days = days
        self.Q = Q
        self.V = V
        self.T = T + 273.15  # Convert to Kelvin
        self.simulate_results = None
        self.output_data = None

        # Feedstock composition data (same as Full_ADM1)
        self.feedstocks = ["Maize Silage", "Grass Silage", "Food Waste", "Cattle Slurry"]
        self.feedstock_data = {
            "Maize Silage": [202.5, 16.1, 11.8, 2.6, 0.364, 0.56, 0.2, 1.583, 718, 0.15, 0.02],
            "Grass Silage": [177.8, 10.8, 1.4, 3.9, 0.361, 0.893, 0.292, 1.71, 941, 0.15, 0.02],
            "Food Waste": [177.8, 5.6, 6.1, 4.29, 1.056, 0.924, 0.33, 1.378, 775, 0.07, 0.02],
            "Cattle Slurry": [177.8, 42.7, 15.4, 0.85, 0.212, 0.198, 0.235, 1.378, 356, 0.07, 0.02]
        }
        self.characteristics = [
            'carbohydrates', 'proteins', 'lipids',
            'inorganic nitrogen (TAN)', 'moisture content', 'cation', 'anion'
        ]
        self.column_mapping = {
            'carbohydrates': 'X_ch',
            'proteins': 'X_pr',
            'lipids': 'X_li',
            'inorganic nitrogen (TAN)': 'S_IN',
            'moisture content': 'S_h2o',
            'cation': 'S_cation',
            'anion': 'S_anion'
        }

    def run_codigestion(self):
        """Generate influent DataFrame from feedstock ratios and composition (adapted for ADM1-R3)."""
        # Weighted sum for each characteristic
        renamed_characteristics = [self.column_mapping.get(char, char) for char in self.characteristics]
        codigested_values = {new_char: [] for new_char in renamed_characteristics}
        for day in range(self.days):
            for char, new_char in zip(self.characteristics, renamed_characteristics):
                char_values = []
                for feed in self.feedstocks:
                    idx = self.characteristics.index(char)
                    value = self.feedstock_data[feed][idx]
                    ratio = self.feedstock_ratios.get(feed, 0.0)
                    char_values.append(ratio * value)
                weighted_value = sum(char_values)
                codigested_values[new_char].append(weighted_value)

        # Aggregate VFAs into S_ac (sum of acetate, propionate, butyrate, valerate)
        codigested_values['S_ac'] = []
        for day in range(self.days):
            vfa_values = []
            for feed in self.feedstocks:
                vfa_sum = sum(self.feedstock_data[feed][3:7])  # indices 3-6: ac, pro, bu, va
                ratio = self.feedstock_ratios.get(feed, 0.0)
                vfa_values.append(ratio * vfa_sum)
            codigested_values['S_ac'].append(sum(vfa_values))

        # Add time, Q, V, T, V_gas columns
        codigested_values['time'] = list(range(self.days))
        codigested_values['Q'] = [self.Q] * self.days
        codigested_values['V'] = [self.V] * self.days
        codigested_values['T'] = [self.T] * self.days
        codigested_values['V_gas'] = [0.3 * self.V] * self.days  # Assume V_gas = 0.3 * V_liq

        # Set default zero values for other columns
        zero_columns = ['S_ch4', 'S_IC', 'X_bac', 'X_ac']
        for col in zero_columns:
            codigested_values[col] = [0.0] * self.days

        # Template columns for ADM1-R3
        template_columns = [
            'time', 'S_ac', 'S_ch4', 'S_IC', 'S_IN', 'S_h2o', 'X_ch', 'X_pr', 'X_li', 'X_bac', 'X_ac',
            'S_cation', 'S_anion', 'Q', 'T', 'V', 'V_gas',
            'S_ac_ion', 'S_hco3_ion', 'S_nh3', 'S_gas_ch4', 'S_gas_co2', 'pH'
        ]
        result_df = pd.DataFrame()
        for col in template_columns:
            result_df[col] = codigested_values.get(col, [0] * self.days)
        return result_df

    def run_simulation(self, influent_df):
        """Run the ADM1-R3 simulation using the influent DataFrame."""
        # Constants from ADM1-R3
        R = 0.083145
        p_atm = 1.0133
        p_h2o = 0.0313
        K_H_ch4 = 0.0011
        K_H_co2 = 0.025
        k_La = 200.0
        k_p = 50000
        k_dec = 0.02
        k_ch = 0.25670070949996215
        k_pr = 0.23201964427542568
        k_li = 0.18043930268695954
        pK_l_ac = 6
        pK_u_ac = 7
        K_I_IN = 0.0017
        K_w = 2.078771055954360e-14
        k_AB_IN = 10000000000
        k_AB_ac = 10000000000
        k_AB_co2 = 10000000000
        K_I_nh3 = 0.0306
        K_a_IN = 1.110286652708067e-09
        K_a_ac = 1.737800828749374e-05
        K_a_co2 = 4.937073397534361e-07
        k_m_ac = 0.4  # Initial, can be optimized externally if needed
        K_ac = 0.14   # Initial, can be optimized externally if needed
        rtol = 1e-6
        atol = 1e-8

        # Initial state from ADM1-R3
        state_zero = [
            0.02202321431773031, 0.018428118130450387, 5.251770087877549, 1.063413245579135, 825.5148937557987,
            13.638734129765595, 1.9275086207501004, 1.1373607051400034, 9.132926183321803, 2.238103730133979,
            0.10206581575041813, 0.01367566168668581, 0.049315335796598116, 4.545537837111362, 0.022397061704351174,
            0.4300774656755433, 1.0353603800104891
        ]
        columns = [
            'S_ac', 'S_ch4', 'S_IC', 'S_IN', 'S_h2o', 'X_ch', 'X_pr', 'X_li', 'X_bac', 'X_ac',
            'S_cation', 'S_anion', 'S_ac_ion', 'S_hco3_ion', 'S_nh3', 'S_gas_ch4', 'S_gas_co2'
        ]

        # ODE function adapted from ADM1-R3
        def ADM1_R3_ODE(t, y, state_input):
            S_ac, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, X_li, X_bac, X_ac, S_cation, S_anion, S_ac_ion, S_hco3_ion, S_nh3, S_gas_ch4, S_gas_co2 = y
            S_ac_in, S_ch4_in, S_IC_in, S_IN_in, S_h2o_in, X_ch_in, X_pr_in, X_li_in, X_bac_in, X_ac_in, S_cation_in, S_anion_in, T_ad, q_ad, V_liq, V_gas = state_input

            S_nh4_i = S_IN - S_nh3
            S_co2 = S_IC - S_hco3_ion
            phi = S_cation + S_nh4_i / 17 - S_hco3_ion / 44 - S_ac_ion / 60 - S_anion
            S_H = -phi * 0.5 + 0.5 * np.sqrt(phi * phi + 4 * K_w)
            pH = -np.log10(S_H + 1e-10)

            p_ch4 = S_gas_ch4 * R * T_ad / 16
            p_co2 = S_gas_co2 * R * T_ad / 44
            p_gas = p_ch4 + p_co2 + p_h2o
            q_gas = k_p * (p_gas - p_atm) * p_gas / p_atm

            # Inhibition functions
            I_0 = S_IN / (S_IN + K_I_IN)
            I_5 = 10 ** (-3 / (pK_u_ac - pK_l_ac) * (pK_l_ac + pK_u_ac) / 2) / (S_H ** (3 / (pK_u_ac - pK_l_ac)) + 10 ** (-3 / (pK_u_ac - pK_l_ac) * (pK_l_ac + pK_u_ac) / 2))
            I_7 = K_I_nh3 / (K_I_nh3 + S_nh3)

            # Rate equations
            rate_0 = k_ch * X_ch
            rate_1 = k_pr * X_pr
            rate_2 = k_li * X_li
            rate_3 = k_m_ac * S_ac / (K_ac + S_ac) * X_ac * I_0 * I_5 * I_7
            rate_4 = k_dec * X_bac
            rate_5 = k_dec * X_ac
            rate_6 = k_AB_ac * (S_ac_ion * (K_a_ac + S_H) - K_a_ac * S_ac)
            rate_7 = k_AB_co2 * (S_hco3_ion * (K_a_co2 + S_H) - K_a_co2 * S_IC)
            rate_8 = k_AB_IN * (S_nh3 * (K_a_IN + S_H) - K_a_IN * S_IN)
            rate_9 = k_La * (S_ch4 - 16 * (K_H_ch4 * p_ch4))
            rate_10 = k_La * (S_co2 - 44 * (K_H_co2 * p_co2))

            # Process rates
            process_0 = 0.6555 * rate_0 + 0.9947 * rate_1 + 1.7651 * rate_2 - 26.5447 * rate_3
            process_1 = 0.081837 * rate_0 + 0.069636 * rate_1 + 0.19133 * rate_2 + 6.7367 * rate_3 - rate_9
            process_2 = 0.2245 * rate_0 + 0.10291 * rate_1 - 0.64716 * rate_2 + 18.4808 * rate_3 - rate_10
            process_3 = -0.016932 * rate_0 + 0.17456 * rate_1 - 0.024406 * rate_2 - 0.15056 * rate_3
            process_4 = -0.057375 * rate_0 - 0.47666 * rate_1 - 0.44695 * rate_2 + 0.4778 * rate_3
            process_5 = -rate_0 + 0.18 * rate_4 + 0.18 * rate_5
            process_6 = -rate_1 + 0.77 * rate_4 + 0.77 * rate_5
            process_7 = -rate_2 + 0.05 * rate_4 + 0.05 * rate_5
            process_8 = 0.11246 * rate_0 + 0.13486 * rate_1 + 0.1621 * rate_2 - rate_4
            process_9 = rate_3 - rate_5
            process_10 = 0
            process_11 = 0
            process_12 = -rate_6
            process_13 = -rate_7
            process_14 = -rate_8
            process_15 = (V_liq / V_gas) * rate_9
            process_16 = (V_liq / V_gas) * rate_10

            dx = [
                q_ad * (S_ac_in - S_ac) / V_liq + process_0,
                q_ad * (S_ch4_in - S_ch4) / V_liq + process_1,
                q_ad * (S_IC_in - S_IC) / V_liq + process_2,
                q_ad * (S_IN_in - S_IN) / V_liq + process_3,
                q_ad * (S_h2o_in - S_h2o) / V_liq + process_4,
                q_ad * (X_ch_in - X_ch) / V_liq + process_5,
                q_ad * (X_pr_in - X_pr) / V_liq + process_6,
                q_ad * (X_li_in - X_li) / V_liq + process_7,
                q_ad * (X_bac_in - X_bac) / V_liq + process_8,
                q_ad * (X_ac_in - X_ac) / V_liq + process_9,
                q_ad * (S_cation_in - S_cation) / V_liq + process_10,
                q_ad * (S_anion_in - S_anion) / V_liq + process_11,
                process_12,
                process_13,
                process_14,
                -S_gas_ch4 * q_gas / V_gas + process_15,
                -S_gas_co2 * q_gas / V_gas + process_16
            ]
            return dx

        # Simulation setup
        t = influent_df['time'].values
        state_list = [state_zero]
        gasflow_list = [{'time': 0, 'q_gas': 0, 'q_ch4': 0}]
        pressure_list = [{'time': 0, 'p_ch4': 0, 'p_co2': 0, 'p_h2o': 0, 'p_gas': 0}]
        extras_list = [{'time': 0, 'pH': 7.0, 'OLR': 0, 'HRT': 0, 'FOS': 0, 'TAC': 0, 'FOS/TAC': 0}]

        # Helper function for OLR
        def total_concentration(row):
            return (row['S_ac'] + row['S_ch4'] + row['X_ch'] + row['X_pr'] + row['X_li'] +
                    row['X_bac'] + row['X_ac'])

        # Simulation loop
        for i in range(1, len(t)):
            current_influent = influent_df.iloc[i]
            state_input = [
                current_influent[col] for col in ['S_ac', 'S_ch4', 'S_IC', 'S_IN', 'S_h2o', 'X_ch', 'X_pr',
                                                 'X_li', 'X_bac', 'X_ac', 'S_cation', 'S_anion', 'T', 'Q',
                                                 'V', 'V_gas']
            ]
            t_step = [t[i-1], t[i]]
            result = scipy.integrate.solve_ivp(
                ADM1_R3_ODE, t_step, state_list[-1], method='BDF', args=(state_input,), rtol=rtol, atol=atol
            )
            new_state = result.y[:, -1]
            state_list.append(new_state)

            S_ac, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, X_li, X_bac, X_ac, S_cation, S_anion, S_ac_ion, S_hco3_ion, S_nh3, S_gas_ch4, S_gas_co2 = new_state
            T_ad = state_input[-4]
            q_ad = state_input[-3]
            V_liq = state_input[-2]
            V_gas = state_input[-1]
            p_ch4 = S_gas_ch4 * R * T_ad / 16
            p_co2 = S_gas_co2 * R * T_ad / 44
            p_gas = p_ch4 + p_co2 + p_h2o
            q_gas = k_p * (p_gas - p_atm) * p_gas / p_atm
            q_ch4 = q_gas * (p_ch4 / p_gas) if p_gas > 0 else 0

            S_nh4_i = S_IN - S_nh3
            S_co2 = S_IC - S_hco3_ion
            phi = S_cation + S_nh4_i / 17 - S_hco3_ion / 44 - S_ac_ion / 60 - S_anion
            S_H = -phi * 0.5 + 0.5 * np.sqrt(phi * phi + 4 * K_w)
            pH = -np.log10(S_H + 1e-10)

            OLR = total_concentration(current_influent) * q_ad / V_liq
            HRT = V_liq / q_ad if q_ad > 0 else 0
            FOS = ((S_ac + S_ac_ion) / 60) * 1000 * 1000
            TAC = (S_anion + S_hco3_ion + S_ac_ion / 60 - S_cation) * 1000
            FOS_TAC = FOS / TAC if TAC != 0 else 0

            gasflow_list.append({'time': t[i], 'q_gas': q_gas, 'q_ch4': q_ch4})
            pressure_list.append({'time': t[i], 'p_ch4': p_ch4, 'p_co2': p_co2, 'p_h2o': p_h2o, 'p_gas': p_gas})
            extras_list.append({'time': t[i], 'pH': pH, 'OLR': OLR, 'HRT': HRT, 'FOS': FOS, 'TAC': TAC, 'FOS/TAC': FOS_TAC})

        # Convert lists to DataFrames
        self.simulate_results = pd.DataFrame(state_list, columns=columns)
        gasflow = pd.DataFrame(gasflow_list)
        pressure_results = pd.DataFrame(pressure_list)
        extras = pd.DataFrame(extras_list)
        self.output_data = pd.concat([
            influent_df['time'],
            gasflow[['q_gas', 'q_ch4']],
            pressure_results[['p_ch4', 'p_co2', 'p_h2o', 'p_gas']],
            extras[['pH', 'OLR', 'HRT', 'FOS', 'TAC', 'FOS/TAC']]
        ], axis=1)

    def run(self):
        """Execute the full simulation process."""
        print("ADM1 Simulation in Progress....")
        influent_df = self.run_codigestion()
        self.run_simulation(influent_df)

    def get_results(self):
        """Return only process indicators for plotting."""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        return self.output_data

    def save_output(self, file_path="Output/ADM1R3_Process_Indicators.xlsx"):
        """Save only process indicators to an Excel file."""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        with pd.ExcelWriter(file_path) as writer:
            self.simulate_results.to_excel(writer, sheet_name='ADM1_R3_States', index=False)
            self.output_data.to_excel(writer, sheet_name='Process_Data', index=False)
        print(f"Simulation results saved to {file_path}")


