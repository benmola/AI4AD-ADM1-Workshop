import numpy as np
import pandas as pd
import scipy.integrate

class ADM1Simulator:
    def __init__(self, feedstock_ratios, days=100, Q=100.0, V=1000.0, T=35):
        """
        Initialize the simulator with feedstock ratios and simulation parameters.
        feedstock_ratios: dict with keys as feedstock names and values as ratios (sum to 1)
        days: number of simulation days
        Q: influent flow rate (default 1.0)
        V: reactor volume (default 1.0)
        T: temperature in Kelvin (default 308.15)
        """
        self.feedstock_ratios = feedstock_ratios
        self.days = days
        self.Q = Q
        self.V = V
        self.T = T + 273.15  # Convert to Kelvin
        self.simulate_results = None
        self.output_data = None

        # Feedstock composition data
        self.feedstocks = ["Maize Silage", "Grass Silage", "Food Waste", "Cattle Slurry"]
        self.feedstock_data = {
            "Maize Silage": [202.5, 16.1, 11.8, 2.6, 0.364, 0.56, 0.2, 1.583, 718, 0.15, 0.02],
            "Grass Silage": [177.8, 10.8, 1.4, 3.9, 0.361, 0.893, 0.292, 1.71, 941, 0.15, 0.02],
            "Food Waste": [177.8, 5.6, 6.1, 4.29, 1.056, 0.924, 0.33, 1.378, 775, 0.07, 0.02],
            "Cattle Slurry": [177.8, 82.7, 35.4, 0.45, 0.112, 0.098, 0.035, 1.378, 156, 0.07, 0.02]
        }
        self.characteristics = [
            'carbohydrates', 'proteins', 'lipids', 'acetate', 'propionate', 'butyrate', 'valerate',
            'inorganic nitrogen (TAN)', 'moisture content', 'cation', 'anion'
        ]
        self.column_mapping = {
            'carbohydrates': 'X_ch',
            'proteins': 'X_pr',
            'lipids': 'X_li',
            'acetate': 'S_ac',
            'propionate': 'S_pro',
            'butyrate': 'S_bu',
            'valerate': 'S_va',
            'inorganic nitrogen (TAN)': 'S_IN',
            'moisture content': 'S_h2o',
            'cation': 'S_cation',
            'anion': 'S_anion'
        }

    def run_codigestion(self):
        """Generate influent DataFrame from feedstock ratios and composition."""
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
        # Add time, Q, V, T columns
        codigested_values['time'] = list(range(self.days))
        codigested_values['Q'] = [self.Q] * self.days
        codigested_values['V'] = [self.V] * self.days
        codigested_values['T'] = [self.T] * self.days

        # Template columns for ADM1
        template_columns = [
            'time', 'S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac', 'S_h2', 'S_ch4', 
            'S_IC', 'S_IN', 'S_h2o', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa', 'X_va', 
            'X_bu', 'X_pro', 'X_ac', 'X_h2', 'S_cation', 'S_anion', 'Q', 'T', 'V',  # Added 'V' here
            'S_va_ion', 'S_bu_ion', 'S_pro_ion', 'S_ac_ion', 'S_hco3_ion', 'S_nh3', 
            'S_gas_h2', 'S_gas_ch4', 'S_gas_co2', 'pH'
        ]
        result_df = pd.DataFrame()
        for col in template_columns:
            result_df[col] = codigested_values[col] if col in codigested_values else 0
        return result_df

    def run_simulation(self, influent_df):
        """Run the ADM1 simulation using the influent DataFrame."""
        # Constants
        R = 0.083145
        p_atm = 1.0133
        K_H_ch4 = 0.0011
        K_H_co2 = 0.025
        K_H_h2 = 0.00072
        K_I_IN = 0.0017
        K_I_nh3 = 0.0306
        K_I_c4 = 1.3e-06
        K_I_fa = 6.3e-07
        K_I_pro = 4.4e-07
        K_a_IN = 1.11028665270807e-09
        K_a_ac = 1.73780082874937e-05
        K_a_bu = 1.51356124843621e-05
        K_a_co2 = 4.93707339753436e-07
        K_a_pro = 1.31825673855641e-05
        K_a_va = 1.38038426460288e-05
        K_ac = 0.14
        K_bu = 0.11
        K_pro = 0.07
        K_va = 0.10
        K_aa = 0.2
        K_h2 = 8.8e-07
        K_su = 0.47
        K_fa = 0.14
        K_w = 2.07877105595436e-14
        k_AB_IN = 1e10
        k_AB_ac = 1e10
        k_AB_bu = 1e10
        k_AB_co2 = 1e10
        k_AB_pro = 1e10
        k_AB_va = 1e10
        k_La = 200
        k_ch = 0.2671
        k_dec = 0.02
        k_li = 0.1838
        k_m_ac = 0.4
        k_m_bu = 1.2
        k_m_pro = 0.52
        k_m_va = 1.2
        k_m_aa = 4
        k_m_fa = 0.36
        k_m_h2 = 2.1
        k_m_su = 3
        k_p = 50000
        k_pr = 0.264
        pK_l_aa = 4
        pK_u_aa = 5.5
        pK_l_ac = 6
        pK_u_ac = 7
        pK_l_h2 = 5
        pK_u_h2 = 6
        rtol = 1e-6
        atol = 1e-8

        # Initial state
        state_zero = [
            0.007981285, 0.002492329, 0.024244036, 0.009391295, 0.009903645, 0.008043775, 0.041508584, 
            3.90797E-07, 0.020440103, 8.09018482, 1.051553699, 712.4058052, 19.83759232, 2.947358352, 
            2.437441052, 8.674399317, 1.060185924, 0.584681747, 0.178793053, 0.826209004, 0.9451213, 
            3.43436471, 1.864492601, 0.128208453, 0.019407639, 0.00935897, 0.009872404, 0.008014261, 
            0.04139487, 7.399735346, 0.025135342, 5.5933E-07, 0.420001299, 1.001769258
        ]
        columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", 
                   "S_IN", "S_h2o", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_va", "X_bu", 
                   "X_pro", "X_ac", "X_h2", "S_cation", "S_anion", "S_va_ion", "S_bu_ion", "S_pro_ion", 
                   "S_ac_ion", "S_hco3_ion", "S_nh3", "S_gas_h2", "S_gas_ch4", "S_gas_co2"]

        # ODE function
        def ADM1_ODE(t, y, state_input):
            S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_va, X_bu, X_pro, X_ac, X_h2, S_cation, S_anion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_gas_h2, S_gas_ch4, S_gas_co2 = y
            S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_ch4_in, S_IC_in, S_IN_in, S_h2o_in, S_h2_in, X_ch_in, X_pr_in, X_li_in, X_su_in, X_aa_in, X_fa_in, X_h2_in, X_va_in, X_bu_in, X_pro_in, X_ac_in, S_cation_in, S_anion_in, T_ad, q_ad, V_liq = state_input

            S_nh4_i = S_IN - S_nh3
            S_co2 = S_IC - S_hco3_ion
            phi = S_cation + S_nh4_i/17 - S_hco3_ion/44 - S_ac_ion/60 - S_pro_ion/74 - S_bu_ion/88 - S_va_ion/102 - S_anion
            S_H_ion = -phi * 0.5 + 0.5 * np.sqrt(phi **2 + 4 * K_w)
            pH = -np.log10(np.maximum(S_H_ion, 1e-14))
            p_ch4 = S_gas_ch4 * R * T_ad / 16
            p_co2 = S_gas_co2 * R * T_ad / 44
            p_h2 = S_gas_h2 * R * T_ad / 2
            p_h2o = 0.0657
            p_gas = p_ch4 + p_co2 + p_h2o
            q_gas = k_p * (p_gas - p_atm) * p_gas / p_atm

            I_0 = S_IN / (S_IN + K_I_IN)
            I_1 = K_I_fa / (K_I_fa + S_h2)
            I_2 = K_I_c4 / (K_I_c4 + S_h2)
            I_3 = K_I_pro / (K_I_pro + S_h2)
            I_4 = 10**(-(3/(pK_u_aa - pK_l_aa))*(pK_l_aa+pK_u_aa)/2) / (S_H_ion**(3/(pK_u_aa - pK_l_aa)) + 10**(-3/(pK_u_aa - pK_l_aa)*(pK_l_aa+pK_u_aa)/2))
            I_5 = 10**(-3/(pK_u_ac - pK_l_ac)*(pK_l_ac+pK_u_ac)/2) / (S_H_ion**(3/(pK_u_ac - pK_l_ac)) + 10**(-3/(pK_u_ac - pK_l_ac)*(pK_l_ac+pK_u_ac)/2))
            I_6 = 10**(-3/(pK_u_h2 - pK_l_h2)*(pK_l_h2+pK_u_h2)/2) / (S_H_ion**(3/(pK_u_h2 - pK_l_h2)) + 10**(-3/(pK_u_h2 - pK_l_h2)*(pK_l_h2+pK_u_h2)/2))
            I_7 = K_I_nh3 / (K_I_nh3 + S_nh3)

            rate_1 = k_ch * X_ch
            rate_2 = k_pr * X_pr
            rate_3 = k_li * X_li
            rate_4 = k_m_su * S_su / (K_su + S_su) * X_su * I_0 * I_4
            rate_5 = k_m_aa * S_aa / (K_aa + S_aa) * X_aa * I_0 * I_4
            rate_6 = k_m_fa * S_fa / (K_fa + S_fa) * X_fa * I_0 * I_1 * I_4
            rate_7 = k_m_va * S_va / (K_va + S_va) * X_va * S_va / (S_bu + S_va + 1e-8) * I_0 * I_2 * I_4
            rate_8 = k_m_bu * S_bu / (K_bu + S_bu) * X_bu * S_bu / (S_va + S_bu + 1e-8) * I_0 * I_2 * I_4
            rate_9 = k_m_pro * S_pro / (K_pro + S_pro) * X_pro * I_0 * I_3 * I_4
            rate_10 = k_m_ac * S_ac / (K_ac + S_ac) * X_ac * I_0 * I_5 * I_7
            rate_11 = k_m_h2 * S_h2 / (K_h2 + S_h2) * X_h2 * I_0 * I_6
            rate_12 = k_dec * X_su
            rate_13 = k_dec * X_aa
            rate_14 = k_dec * X_fa
            rate_15 = k_dec * X_va
            rate_16 = k_dec * X_bu
            rate_17 = k_dec * X_pro
            rate_18 = k_dec * X_ac
            rate_19 = k_dec * X_h2
            rate_20 = k_AB_va * (S_va_ion * (K_a_va + S_H_ion) - K_a_va * S_va)
            rate_21 = k_AB_bu * (S_bu_ion * (K_a_bu + S_H_ion) - K_a_bu * S_bu)
            rate_22 = k_AB_pro * (S_pro_ion * (K_a_pro + S_H_ion) - K_a_pro * S_pro)
            rate_23 = k_AB_ac * (S_ac_ion * (K_a_ac + S_H_ion) - K_a_ac * S_ac)
            rate_24 = k_AB_co2 * (S_hco3_ion * (K_a_co2 + S_H_ion) - K_a_co2 * S_IC)
            rate_25 = k_AB_IN * (S_nh3 * (K_a_IN + S_H_ion) - K_a_IN * S_IN)
            rate_26 = k_La * (S_h2 - 2 * (K_H_h2 * p_h2))
            rate_27 = k_La * (S_ch4 - 16 * (K_H_ch4 * p_ch4))
            rate_28 = k_La * (S_co2 - 44 * (K_H_co2 * p_co2))

            process_1 = 1.1111 * rate_1 + 0.13482 * rate_3 - 13.2724 * rate_4
            process_2 = rate_2 - 11.5665 * rate_5
            process_3 = 0.95115 * rate_3 - 8.2136 * rate_6
            process_4 = 1.8371 * rate_5 - 11.5757 * rate_7
            process_5 = 0.91131 * rate_4 + 2.3289 * rate_5 - 12.9817 * rate_8
            process_6 = 2.2734 * rate_4 + 0.53795 * rate_5 + 7.9149 * rate_7 - 23.3892 * rate_9
            process_7 = 4.8975 * rate_4 + 6.1053 * rate_5 + 14.5554 * rate_6 + 6.4459 * rate_7 + 16.6347 * rate_8 + 18.1566 * rate_9 - 26.5447 * rate_10
            process_8 = 0.30475 * rate_4 + 0.12297 * rate_5 + 0.83761 * rate_6 + 0.41881 * rate_7 + 0.55841 * rate_8 + 1.8392 * rate_9 - 2.9703 * rate_11 - rate_26
            process_9 = 6.7367 * rate_10 + 5.5548 * rate_11 - rate_27
            process_10 = -0.02933 * rate_3 + 4.4571 * rate_4 + 2.8335 * rate_5 -0.72457 * rate_6 -0.55945 * rate_7 -0.38907 * rate_8 + 13.1283 * rate_9 + 18.4808 * rate_10 -17.1839 * rate_11 - rate_28
            process_11 = -0.15056 * rate_4 + 2.1033 * rate_5 -0.15056 * rate_6 -0.15056 * rate_7 -0.15056 * rate_8 -0.15056 * rate_9 -0.15056 * rate_10 -0.15056 * rate_11
            process_12 = -0.1111 * rate_1 -0.05664 * rate_3 -0.4211 * rate_4 -5.3025 * rate_5 -7.3043 * rate_6 -3.4939 * rate_7 -4.6718 * rate_8 -10.5843 * rate_9 + 0.47776 * rate_10 + 13.75 * rate_11
            process_13 = -rate_1 + 0.18 * rate_12 + 0.18 * rate_13 + 0.18 * rate_14 + 0.18 * rate_15 + 0.18 * rate_16 + 0.18 * rate_17 + 0.18 * rate_18 + 0.18 * rate_19
            process_14 = -rate_2 + 0.77 * rate_12 + 0.77 * rate_13 + 0.77 * rate_14 + 0.77 * rate_15 + 0.77 * rate_16 + 0.77 * rate_17 + 0.77 * rate_18 + 0.77 * rate_19
            process_15 = -rate_3 + 0.05 * rate_12 + 0.05 * rate_13 + 0.05 * rate_14 + 0.05 * rate_15 + 0.05 * rate_16 + 0.05 * rate_17 + 0.05 * rate_18 + 0.05 * rate_19
            process_16 = rate_4 - rate_12
            process_17 = rate_5 - rate_13
            process_18 = rate_6 - rate_14
            process_19 = rate_7 - rate_15
            process_20 = rate_8 - rate_16
            process_21 = rate_9 - rate_17
            process_22 = rate_10 - rate_18
            process_23 = rate_11 - rate_19
            process_24 = process_25 = 0
            process_26 = -rate_20
            process_27 = -rate_21
            process_28 = -rate_22
            process_29 = -rate_23
            process_30 = -rate_24
            process_31 = -rate_25
            process_32 = (V_liq / V_liq*0.3) * rate_26
            process_33 = (V_liq / V_liq*0.3) * rate_27
            process_34 = (V_liq / V_liq*0.3) * rate_28

            dx = [
                q_ad * (S_su_in - S_su) / V_liq + process_1,
                q_ad * (S_aa_in - S_aa) / V_liq + process_2,
                q_ad * (S_fa_in - S_fa) / V_liq + process_3,
                q_ad * (S_va_in - S_va) / V_liq + process_4,
                q_ad * (S_bu_in - S_bu) / V_liq + process_5,
                q_ad * (S_pro_in - S_pro) / V_liq + process_6,
                q_ad * (S_ac_in - S_ac) / V_liq + process_7,
                q_ad * (S_h2_in - S_h2) / V_liq + process_8,
                q_ad * (S_ch4_in - S_ch4) / V_liq + process_9,
                q_ad * (S_IC_in - S_IC) / V_liq + process_10,
                q_ad * (S_IN_in - S_IN) / V_liq + process_11,
                q_ad * (S_h2o_in - S_h2o) / V_liq + process_12,
                q_ad * (X_ch_in - X_ch) / V_liq + process_13,
                q_ad * (X_pr_in - X_pr) / V_liq + process_14,
                q_ad * (X_li_in - X_li) / V_liq + process_15,
                q_ad * (X_su_in - X_su) / V_liq + process_16,
                q_ad * (X_aa_in - X_aa) / V_liq + process_17,
                q_ad * (X_fa_in - X_fa) / V_liq + process_18,
                q_ad * (X_va_in - X_va) / V_liq + process_19,
                q_ad * (X_bu_in - X_bu) / V_liq + process_20,
                q_ad * (X_pro_in - X_pro) / V_liq + process_21,
                q_ad * (X_ac_in - X_ac) / V_liq + process_22,
                q_ad * (X_h2_in - X_h2) / V_liq + process_23,
                q_ad * (S_cation_in - S_cation) / V_liq + process_24,
                q_ad * (S_anion_in - S_anion) / V_liq + process_25,
                process_26,
                process_27,
                process_28,
                process_29,
                process_30,
                process_31,
                -S_gas_h2 * q_gas / V_liq*0.3 + process_32,
                -S_gas_ch4 * q_gas / V_liq*0.3 + process_33,
                -S_gas_co2 * q_gas / V_liq*0.3 + process_34
            ]
            return dx

        # Simulation setup
        t = influent_df['time'].values
        state_list = [state_zero]
        gasflow_list = [{'time': 0, 'q_gas': 0, 'q_ch4': 0}]
        pressure_list = [{'time': 0, 'p_ch4': 0, 'p_co2':0, 'p_h2o': 0, 'p_gas': 0}]
        extras_list = [{'time': 0, 'pH': 7.52, 'OLR': 0,'FOS':0,'TAC':0, 'FOS/TAC': 0}]

        # Helper function for OLR
        def total_concentration(row):
            return (row['S_su'] + row['S_aa'] + row['S_fa'] + row['S_va'] + row['S_bu'] + row['S_pro'] + 
                    row['S_ac'] + row['S_ch4'] + row['X_ch'] + row['X_pr'] + row['X_li'] + row['X_su'] + 
                    row['X_aa'] + row['X_fa'] + row['X_h2'] + row['X_va'] + row['X_bu'] + row['X_pro'] + 
                    row['X_ac'])

        # Simulation loop
        for i in range(1, len(t)):
            current_influent = influent_df.iloc[i]
            state_input = [
                current_influent[col] for col in ['S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac', 
                                                 'S_ch4', 'S_IC', 'S_IN', 'S_h2o', 'S_h2', 'X_ch', 'X_pr', 
                                                 'X_li', 'X_su', 'X_aa', 'X_fa', 'X_h2', 'X_va', 'X_bu', 
                                                 'X_pro', 'X_ac', 'S_cation', 'S_anion', 'T', 'Q','V']
            ]
            t_step = [t[i-1], t[i]]
            result = scipy.integrate.solve_ivp(
                ADM1_ODE, t_step, state_list[-1], method='BDF', args=(state_input,), rtol=rtol, atol=atol
            )
            new_state = result.y[:, -1]
            state_list.append(new_state)

            S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_h2o, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_va, X_bu, X_pro, X_ac, X_h2, S_cation, S_anion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_gas_h2, S_gas_ch4, S_gas_co2 = new_state
            T_ad = state_input[-3]
            q_ad = state_input[-2]
            V_liq = state_input[-1]
            p_ch4 = S_gas_ch4 * R * T_ad / 16
            p_co2 = S_gas_co2 * R * T_ad / 44
            p_h2 = S_gas_h2 * R * T_ad / 2
            p_h2o = 0.0657
            p_gas = p_ch4 + p_co2 + p_h2o
            q_gas = k_p * (p_gas - p_atm) * p_gas / p_atm
            q_ch4 = q_gas * (p_ch4 / p_gas)

           
            S_nh4_i = S_IN - S_nh3
            S_co2 = S_IC - S_hco3_ion
            phi = S_cation + S_nh4_i/17 - S_hco3_ion/44 - S_ac_ion/60 - S_pro_ion/74 - S_bu_ion/88 - S_va_ion/102 - S_anion
            S_H_ion = -phi * 0.5 + 0.5 * np.sqrt(phi **2 + 4 * K_w)
            pH = -np.log10(np.maximum(S_H_ion, 1e-14))

            OLR = total_concentration(current_influent) * q_ad / V_liq
            HRT = V_liq / q_ad
            FOS = ((S_ac + S_ac_ion)/60 + (S_pro + S_pro_ion)/74 +  (S_bu + S_bu_ion)/88 + (S_va + S_va_ion)/102)*1000*1000
            TAC = (S_anion + S_hco3_ion + S_ac_ion/60 +  S_pro_ion/74 + S_bu_ion/88 + S_va_ion/102 - S_cation )*1000
            FOS_TAC = FOS / TAC if TAC != 0 else 0
            
            gasflow_list.append({'time': t[i], 'q_gas': q_gas, 'q_ch4': q_ch4})
            pressure_list.append({'time': t[i], 'p_ch4': p_ch4, 'p_co2': p_co2, 'p_h2o': p_h2o, 'p_gas': p_gas})
            extras_list.append({'time': t[i], 'pH': pH, 'OLR': OLR, 'HRT': HRT,'FOS':FOS,'TAC':TAC ,'FOS/TAC': FOS_TAC})

        # Convert lists to DataFrames
        self.simulate_results = pd.DataFrame(state_list, columns=columns)
        gasflow = pd.DataFrame(gasflow_list)
        pressure_results = pd.DataFrame(pressure_list)
        extras = pd.DataFrame(extras_list)
        self.output_data = pd.concat([
            influent_df['time'],
            gasflow[['q_gas', 'q_ch4']],
            pressure_results[['p_ch4', 'p_co2', 'p_h2o', 'p_gas']],
            extras[['pH', 'OLR', 'HRT','FOS','TAC','FOS/TAC']]
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
        # Only return process indicators (self.output_data)
        return self.output_data

    def save_output(self, file_path="Output/ADM1_Process_Indicators.xlsx"):
        """Save only process indicators to an Excel file."""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        self.output_data.to_excel(file_path, index=False)
        print(f"Process indicators saved to {file_path}")
        with pd.ExcelWriter(file_path) as writer:
            self.simulate_results.to_excel(writer, sheet_name='ADM1_States', index=False)
            self.output_data.to_excel(writer, sheet_name='Process_Data', index=False)
        print(f"Simulation results saved to {file_path}")

