import numpy as np
import scipy.integrate
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import time

class ADM1Simulator:
    def __init__(self, feedstock_ratios, days=100, Q=0.044, V=6.6, T=35):
        """
        Initialize the simulator with feedstock ratios and simulation parameters.
        feedstock_ratios: dict with keys as feedstock names and values as ratios (sum to 1)
        days: number of simulation days
        Q: influent flow rate (m^3/d)
        V: liquid reactor volume (m^3)
        T: temperature in C
        """
        self.feedstock_ratios = feedstock_ratios
        self.days = days
        self.Q = Q
        self.V = V
        self.T = T + 273.15  # Convert to Kelvin
        self.V_liq = V
        self.V_gas = 0.1 * V  # Assuming 10% gas volume
        self.simulate_results = None
        self.output_data = None
        self.gasflow = None
        self.vta = None

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

        # Constants from Rosen et al (2006) BSM2 report, adjusted for temperature
        self.R = 0.083145  # bar.M^-1.K^-1
        self.T_base = 298.15  # K
        self.T_op = self.T  # K
        self.k_p = 2  # m^3.d^-1.bar^-1
        self.p_atm = 1.013  # bar
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1 / self.T_base - 1 / self.T_op))  # bar

        # Stoichiometric parameters
        self.f_sI_xc = 0.1
        self.f_xI_xc = 0.2
        self.f_ch_xc = 0.2
        self.f_pr_xc = 0.2
        self.f_li_xc = 0.3
        self.N_xc = 0.0027
        self.N_I = 0.0014
        self.N_aa = 0.0114
        self.C_xc = 0.02786
        self.C_sI = 0.03
        self.C_ch = 0.0313
        self.C_pr = 0.03
        self.C_li = 0.022
        self.C_xI = 0.03
        self.C_su = 0.0313
        self.C_aa = 0.03
        self.f_fa_li = 0.95 
        self.C_fa = 0.0217
        self.f_h2_su = 0.19 
        self.f_bu_su = 0.13
        self.f_pro_su = 0.27
        self.f_ac_su = 0.41
        self.N_bac = 0.0245
        self.C_bu = 0.025
        self.C_pro = 0.0268
        self.C_ac = 0.0313
        self.C_bac = 0.0313
        self.Y_su = 0.1
        self.f_h2_aa = 0.06
        self.f_va_aa = 0.23
        self.f_bu_aa = 0.26
        self.f_pro_aa = 0.05
        self.f_ac_aa = 0.40
        self.C_va = 0.024
        self.Y_aa = 0.08
        self.Y_fa = 0.06
        self.Y_c4 = 0.06
        self.Y_pro = 0.04
        self.C_ch4 = 0.0156
        self.Y_ac = 0.05
        self.Y_h2 = 0.06

        # Biochemical parameters
        self.k_dis = 0.5  # d^-1
        self.k_hyd_ch = 10  # d^-1
        self.k_hyd_pr = 10  # d^-1
        self.k_hyd_li = 10  # d^-1
        self.K_S_IN = 10 ** -4  # M
        self.k_m_su = 30  # d^-1
        self.K_S_su = 0.5  # kgCOD.m^-3
        self.pH_UL_aa = 5.5
        self.pH_LL_aa = 4
        self.k_m_aa = 50  # d^-1
        self.K_S_aa = 0.3  # kgCOD.m^-3
        self.k_m_fa = 6  # d^-1
        self.K_S_fa = 0.4  # kgCOD.m^-3
        self.K_I_h2_fa = 5 * 10 ** -6  # kgCOD.m^-3
        self.k_m_c4 = 20  # d^-1
        self.K_S_c4 = 0.2  # kgCOD.m^-3
        self.K_I_h2_c4 = 10 ** -5  # kgCOD.m^-3
        self.k_m_pro = 13  # d^-1
        self.K_S_pro = 0.1  # kgCOD.m^-3
        self.K_I_h2_pro = 3.5 * 10 ** -6  # kgCOD.m^-3
        self.k_m_ac = 8  # d^-1
        self.K_S_ac = 0.15  # kgCOD.m^-3
        self.K_I_nh3 = 0.0018  # M
        self.pH_UL_ac = 7
        self.pH_LL_ac = 6
        self.k_m_h2 = 35  # d^-1
        self.K_S_h2 = 7 * 10 ** -6  # kgCOD.m^-3
        self.pH_UL_h2 = 6
        self.pH_LL_h2 = 5
        self.k_dec_X_su = 0.02  # d^-1
        self.k_dec_X_aa = 0.02  # d^-1
        self.k_dec_X_fa = 0.02  # d^-1
        self.k_dec_X_c4 = 0.02  # d^-1
        self.k_dec_X_pro = 0.02  # d^-1
        self.k_dec_X_ac = 0.02  # d^-1
        self.k_dec_X_h2 = 0.02  # d^-1

        # Physico-chemical parameters
        self.K_w = (10 ** -14.0) * np.exp((55900 / (100 * self.R)) * (1 / self.T_base - 1 / self.T_op))  # M
        self.K_a_va = 10 ** -4.86  # M
        self.K_a_bu = 10 ** -4.82  # M
        self.K_a_pro = 10 ** -4.88  # M
        self.K_a_ac = 10 ** -4.76  # M
        self.K_a_co2 = (10 ** -6.35) * np.exp((7646 / (100 * self.R)) * (1 / self.T_base - 1 / self.T_op))  # M
        self.K_a_IN = (10 ** -9.25) * np.exp((51965 / (100 * self.R)) * (1 / self.T_base - 1 / self.T_op))  # M
        self.k_A_B_va = 10 ** 10  # M^-1 * d^-1
        self.k_A_B_bu = 10 ** 10  # M^-1 * d^-1
        self.k_A_B_pro = 10 ** 10  # M^-1 * d^-1
        self.k_A_B_ac = 10 ** 10  # M^-1 * d^-1
        self.k_A_B_co2 = 10 ** 10  # M^-1 * d^-1
        self.k_A_B_IN = 10 ** 10  # M^-1 * d^-1
        self.k_L_a = 200.0  # d^-1
        self.K_H_co2 = 0.035 * np.exp((-19410 / (100 * self.R)) * (1 / self.T_base - 1 / self.T_op))  # Mliq.bar^-1
        self.K_H_ch4 = 0.0014 * np.exp((-14240 / (100 * self.R)) * (1 / self.T_base - 1 / self.T_op))  # Mliq.bar^-1
        self.K_H_h2 = (7.8 * 10 ** -4) * np.exp(-4180 / (100 * self.R) * (1 / self.T_base - 1 / self.T_op))  # Mliq.bar^-1

        # pH inhibition parameters
        self.K_pH_aa = (10 ** (-1 * (self.pH_LL_aa + self.pH_UL_aa) / 2.0))
        self.nn_aa = (3.0 / (self.pH_UL_aa - self.pH_LL_aa))
        self.K_pH_ac = (10 ** (-1 * (self.pH_LL_ac + self.pH_UL_ac) / 2.0))
        self.n_ac = (3.0 / (self.pH_UL_ac - self.pH_LL_ac))
        self.K_pH_h2 = (10 ** (-1 * (self.pH_LL_h2 + self.pH_UL_h2) / 2.0))
        self.n_h2 = (3.0 / (self.pH_UL_h2 - self.pH_LL_h2))

        # Default initial state (approximate values based on typical BSM2 conditions)
        self.initial_state = [
            0.5, 0.5, 0.1, 0.2, 0.2, 0.3, 1.0, 5e-5, 0.5, 0.25, 0.03, 0.5,  # Soluble
            200.0, 0.0, 0.0, 0.0, 30.0, 20.0, 10.0, 15.0, 10.0, 40.0, 10.0, 50.0,  # Particulate
            0.05, 0.05,  # Cation, Anion
            10 ** -7.4655377, 0.0, 0.0, 0.0, 0.0,  # S_H_ion, va_ion, bu_ion, pro_ion, ac_ion
            0.0, 0.14, 0.0, 0.0041,  # hco3_ion, co2, nh3, nh4_ion
            0.0, 0.0, 0.0  # Gas
        ]

    def run_codigestion(self):
        """Generate influent DataFrame from feedstock ratios and composition."""
        influent_columns = [
            'S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac', 'S_h2', 'S_ch4',
            'S_IC', 'S_IN', 'S_I',
            'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa', 'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
            'S_cation', 'S_anion'
        ]
        codigested_values = {col: [] for col in influent_columns}
        for day in range(self.days):
            x_xc_val = 0.0
            s_va_val = 0.0
            s_bu_val = 0.0
            s_pro_val = 0.0
            s_ac_val = 0.0
            s_in_val = 0.0
            s_cation_val = 0.0
            s_anion_val = 0.0
            for feed in self.feedstocks:
                ratio = self.feedstock_ratios.get(feed, 0.0)
                data = self.feedstock_data[feed]
                x_xc_val += (data[0] + data[1] + data[2]) * ratio  # Sum organics for X_xc
                s_ac_val += data[3] * ratio
                s_pro_val += data[4] * ratio
                s_bu_val += data[5] * ratio
                s_va_val += data[6] * ratio
                s_in_val += data[7] * ratio
                s_cation_val += data[9] * ratio
                s_anion_val += data[10] * ratio
            codigested_values['X_xc'].append(x_xc_val)
            codigested_values['S_va'].append(s_va_val)
            codigested_values['S_bu'].append(s_bu_val)
            codigested_values['S_pro'].append(s_pro_val)
            codigested_values['S_ac'].append(s_ac_val)
            codigested_values['S_IN'].append(s_in_val)
            codigested_values['S_cation'].append(s_cation_val)
            codigested_values['S_anion'].append(s_anion_val)
            for col in influent_columns:
                if col not in ['X_xc', 'S_va', 'S_bu', 'S_pro', 'S_ac', 'S_IN', 'S_cation', 'S_anion']:
                    codigested_values[col].append(0.0)
        influent_df = pd.DataFrame(codigested_values)
        influent_df['time'] = list(range(self.days))
        return influent_df

    def ADM1_ODE(self, t, state_zero, state_input):
        # Unpack state
        S_su = state_zero[0]
        S_aa = state_zero[1]
        S_fa = state_zero[2]
        S_va = state_zero[3]
        S_bu = state_zero[4]
        S_pro = state_zero[5]
        S_ac = state_zero[6]
        S_h2 = state_zero[7]
        S_ch4 = state_zero[8]
        S_IC = state_zero[9]
        S_IN = state_zero[10]
        S_I = state_zero[11]
        X_xc = state_zero[12]
        X_ch = state_zero[13]
        X_pr = state_zero[14]
        X_li = state_zero[15]
        X_su = state_zero[16]
        X_aa = state_zero[17]
        X_fa = state_zero[18]
        X_c4 = state_zero[19]
        X_pro = state_zero[20]
        X_ac = state_zero[21]
        X_h2 = state_zero[22]
        X_I = state_zero[23]
        S_cation = state_zero[24]
        S_anion = state_zero[25]
        S_H_ion = state_zero[26]
        S_va_ion = state_zero[27]
        S_bu_ion = state_zero[28]
        S_pro_ion = state_zero[29]
        S_ac_ion = state_zero[30]
        S_hco3_ion = state_zero[31]
        S_co2 = state_zero[32]
        S_nh3 = state_zero[33]
        S_nh4_ion = state_zero[34]
        S_gas_h2 = state_zero[35]
        S_gas_ch4 = state_zero[36]
        S_gas_co2 = state_zero[37]

        # Unpack input
        S_su_in = state_input[0]
        S_aa_in = state_input[1]
        S_fa_in = state_input[2]
        S_va_in = state_input[3]
        S_bu_in = state_input[4]
        S_pro_in = state_input[5]
        S_ac_in = state_input[6]
        S_h2_in = state_input[7]
        S_ch4_in = state_input[8]
        S_IC_in = state_input[9]
        S_IN_in = state_input[10]
        S_I_in = state_input[11]
        X_xc_in = state_input[12]
        X_ch_in = state_input[13]
        X_pr_in = state_input[14]
        X_li_in = state_input[15]
        X_su_in = state_input[16]
        X_aa_in = state_input[17]
        X_fa_in = state_input[18]
        X_c4_in = state_input[19]
        X_pro_in = state_input[20]
        X_ac_in = state_input[21]
        X_h2_in = state_input[22]
        X_I_in = state_input[23]
        S_cation_in = state_input[24]
        S_anion_in = state_input[25]
        q_ad_in = state_input[26]

        S_nh4_ion = (S_IN - S_nh3)
        S_co2 = (S_IC - S_hco3_ion)

        # Gas phase
        p_gas_h2 = (S_gas_h2 * self.R * self.T_op / 16)
        p_gas_ch4 = (S_gas_ch4 * self.R * self.T_op / 64)
        p_gas_co2 = (S_gas_co2 * self.R * self.T_op)

        # Inhibition
        I_pH_aa = ((self.K_pH_aa ** self.nn_aa) / (S_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa))
        I_pH_ac = ((self.K_pH_ac ** self.n_ac) / (S_H_ion ** self.n_ac + self.K_pH_ac ** self.n_ac))
        I_pH_h2 = ((self.K_pH_h2 ** self.n_h2) / (S_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2))
        I_IN_lim = (1 / (1 + (self.K_S_IN / S_IN)))
        I_h2_fa = (1 / (1 + (S_h2 / self.K_I_h2_fa)))
        I_h2_c4 = (1 / (1 + (S_h2 / self.K_I_h2_c4)))
        I_h2_pro = (1 / (1 + (S_h2 / self.K_I_h2_pro)))
        I_nh3 = (1 / (1 + (S_nh3 / self.K_I_nh3)))

        I_5 = (I_pH_aa * I_IN_lim)
        I_6 = I_5
        I_7 = (I_pH_aa * I_IN_lim * I_h2_fa)
        I_8 = (I_pH_aa * I_IN_lim * I_h2_c4)
        I_9 = I_8
        I_10 = (I_pH_aa * I_IN_lim * I_h2_pro)
        I_11 = (I_pH_ac * I_IN_lim * I_nh3)
        I_12 = (I_pH_h2 * I_IN_lim)

        # Biochemical process rates
        Rho_1 = (self.k_dis * X_xc)
        Rho_2 = (self.k_hyd_ch * X_ch)
        Rho_3 = (self.k_hyd_pr * X_pr)
        Rho_4 = (self.k_hyd_li * X_li)
        Rho_5 = self.k_m_su * S_su / (self.K_S_su + S_su) * X_su * I_5
        Rho_6 = (self.k_m_aa * (S_aa / (self.K_S_aa + S_aa)) * X_aa * I_6)
        Rho_7 = (self.k_m_fa * (S_fa / (self.K_S_fa + S_fa)) * X_fa * I_7)
        Rho_8 = (self.k_m_c4 * (S_va / (self.K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8)
        Rho_9 = (self.k_m_c4 * (S_bu / (self.K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9)
        Rho_10 = (self.k_m_pro * (S_pro / (self.K_S_pro + S_pro)) * X_pro * I_10)
        Rho_11 = (self.k_m_ac * (S_ac / (self.K_S_ac + S_ac)) * X_ac * I_11)
        Rho_12 = (self.k_m_h2 * (S_h2 / (self.K_S_h2 + S_h2)) * X_h2 * I_12)
        Rho_13 = (self.k_dec_X_su * X_su)
        Rho_14 = (self.k_dec_X_aa * X_aa)
        Rho_15 = (self.k_dec_X_fa * X_fa)
        Rho_16 = (self.k_dec_X_c4 * X_c4)
        Rho_17 = (self.k_dec_X_pro * X_pro)
        Rho_18 = (self.k_dec_X_ac * X_ac)
        Rho_19 = (self.k_dec_X_h2 * X_h2)

        # Acid-base rates
        Rho_A_4 = (self.k_A_B_va * (S_va_ion * (self.K_a_va + S_H_ion) - self.K_a_va * S_va))
        Rho_A_5 = (self.k_A_B_bu * (S_bu_ion * (self.K_a_bu + S_H_ion) - self.K_a_bu * S_bu))
        Rho_A_6 = (self.k_A_B_pro * (S_pro_ion * (self.K_a_pro + S_H_ion) - self.K_a_pro * S_pro))
        Rho_A_7 = (self.k_A_B_ac * (S_ac_ion * (self.K_a_ac + S_H_ion) - self.K_a_ac * S_ac))
        Rho_A_10 = (self.k_A_B_co2 * (S_hco3_ion * (self.K_a_co2 + S_H_ion) - self.K_a_co2 * S_IC))
        Rho_A_11 = (self.k_A_B_IN * (S_nh3 * (self.K_a_IN + S_H_ion) - self.K_a_IN * S_IN))

        # Gas transfer rates
        Rho_T_8 = (self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2))
        Rho_T_9 = (self.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4))
        Rho_T_10 = (self.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2))

        # Stoichiometric constants for carbon balance
        s_1 = (-1 * self.C_xc + self.f_sI_xc * self.C_sI + self.f_ch_xc * self.C_ch + self.f_pr_xc * self.C_pr + self.f_li_xc * self.C_li + self.f_xI_xc * self.C_xI)
        s_2 = (-1 * self.C_ch + self.C_su)
        s_3 = (-1 * self.C_pr + self.C_aa)
        s_4 = (-1 * self.C_li + (1 - self.f_fa_li) * self.C_su + self.f_fa_li * self.C_fa)
        s_5 = (-1 * self.C_su + (1 - self.Y_su) * (self.f_bu_su * self.C_bu + self.f_pro_su * self.C_pro + self.f_ac_su * self.C_ac) + self.Y_su * self.C_bac)
        s_6 = (-1 * self.C_aa + (1 - self.Y_aa) * (self.f_va_aa * self.C_va + self.f_bu_aa * self.C_bu + self.f_pro_aa * self.C_pro + self.f_ac_aa * self.C_ac) + self.Y_aa * self.C_bac)
        s_7 = (-1 * self.C_fa + (1 - self.Y_fa) * 0.7 * self.C_ac + self.Y_fa * self.C_bac)
        s_8 = (-1 * self.C_va + (1 - self.Y_c4) * 0.54 * self.C_pro + (1 - self.Y_c4) * 0.31 * self.C_ac + self.Y_c4 * self.C_bac)
        s_9 = (-1 * self.C_bu + (1 - self.Y_c4) * 0.8 * self.C_ac + self.Y_c4 * self.C_bac)
        s_10 = (-1 * self.C_pro + (1 - self.Y_pro) * 0.57 * self.C_ac + self.Y_pro * self.C_bac)
        s_11 = (-1 * self.C_ac + (1 - self.Y_ac) * self.C_ch4 + self.Y_ac * self.C_bac)
        s_12 = ((1 - self.Y_h2) * self.C_ch4 + self.Y_h2 * self.C_bac)
        s_13 = (-1 * self.C_bac + self.C_xc)

        p_gas = (p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o)
        q_gas = (self.k_p * (p_gas - self.p_atm))
        if q_gas < 0:
            q_gas = 0
        q_ch4 = q_gas * (p_gas_ch4 / p_gas)

        # Differential equations
        diff_S_su = q_ad_in / self.V_liq * (S_su_in - S_su) + Rho_2 + (1 - self.f_fa_li) * Rho_4 - Rho_5
        diff_S_aa = q_ad_in / self.V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6
        diff_S_fa = q_ad_in / self.V_liq * (S_fa_in - S_fa) + (self.f_fa_li * Rho_4) - Rho_7
        diff_S_va = q_ad_in / self.V_liq * (S_va_in - S_va) + (1 - self.Y_aa) * self.f_va_aa * Rho_6 - Rho_8
        diff_S_bu = q_ad_in / self.V_liq * (S_bu_in - S_bu) + (1 - self.Y_su) * self.f_bu_su * Rho_5 + (1 - self.Y_aa) * self.f_bu_aa * Rho_6 - Rho_9
        diff_S_pro = q_ad_in / self.V_liq * (S_pro_in - S_pro) + (1 - self.Y_su) * self.f_pro_su * Rho_5 + (1 - self.Y_aa) * self.f_pro_aa * Rho_6 + (1 - self.Y_c4) * 0.54 * Rho_8 - Rho_10
        diff_S_ac = q_ad_in / self.V_liq * (S_ac_in - S_ac) + (1 - self.Y_su) * self.f_ac_su * Rho_5 + (1 - self.Y_aa) * self.f_ac_aa * Rho_6 + (1 - self.Y_fa) * 0.7 * Rho_7 + (1 - self.Y_c4) * 0.31 * Rho_8 + (1 - self.Y_c4) * 0.8 * Rho_9 + (1 - self.Y_pro) * 0.57 * Rho_10 - Rho_11
        diff_S_h2 = 0
        diff_S_ch4 = q_ad_in / self.V_liq * (S_ch4_in - S_ch4) + (1 - self.Y_ac) * Rho_11 + (1 - self.Y_h2) * Rho_12 - Rho_T_9
        Sigma = (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))
        diff_S_IC = q_ad_in / self.V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
        diff_S_IN = q_ad_in / self.V_liq * (S_IN_in - S_IN) + (self.N_xc - self.f_xI_xc * self.N_I - self.f_sI_xc * self.N_I - self.f_pr_xc * self.N_aa) * Rho_1 - self.Y_su * self.N_bac * Rho_5 + (self.N_aa - self.Y_aa * self.N_bac) * Rho_6 - self.Y_fa * self.N_bac * Rho_7 - self.Y_c4 * self.N_bac * Rho_8 - self.Y_c4 * self.N_bac * Rho_9 - self.Y_pro * self.N_bac * Rho_10 - self.Y_ac * self.N_bac * Rho_11 - self.Y_h2 * self.N_bac * Rho_12 + (self.N_bac - self.N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        diff_S_I = q_ad_in / self.V_liq * (S_I_in - S_I) + self.f_sI_xc * Rho_1
        diff_X_xc = q_ad_in / self.V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19
        diff_X_ch = q_ad_in / self.V_liq * (X_ch_in - X_ch) + self.f_ch_xc * Rho_1 - Rho_2
        diff_X_pr = q_ad_in / self.V_liq * (X_pr_in - X_pr) + self.f_pr_xc * Rho_1 - Rho_3
        diff_X_li = q_ad_in / self.V_liq * (X_li_in - X_li) + self.f_li_xc * Rho_1 - Rho_4
        diff_X_su = q_ad_in / self.V_liq * (X_su_in - X_su) + self.Y_su * Rho_5 - Rho_13
        diff_X_aa = q_ad_in / self.V_liq * (X_aa_in - X_aa) + self.Y_aa * Rho_6 - Rho_14
        diff_X_fa = q_ad_in / self.V_liq * (X_fa_in - X_fa) + self.Y_fa * Rho_7 - Rho_15
        diff_X_c4 = q_ad_in / self.V_liq * (X_c4_in - X_c4) + self.Y_c4 * Rho_8 + self.Y_c4 * Rho_9 - Rho_16
        diff_X_pro = q_ad_in / self.V_liq * (X_pro_in - X_pro) + self.Y_pro * Rho_10 - Rho_17
        diff_X_ac = q_ad_in / self.V_liq * (X_ac_in - X_ac) + self.Y_ac * Rho_11 - Rho_18
        diff_X_h2 = q_ad_in / self.V_liq * (X_h2_in - X_h2) + self.Y_h2 * Rho_12 - Rho_19
        diff_X_I = q_ad_in / self.V_liq * (X_I_in - X_I) + self.f_xI_xc * Rho_1
        diff_S_cation = q_ad_in / self.V_liq * (S_cation_in - S_cation)
        diff_S_anion = q_ad_in / self.V_liq * (S_anion_in - S_anion)
        diff_S_H_ion = 0
        diff_S_va_ion = 0
        diff_S_bu_ion = 0
        diff_S_pro_ion = 0
        diff_S_ac_ion = 0
        diff_S_hco3_ion = 0
        diff_S_nh3 = 0
        diff_S_co2 = 0
        diff_S_nh4_ion = 0
        diff_S_gas_h2 = (q_gas / self.V_gas * -1 * S_gas_h2) + (Rho_T_8 * self.V_liq / self.V_gas)
        diff_S_gas_ch4 = (q_gas / self.V_gas * -1 * S_gas_ch4) + (Rho_T_9 * self.V_liq / self.V_gas)
        diff_S_gas_co2 = (q_gas / self.V_gas * -1 * S_gas_co2) + (Rho_T_10 * self.V_liq / self.V_gas)

        return [diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I, diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I, diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion, diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2, diff_S_nh3, diff_S_nh4_ion, diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2]

    def simulate(self, t_step, state_zero, state_input, solvermethod):
        def ode_wrapper(t, y):
            return self.ADM1_ODE(t, y, state_input)
        r = scipy.integrate.solve_ivp(ode_wrapper, t_step, state_zero, method=solvermethod)
        return r.y

    def DAESolve(self, state_zero):
        # Unpack
        S_su = state_zero[0]
        S_aa = state_zero[1]
        S_fa = state_zero[2]
        S_va = state_zero[3]
        S_bu = state_zero[4]
        S_pro = state_zero[5]
        S_ac = state_zero[6]
        S_h2 = state_zero[7]
        S_ch4 = state_zero[8]
        S_IC = state_zero[9]
        S_IN = state_zero[10]
        S_I = state_zero[11]
        X_xc = state_zero[12]
        X_ch = state_zero[13]
        X_pr = state_zero[14]
        X_li = state_zero[15]
        X_su = state_zero[16]
        X_aa = state_zero[17]
        X_fa = state_zero[18]
        X_c4 = state_zero[19]
        X_pro = state_zero[20]
        X_ac = state_zero[21]
        X_h2 = state_zero[22]
        X_I = state_zero[23]
        S_cation = state_zero[24]
        S_anion = state_zero[25]
        S_H_ion = state_zero[26]
        S_va_ion = state_zero[27]
        S_bu_ion = state_zero[28]
        S_pro_ion = state_zero[29]
        S_ac_ion = state_zero[30]
        S_hco3_ion = state_zero[31]
        S_co2 = state_zero[32]
        S_nh3 = state_zero[33]
        S_nh4_ion = state_zero[34]
        S_gas_h2 = state_zero[35]
        S_gas_ch4 = state_zero[36]
        S_gas_co2 = state_zero[37]

        eps = 0.0000001
        prevS_H_ion = S_H_ion

        # Newton-Raphson for S_H_ion
        shdelta = 1.0
        shgradeq = 1.0
        tol = 10 ** (-12)
        maxIter = 1000
        i = 1
        while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
            S_va_ion = self.K_a_va * S_va / (self.K_a_va + S_H_ion)
            S_bu_ion = self.K_a_bu * S_bu / (self.K_a_bu + S_H_ion)
            S_pro_ion = self.K_a_pro * S_pro / (self.K_a_pro + S_H_ion)
            S_ac_ion = self.K_a_ac * S_ac / (self.K_a_ac + S_H_ion)
            S_hco3_ion = self.K_a_co2 * S_IC / (self.K_a_co2 + S_H_ion)
            S_nh3 = self.K_a_IN * S_IN / (self.K_a_IN + S_H_ion)
            shdelta = S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 - S_va_ion / 208.0 - self.K_w / S_H_ion - S_anion
            shgradeq = 1 + self.K_a_IN * S_IN / ((self.K_a_IN + S_H_ion) * (self.K_a_IN + S_H_ion)) + self.K_a_co2 * S_IC / ((self.K_a_co2 + S_H_ion) * (self.K_a_co2 + S_H_ion)) + 1 / 64.0 * self.K_a_ac * S_ac / ((self.K_a_ac + S_H_ion) * (self.K_a_ac + S_H_ion)) + 1 / 112.0 * self.K_a_pro * S_pro / ((self.K_a_pro + S_H_ion) * (self.K_a_pro + S_H_ion)) + 1 / 160.0 * self.K_a_bu * S_bu / ((self.K_a_bu + S_H_ion) * (self.K_a_bu + S_H_ion)) + 1 / 208.0 * self.K_a_va * S_va / ((self.K_a_va + S_H_ion) * (self.K_a_va + S_H_ion)) + self.K_w / (S_H_ion * S_H_ion)
            S_H_ion = S_H_ion - shdelta / shgradeq
            if S_H_ion <= 0:
                S_H_ion = tol
            i += 1

        # Newton-Raphson for S_h2
        S_h2delta = 1.0
        S_h2gradeq = 1.0
        j = 1
        while ((S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter)):
            I_pH_aa = (self.K_pH_aa ** self.nn_aa) / (prevS_H_ion ** self.nn_aa + self.K_pH_aa ** self.nn_aa)
            I_pH_h2 = (self.K_pH_h2 ** self.n_h2) / (prevS_H_ion ** self.n_h2 + self.K_pH_h2 ** self.n_h2)
            I_IN_lim = 1 / (1 + (self.K_S_IN / S_IN))
            I_h2_fa = 1 / (1 + (S_h2 / self.K_I_h2_fa))
            I_h2_c4 = 1 / (1 + (S_h2 / self.K_I_h2_c4))
            I_h2_pro = 1 / (1 + (S_h2 / self.K_I_h2_pro))
            I_5 = I_pH_aa * I_IN_lim
            I_6 = I_5
            I_7 = I_pH_aa * I_IN_lim * I_h2_fa
            I_8 = I_pH_aa * I_IN_lim * I_h2_c4
            I_9 = I_8
            I_10 = I_pH_aa * I_IN_lim * I_h2_pro
            I_12 = I_pH_h2 * I_IN_lim
            Rho_5 = self.k_m_su * (S_su / (self.K_S_su + S_su)) * X_su * I_5
            Rho_6 = self.k_m_aa * (S_aa / (self.K_S_aa + S_aa)) * X_aa * I_6
            Rho_7 = self.k_m_fa * (S_fa / (self.K_S_fa + S_fa)) * X_fa * I_7
            Rho_8 = self.k_m_c4 * (S_va / (self.K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
            Rho_9 = self.k_m_c4 * (S_bu / (self.K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
            Rho_10 = self.k_m_pro * (S_pro / (self.K_S_pro + S_pro)) * X_pro * I_10
            Rho_12 = self.k_m_h2 * (S_h2 / (self.K_S_h2 + S_h2)) * X_h2 * I_12
            p_gas_h2 = S_gas_h2 * self.R * self.T_op / 16
            Rho_T_8 = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
            S_h2delta = self.Q / self.V_liq * (0 - S_h2) + (1 - self.Y_su) * self.f_h2_su * Rho_5 + (1 - self.Y_aa) * self.f_h2_aa * Rho_6 + (1 - self.Y_fa) * 0.3 * Rho_7 + (1 - self.Y_c4) * 0.15 * Rho_8 + (1 - self.Y_c4) * 0.2 * Rho_9 + (1 - self.Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8  # Note: S_h2_in assumed 0, adjust if needed
            S_h2gradeq = -1.0 / self.V_liq * self.Q - 3.0 / 10.0 * (1 - self.Y_fa) * self.k_m_fa * S_fa / (self.K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_fa) * (1 + S_h2 / self.K_I_h2_fa)) / self.K_I_h2_fa - 3.0 / 20.0 * (1 - self.Y_c4) * self.k_m_c4 * S_va * S_va / (self.K_S_c4 + S_va) * X_c4 / (S_bu + S_va + 1e-6) * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_c4) * (1 + S_h2 / self.K_I_h2_c4)) / self.K_I_h2_c4 - 1.0 / 5.0 * (1 - self.Y_c4) * self.k_m_c4 * S_bu * S_bu / (self.K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + 1e-6) * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_c4) * (1 + S_h2 / self.K_I_h2_c4)) / self.K_I_h2_c4 - 43.0 / 100.0 * (1 - self.Y_pro) * self.k_m_pro * S_pro / (self.K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + self.K_S_IN / S_IN) / ((1 + S_h2 / self.K_I_h2_pro) * (1 + S_h2 / self.K_I_h2_pro)) / self.K_I_h2_pro - self.k_m_h2 / (self.K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + self.K_S_IN / S_IN) + self.k_m_h2 * S_h2 / ((self.K_S_h2 + S_h2) * (self.K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + self.K_S_IN / S_IN) - self.k_L_a
            S_h2 = S_h2 - S_h2delta / S_h2gradeq
            if S_h2 <= 0:
                S_h2 = tol
            j += 1

        # Update state_zero in place
        state_zero[26] = S_H_ion
        state_zero[7] = S_h2
        state_zero[27] = S_va_ion
        state_zero[28] = S_bu_ion
        state_zero[29] = S_pro_ion
        state_zero[30] = S_ac_ion
        state_zero[31] = S_hco3_ion
        state_zero[33] = S_nh3

    def run_simulation(self, influent_df):
        """Run the ADM1 simulation using the influent DataFrame."""
        t = influent_df['time'].values
        state_zero = self.initial_state.copy()
        simulate_results = pd.DataFrame([state_zero])
        columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion", "S_H_ion", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"]
        simulate_results.columns = columns
        solvermethod = 'BDF'
        t0 = 0
        gasflow = pd.DataFrame({'time': [0], 'q_gas': [0], 'q_ch4': [0]})
        pressure = pd.DataFrame({'time': 0, 'p_gas_ch4': [0], 'p_gas_co2':[0], 'p_gas_h2': [0], 'p_gas': [0]})
        vta = pd.DataFrame({'time': [0], 'FOS': [0], 'TAC': [0], 'FOS/TAC': [0]})
        output_list = []
        influent_columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion"]
        for i in range(1, len(t)):
            row = influent_df.iloc[i]
            state_input = [row[col] for col in influent_columns] + [self.Q]
            tstep = [t0, t[i]]
            sim_y = self.simulate(tstep, state_zero, state_input, solvermethod)
            state_zero = sim_y[:, -1].tolist()
            # Unpack for DAESolve and calculations
            S_su = state_zero[0]
            S_aa = state_zero[1]
            S_fa = state_zero[2]
            S_va = state_zero[3]
            S_bu = state_zero[4]
            S_pro = state_zero[5]
            S_ac = state_zero[6]
            S_h2 = state_zero[7]
            S_ch4 = state_zero[8]
            S_IC = state_zero[9]
            S_IN = state_zero[10]
            S_I = state_zero[11]
            X_xc = state_zero[12]
            X_ch = state_zero[13]
            X_pr = state_zero[14]
            X_li = state_zero[15]
            X_su = state_zero[16]
            X_aa = state_zero[17]
            X_fa = state_zero[18]
            X_c4 = state_zero[19]
            X_pro = state_zero[20]
            X_ac = state_zero[21]
            X_h2 = state_zero[22]
            X_I = state_zero[23]
            S_cation = state_zero[24]
            S_anion = state_zero[25]
            S_H_ion = state_zero[26]
            S_va_ion = state_zero[27]
            S_bu_ion = state_zero[28]
            S_pro_ion = state_zero[29]
            S_ac_ion = state_zero[30]
            S_hco3_ion = state_zero[31]
            S_co2 = state_zero[32]
            S_nh3 = state_zero[33]
            S_nh4_ion = state_zero[34]
            S_gas_h2 = state_zero[35]
            S_gas_ch4 = state_zero[36]
            S_gas_co2 = state_zero[37]
            self.DAESolve(state_zero)
            S_nh4_ion = S_IN - S_nh3
            S_co2 = S_IC - S_hco3_ion
            state_zero[32] = S_co2
            state_zero[34] = S_nh4_ion
            pH = -np.log10(state_zero[26])
            p_gas_h2 = (S_gas_h2 * self.R * self.T_op / 16)
            p_gas_ch4 = (S_gas_ch4 * self.R * self.T_op / 64)
            p_gas_co2 = (S_gas_co2 * self.R * self.T_op)
            p_gas = (p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o)
            q_gas = (self.k_p * (p_gas - self.p_atm))
            if q_gas < 0:
                q_gas = 0
            q_ch4 = q_gas * (p_gas_ch4 / p_gas) if p_gas > 0 else 0
            if q_ch4 < 0:
                q_ch4 = 0
            FOS = S_ac + S_pro + S_bu + S_va
            TAC = S_anion + S_hco3_ion + S_ac + S_pro + S_bu + S_va + S_cation
            FOS_TAC = FOS / TAC if TAC != 0 else 0
            total_cod_in = sum([row[col] for col in influent_columns if col not in ['S_IC', 'S_IN', 'S_cation', 'S_anion']])
            OLR = total_cod_in * self.Q / self.V_liq
            # Append to results
            dfstate_zero = pd.DataFrame([state_zero], columns=columns)
            simulate_results = pd.concat([simulate_results, dfstate_zero], ignore_index=True)
            gasflow = pd.concat([gasflow, pd.DataFrame({'time': [t[i]], 'q_gas': [q_gas], 'q_ch4': [q_ch4]})], ignore_index=True)
            pressure = pd.concat([pressure, pd.DataFrame({'time': [t[i]],  'p_gas_ch4': [p_gas_ch4], 'p_gas_co2':[p_gas_co2], 'p_gas_h2': [p_gas_h2], 'p_gas': [p_gas])], ignore_index=True)
            vta = pd.concat([vta, pd.DataFrame({'time': [t[i]], 'FOS': [FOS], 'TAC': [TAC], 'FOS/TAC': [FOS_TAC]})], ignore_index=True)
            output_list.append({'time': t[i],'p_gas', 'q_gas': q_gas, 'q_ch4': q_ch4, 'pH': pH, 'OLR': OLR, 'FOS': FOS, 'TAC': TAC, 'FOS/TAC': FOS_TAC})
            t0 = t[i]
        self.simulate_results = simulate_results
        self.gasflow = gasflow
        self.pressure = pressure
        self.vta = vta
        self.output_data = pd.DataFrame(output_list)
        initial_pH = -np.log10(self.initial_state[26])
        initial_output = {'time': 0,'p_gas': 0, 'q_gas': 0, 'q_ch4': 0, 'pH': initial_pH, 'OLR': 0, 'FOS': 0, 'TAC': 0, 'FOS/TAC': 0}
        self.output_data = pd.concat([pd.DataFrame([initial_output]), self.output_data], ignore_index=True)
        self.simulate_results['pH'] = -np.log10(self.simulate_results['S_H_ion'])

    def run(self):
        """Execute the full simulation process."""
        print("Digester Simulation in Progress....")
        influent_df = self.run_codigestion()
        self.run_simulation(influent_df)

    def get_results(self):
        """Return only process indicators for plotting."""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        return self.output_data

    def save_output(self, file_path="dynamic_out.csv"):
        """Save simulation results to CSV files, similar to original script."""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        self.simulate_results.to_csv("dynamic_out.csv", index=False)
        self.vta.to_csv("vfa_ta_ratio.csv", index=False)
        self.gasflow.to_csv("dynamic_gas_flow_rates.csv", index=False)
        self.pressure.to_csv("dynamic_pressure_rates.csv", index=False)
        print(f"Results saved.")




