import numpy as np
import pandas as pd
import scipy.integrate

class ADM1Simulator:
    def __init__(self, feedstock_ratios, days=100, Q=0.044, V_liq=6.6, V_gas=0.66, T=35):
        """
        Initialize the simulator with feedstock ratios and simulation parameters.
        feedstock_ratios: dict with keys as feedstock names and values as ratios (sum to 1)
        days: number of simulation days
        Q: influent flow rate (m^3/day, default 0.044)
        V_liq: liquid reactor volume (m^3, default 6.6)
        V_gas: gas reactor volume (m^3, default 0.66)
        T: temperature in Celsius (default 35)
        """
        self.feedstock_ratios = feedstock_ratios
        self.days = days
        self.Q = Q
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.V_ad = V_liq + V_gas
        self.T_op = T + 273.15  # Convert to Kelvin
        self.simulate_results = None
        self.output_data = None

        # Feedstock composition data (kg/m3 or kmol/m3 depending on component)
        self.feedstock_data = {
            "Maize Silage": {
                'X_ch': 202.5, 'X_pr': 16.1, 'X_li': 11.8, 'S_ac': 2.6, 
                'S_pro': 0.364, 'S_bu': 0.56, 'S_va': 0.2, 'S_IN': 1.583, 
                'moisture_content': 718, 'S_cation': 0.15, 'S_anion': 0.02
            },
            "Grass Silage": {
                'X_ch': 177.8, 'X_pr': 10.8, 'X_li': 1.4, 'S_ac': 3.9,
                'S_pro': 0.361, 'S_bu': 0.893, 'S_va': 0.292, 'S_IN': 1.71,
                'moisture_content': 941, 'S_cation': 0.15, 'S_anion': 0.02
            },
            "Food Waste": {
                'X_ch': 177.8, 'X_pr': 5.6, 'X_li': 6.1, 'S_ac': 4.29,
                'S_pro': 1.056, 'S_bu': 0.924, 'S_va': 0.33, 'S_IN': 1.378,
                'moisture_content': 775, 'S_cation': 0.07, 'S_anion': 0.02
            },
            "Cattle Slurry": {
                'X_ch': 177.8, 'X_pr': 82.7, 'X_li': 35.4, 'S_ac': 0.45,
                'S_pro': 0.112, 'S_bu': 0.098, 'S_va': 0.035, 'S_IN': 1.378,
                'moisture_content': 156, 'S_cation': 0.07, 'S_anion': 0.02
            }
        }

        # Initialize ADM1 parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize ADM1 model parameters from BSM2 report"""
        # Physical constants
        self.R = 0.083145  # bar.M^-1.K^-1
        self.T_base = 298.15  # K
        self.k_p = 2  # m^3.d^-1.bar^-1
        self.p_atm = 1.013  # bar
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1/self.T_base - 1/self.T_op))

        # Stoichiometric parameters
        self.f_sI_xc = 0.1
        self.f_xI_xc = 0.2
        self.f_ch_xc = 0.2
        self.f_pr_xc = 0.2
        self.f_li_xc = 0.3
        self.N_xc = 0.0027
        self.N_I = 0.0014
        self.N_aa = 0.0114
        self.N_bac = 0.0245
        
        # Carbon content parameters
        self.C_xc = 0.02786
        self.C_sI = 0.03
        self.C_ch = 0.0313
        self.C_pr = 0.03
        self.C_li = 0.022
        self.C_xI = 0.03
        self.C_su = 0.0313
        self.C_aa = 0.03
        self.C_fa = 0.0217
        self.C_bu = 0.025
        self.C_pro = 0.0268
        self.C_ac = 0.0313
        self.C_bac = 0.0313
        self.C_va = 0.024
        self.C_ch4 = 0.0156

        # Yield parameters
        self.f_fa_li = 0.95
        self.f_h2_su = 0.19
        self.f_bu_su = 0.13
        self.f_pro_su = 0.27
        self.f_ac_su = 0.41
        self.f_h2_aa = 0.06
        self.f_va_aa = 0.23
        self.f_bu_aa = 0.26
        self.f_pro_aa = 0.05
        self.f_ac_aa = 0.40
        
        self.Y_su = 0.1
        self.Y_aa = 0.08
        self.Y_fa = 0.06
        self.Y_c4 = 0.06
        self.Y_pro = 0.04
        self.Y_ac = 0.05
        self.Y_h2 = 0.06

        # Biochemical parameters
        self.k_dis = 0.5
        self.k_hyd_ch = 10
        self.k_hyd_pr = 10
        self.k_hyd_li = 10
        self.K_S_IN = 10**-4
        self.k_m_su = 30
        self.K_S_su = 0.5
        self.k_m_aa = 50
        self.K_S_aa = 0.3
        self.k_m_fa = 6
        self.K_S_fa = 0.4
        self.K_I_h2_fa = 5e-6
        self.k_m_c4 = 20
        self.K_S_c4 = 0.2
        self.K_I_h2_c4 = 1e-5
        self.k_m_pro = 13
        self.K_S_pro = 0.1
        self.K_I_h2_pro = 3.5e-6
        self.k_m_ac = 8
        self.K_S_ac = 0.15
        self.K_I_nh3 = 0.0018
        self.k_m_h2 = 35
        self.K_S_h2 = 7e-6
        
        # Decay parameters
        self.k_dec_X_su = 0.02
        self.k_dec_X_aa = 0.02
        self.k_dec_X_fa = 0.02
        self.k_dec_X_c4 = 0.02
        self.k_dec_X_pro = 0.02
        self.k_dec_X_ac = 0.02
        self.k_dec_X_h2 = 0.02

        # pH inhibition parameters
        self.pH_UL_aa = 5.5
        self.pH_LL_aa = 4
        self.pH_UL_ac = 7
        self.pH_LL_ac = 6
        self.pH_UL_h2 = 6
        self.pH_LL_h2 = 5

        # Physico-chemical parameters
        self.K_w = (10**-14.0) * np.exp((55900/(100*self.R))*(1/self.T_base - 1/self.T_op))
        self.K_a_va = 10**-4.86
        self.K_a_bu = 10**-4.82
        self.K_a_pro = 10**-4.88
        self.K_a_ac = 10**-4.76
        self.K_a_co2 = (10**-6.35) * np.exp((7646/(100*self.R))*(1/self.T_base - 1/self.T_op))
        self.K_a_IN = (10**-9.25) * np.exp((51965/(100*self.R))*(1/self.T_base - 1/self.T_op))
        
        self.k_A_B_va = 1e10
        self.k_A_B_bu = 1e10
        self.k_A_B_pro = 1e10
        self.k_A_B_ac = 1e10
        self.k_A_B_co2 = 1e10
        self.k_A_B_IN = 1e10
        
        self.k_L_a = 200.0
        self.K_H_co2 = 0.035 * np.exp((-19410/(100*self.R))*(1/self.T_base - 1/self.T_op))
        self.K_H_ch4 = 0.0014 * np.exp((-14240/(100*self.R))*(1/self.T_base - 1/self.T_op))
        self.K_H_h2 = (7.8e-4) * np.exp(-4180/(100*self.R)*(1/self.T_base - 1/self.T_op))

        # pH constants
        self.K_pH_aa = 10**(-(self.pH_LL_aa + self.pH_UL_aa)/2.0)
        self.nn_aa = 3.0/(self.pH_UL_aa - self.pH_LL_aa)
        self.K_pH_ac = 10**(-(self.pH_LL_ac + self.pH_UL_ac)/2.0)
        self.n_ac = 3.0/(self.pH_UL_ac - self.pH_LL_ac)
        self.K_pH_h2 = 10**(-(self.pH_LL_h2 + self.pH_UL_h2)/2.0)
        self.n_h2 = 3.0/(self.pH_UL_h2 - self.pH_LL_h2)

    def run_codigestion(self):
        """Generate simple codigested influent composition from feedstock ratios"""
        influent_data = []
        
        for day in range(self.days):
            # Initialize daily composition with zeros
            daily_composition = {
                'time': day,
                'S_su': 0, 'S_aa': 0, 'S_fa': 0, 'S_va': 0, 'S_bu': 0,
                'S_pro': 0, 'S_ac': 0, 'S_h2': 0, 'S_ch4': 0, 'S_IC': 8.1,
                'S_IN': 0, 'S_I': 0,
                'X_xc': 0, 'X_ch': 0, 'X_pr': 0, 'X_li': 0, 'X_su': 0,
                'X_aa': 0, 'X_fa': 0, 'X_c4': 0, 'X_pro': 0, 'X_ac': 0,
                'X_h2': 0, 'X_I': 0,
                'S_cation': 0, 'S_anion': 0
            }
            
            # Simple weighted average calculation
            for feedstock, ratio in self.feedstock_ratios.items():
                if feedstock in self.feedstock_data:
                    feed_data = self.feedstock_data[feedstock]
                    
                    # Add weighted contributions
                    daily_composition['X_ch'] += feed_data['X_ch'] * ratio
                    daily_composition['X_pr'] += feed_data['X_pr'] * ratio
                    daily_composition['X_li'] += feed_data['X_li'] * ratio
                    daily_composition['S_ac'] += feed_data['S_ac'] * ratio
                    daily_composition['S_pro'] += feed_data['S_pro'] * ratio
                    daily_composition['S_bu'] += feed_data['S_bu'] * ratio
                    daily_composition['S_va'] += feed_data['S_va'] * ratio
                    daily_composition['S_IN'] += feed_data['S_IN'] * ratio
                    daily_composition['S_cation'] += feed_data['S_cation'] * ratio
                    daily_composition['S_anion'] += feed_data['S_anion'] * ratio
            
            influent_data.append(daily_composition)
        
        return pd.DataFrame(influent_data)

    def run_simulation(self, influent_df):
        """Run the comprehensive ADM1 simulation"""
        # Initial state (from BSM2 report)
        state_zero = [
            0.012, 0.005, 0.099, 0.011, 0.013, 0.016, 0.2, 
            2.5e-7, 0.055, 0.15, 0.13, 0.025,
            0.31, 0.028, 0.1, 0.042, 0.42, 0.24, 0.021, 0.04,
            0.14, 0.76, 0.32, 0.76, 
            0.04, 0.02, 1e-7, 0.011, 0.013, 0.016, 0.2,
            0.14, 0.0041, 0.116, 0.0041,
            3.4e-6, 1.6e-5, 0.014
        ]
        
        columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4",
                   "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa",
                   "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion", "S_H_ion",
                   "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2",
                   "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"]

        state_list = [state_zero]
        gasflow_list = [{'time': 0, 'q_gas': 0, 'q_ch4': 0}]
        extras_list = [{'time': 0, 'pH': 7.47, 'OLR': 0, 'HRT': self.V_liq/self.Q}]

        # Simulation loop
        for i in range(1, len(influent_df)):
            current_state = state_list[-1]
            current_influent = influent_df.iloc[i]
            
            # Time step
            t_span = [i-1, i]
            
            # Create state input for current time step
            state_input = [
                current_influent['S_su'], current_influent['S_aa'], current_influent['S_fa'],
                current_influent['S_va'], current_influent['S_bu'], current_influent['S_pro'],
                current_influent['S_ac'], current_influent['S_h2'], current_influent['S_ch4'],
                current_influent['S_IC'], current_influent['S_IN'], current_influent['S_I'],
                current_influent['X_xc'], current_influent['X_ch'], current_influent['X_pr'],
                current_influent['X_li'], current_influent['X_su'], current_influent['X_aa'],
                current_influent['X_fa'], current_influent['X_c4'], current_influent['X_pro'],
                current_influent['X_ac'], current_influent['X_h2'], current_influent['X_I'],
                current_influent['S_cation'], current_influent['S_anion']
            ]
            
            # Solve ODE
            result = scipy.integrate.solve_ivp(
                self._ADM1_ODE, t_span, current_state, 
                method='BDF', args=(state_input,), rtol=1e-6, atol=1e-8
            )
            
            new_state = result.y[:, -1].copy()
            
            # Solve DAE equations
            new_state = self._solve_DAE(new_state, state_input)
            
            # Calculate gas flow
            q_gas, q_ch4 = self._calculate_gas_flow(new_state)
            
            # Calculate pH and other process indicators
            pH = -np.log10(max(new_state[26], 1e-14))
            OLR = self._calculate_OLR(current_influent)
            HRT = self.V_liq / self.Q
            
            state_list.append(new_state)
            gasflow_list.append({'time': i, 'q_gas': q_gas, 'q_ch4': q_ch4})
            extras_list.append({'time': i, 'pH': pH, 'OLR': OLR, 'HRT': HRT})

        # Store results
        self.simulate_results = pd.DataFrame(state_list, columns=columns)
        gasflow_df = pd.DataFrame(gasflow_list)
        extras_df = pd.DataFrame(extras_list)
        
        self.output_data = pd.concat([
            influent_df[['time']],
            gasflow_df[['q_gas', 'q_ch4']],
            extras_df[['pH', 'OLR', 'HRT']]
        ], axis=1)

    def _ADM1_ODE(self, t, state, state_input):
        """ADM1 ODE system based on BSM2 implementation"""
        # Extract state variables
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I = state[:12]
        X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I = state[12:24]
        S_cation, S_anion, S_H_ion = state[24:27]
        S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion = state[27:35]
        S_gas_h2, S_gas_ch4, S_gas_co2 = state[35:38]
        
        # Extract input variables
        S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in = state_input[:7]
        S_h2_in, S_ch4_in, S_IC_in, S_IN_in, S_I_in = state_input[7:12]
        X_xc_in, X_ch_in, X_pr_in, X_li_in, X_su_in, X_aa_in = state_input[12:18]
        X_fa_in, X_c4_in, X_pro_in, X_ac_in, X_h2_in, X_I_in = state_input[18:24]
        S_cation_in, S_anion_in = state_input[24:26]

        # Gas phase equations
        p_gas_h2 = S_gas_h2 * self.R * self.T_op / 16
        p_gas_ch4 = S_gas_ch4 * self.R * self.T_op / 64
        p_gas_co2 = S_gas_co2 * self.R * self.T_op

        # Inhibition functions
        I_pH_aa = (self.K_pH_aa**self.nn_aa) / (S_H_ion**self.nn_aa + self.K_pH_aa**self.nn_aa)
        I_pH_ac = (self.K_pH_ac**self.n_ac) / (S_H_ion**self.n_ac + self.K_pH_ac**self.n_ac)
        I_pH_h2 = (self.K_pH_h2**self.n_h2) / (S_H_ion**self.n_h2 + self.K_pH_h2**self.n_h2)
        I_IN_lim = 1 / (1 + (self.K_S_IN / S_IN))
        I_h2_fa = 1 / (1 + (S_h2 / self.K_I_h2_fa))
        I_h2_c4 = 1 / (1 + (S_h2 / self.K_I_h2_c4))
        I_h2_pro = 1 / (1 + (S_h2 / self.K_I_h2_pro))
        I_nh3 = 1 / (1 + (S_nh3 / self.K_I_nh3))

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        # Biochemical process rates
        Rho_1 = self.k_dis * X_xc
        Rho_2 = self.k_hyd_ch * X_ch
        Rho_3 = self.k_hyd_pr * X_pr
        Rho_4 = self.k_hyd_li * X_li
        Rho_5 = self.k_m_su * S_su / (self.K_S_su + S_su) * X_su * I_5
        Rho_6 = self.k_m_aa * S_aa / (self.K_S_aa + S_aa) * X_aa * I_6
        Rho_7 = self.k_m_fa * S_fa / (self.K_S_fa + S_fa) * X_fa * I_7
        Rho_8 = self.k_m_c4 * S_va / (self.K_S_c4 + S_va) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9 = self.k_m_c4 * S_bu / (self.K_S_c4 + S_bu) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = self.k_m_pro * S_pro / (self.K_S_pro + S_pro) * X_pro * I_10
        Rho_11 = self.k_m_ac * S_ac / (self.K_S_ac + S_ac) * X_ac * I_11
        Rho_12 = self.k_m_h2 * S_h2 / (self.K_S_h2 + S_h2) * X_h2 * I_12

        # Decay rates
        Rho_13 = self.k_dec_X_su * X_su
        Rho_14 = self.k_dec_X_aa * X_aa
        Rho_15 = self.k_dec_X_fa * X_fa
        Rho_16 = self.k_dec_X_c4 * X_c4
        Rho_17 = self.k_dec_X_pro * X_pro
        Rho_18 = self.k_dec_X_ac * X_ac
        Rho_19 = self.k_dec_X_h2 * X_h2

        # Gas transfer rates
        Rho_T_8 = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
        Rho_T_9 = self.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4)
        Rho_T_10 = self.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2)

        # Gas flow calculation
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        q_gas = max(0, self.k_p * (p_gas - self.p_atm))

        # Carbon balance for S_IC
        s_1 = -self.C_xc + self.f_sI_xc * self.C_sI + self.f_ch_xc * self.C_ch + self.f_pr_xc * self.C_pr + self.f_li_xc * self.C_li + self.f_xI_xc * self.C_xI
        s_2 = -self.C_ch + self.C_su
        s_3 = -self.C_pr + self.C_aa
        s_4 = -self.C_li + (1 - self.f_fa_li) * self.C_su + self.f_fa_li * self.C_fa
        s_5 = -self.C_su + (1 - self.Y_su) * (self.f_bu_su * self.C_bu + self.f_pro_su * self.C_pro + self.f_ac_su * self.C_ac) + self.Y_su * self.C_bac
        s_6 = -self.C_aa + (1 - self.Y_aa) * (self.f_va_aa * self.C_va + self.f_bu_aa * self.C_bu + self.f_pro_aa * self.C_pro + self.f_ac_aa * self.C_ac) + self.Y_aa * self.C_bac
        s_7 = -self.C_fa + (1 - self.Y_fa) * 0.7 * self.C_ac + self.Y_fa * self.C_bac
        s_8 = -self.C_va + (1 - self.Y_c4) * 0.54 * self.C_pro + (1 - self.Y_c4) * 0.31 * self.C_ac + self.Y_c4 * self.C_bac
        s_9 = -self.C_bu + (1 - self.Y_c4) * 0.8 * self.C_ac + self.Y_c4 * self.C_bac
        s_10 = -self.C_pro + (1 - self.Y_pro) * 0.57 * self.C_ac + self.Y_pro * self.C_bac
        s_11 = -self.C_ac + (1 - self.Y_ac) * self.C_ch4 + self.Y_ac * self.C_bac
        s_12 = (1 - self.Y_h2) * self.C_ch4 + self.Y_h2 * self.C_bac
        s_13 = -self.C_bac + self.C_xc

        Sigma = (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + 
                s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + 
                s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

        # Differential equations (1-38)
        dxdt = [
            # Soluble components (1-12)
            self.Q/self.V_liq * (S_su_in - S_su) + Rho_2 + (1 - self.f_fa_li) * Rho_4 - Rho_5,
            self.Q/self.V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6,
            self.Q/self.V_liq * (S_fa_in - S_fa) + self.f_fa_li * Rho_4 - Rho_7,
            self.Q/self.V_liq * (S_va_in - S_va) + (1 - self.Y_aa) * self.f_va_aa * Rho_6 - Rho_8,
            self.Q/self.V_liq * (S_bu_in - S_bu) + (1 - self.Y_su) * self.f_bu_su * Rho_5 + (1 - self.Y_aa) * self.f_bu_aa * Rho_6 - Rho_9,
            self.Q/self.V_liq * (S_pro_in - S_pro) + (1 - self.Y_su) * self.f_pro_su * Rho_5 + (1 - self.Y_aa) * self.f_pro_aa * Rho_6 + (1 - self.Y_c4) * 0.54 * Rho_8 - Rho_10,
            self.Q/self.V_liq * (S_ac_in - S_ac) + (1 - self.Y_su) * self.f_ac_su * Rho_5 + (1 - self.Y_aa) * self.f_ac_aa * Rho_6 + (1 - self.Y_fa) * 0.7 * Rho_7 + (1 - self.Y_c4) * 0.31 * Rho_8 + (1 - self.Y_c4) * 0.8 * Rho_9 + (1 - self.Y_pro) * 0.57 * Rho_10 - Rho_11,
            self.Q/self.V_liq * (S_h2_in - S_h2) + (1 - self.Y_su) * self.f_h2_su * Rho_5 + (1 - self.Y_aa) * self.f_h2_aa * Rho_6 + (1 - self.Y_fa) * 0.3 * Rho_7 + (1 - self.Y_c4) * 0.15 * Rho_8 + (1 - self.Y_c4) * 0.2 * Rho_9 + (1 - self.Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8,
            self.Q/self.V_liq * (S_ch4_in - S_ch4) + (1 - self.Y_ac) * Rho_11 + (1 - self.Y_h2) * Rho_12 - Rho_T_9,
            self.Q/self.V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10,
            self.Q/self.V_liq * (S_IN_in - S_IN) + (self.N_xc - self.f_xI_xc * self.N_I - self.f_sI_xc * self.N_I - self.f_pr_xc * self.N_aa) * Rho_1 - self.Y_su * self.N_bac * Rho_5 + (self.N_aa - self.Y_aa * self.N_bac) * Rho_6 - self.Y_fa * self.N_bac * Rho_7 - self.Y_c4 * self.N_bac * Rho_8 - self.Y_c4 * self.N_bac * Rho_9 - self.Y_pro * self.N_bac * Rho_10 - self.Y_ac * self.N_bac * Rho_11 - self.Y_h2 * self.N_bac * Rho_12 + (self.N_bac - self.N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19),
            self.Q/self.V_liq * (S_I_in - S_I) + self.f_sI_xc * Rho_1,
            
            # Particulate components (13-24)
            self.Q/self.V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19,
            self.Q/self.V_liq * (X_ch_in - X_ch) + self.f_ch_xc * Rho_1 - Rho_2,
            self.Q/self.V_liq * (X_pr_in - X_pr) + self.f_pr_xc * Rho_1 - Rho_3,
            self.Q/self.V_liq * (X_li_in - X_li) + self.f_li_xc * Rho_1 - Rho_4,
            self.Q/self.V_liq * (X_su_in - X_su) + self.Y_su * Rho_5 - Rho_13,
            self.Q/self.V_liq * (X_aa_in - X_aa) + self.Y_aa * Rho_6 - Rho_14,
            self.Q/self.V_liq * (X_fa_in - X_fa) + self.Y_fa * Rho_7 - Rho_15,
            self.Q/self.V_liq * (X_c4_in - X_c4) + self.Y_c4 * Rho_8 + self.Y_c4 * Rho_9 - Rho_16,
            self.Q/self.V_liq * (X_pro_in - X_pro) + self.Y_pro * Rho_10 - Rho_17,
            self.Q/self.V_liq * (X_ac_in - X_ac) + self.Y_ac * Rho_11 - Rho_18,
            self.Q/self.V_liq * (X_h2_in - X_h2) + self.Y_h2 * Rho_12 - Rho_19,
            self.Q/self.V_liq * (X_I_in - X_I) + self.f_xI_xc * Rho_1,
            
            # Cations and anions (25-26)
            self.Q/self.V_liq * (S_cation_in - S_cation),
            self.Q/self.V_liq * (S_anion_in - S_anion),
            
            # Ion states (27-35) - set to zero for ODE implementation
            0,  # S_H_ion
            0,  # S_va_ion
            0,  # S_bu_ion
            0,  # S_pro_ion
            0,  # S_ac_ion
            0,  # S_hco3_ion
            0,  # S_co2
            0,  # S_nh3
            0,  # S_nh4_ion
            
            # Gas phase (36-38)
            q_gas/self.V_gas * (-S_gas_h2) + Rho_T_8 * self.V_liq/self.V_gas,
            q_gas/self.V_gas * (-S_gas_ch4) + Rho_T_9 * self.V_liq/self.V_gas,
            q_gas/self.V_gas * (-S_gas_co2) + Rho_T_10 * self.V_liq/self.V_gas
        ]
        
        return dxdt

    def _solve_DAE(self, state, state_input):
        """Solve DAE equations for ion concentrations and pH"""
        eps = 1e-7
        tol = 1e-12
        max_iter = 1000
        
        # Extract current state
        S_va, S_bu, S_pro, S_ac, S_IC, S_IN = state[3], state[4], state[5], state[6], state[9], state[10]
        S_cation, S_anion = state[24], state[25]
        S_H_ion = state[26]
        
        # Newton-Raphson for S_H_ion
        for i in range(max_iter):
            S_va_ion = self.K_a_va * S_va / (self.K_a_va + S_H_ion)
            S_bu_ion = self.K_a_bu * S_bu / (self.K_a_bu + S_H_ion)
            S_pro_ion = self.K_a_pro * S_pro / (self.K_a_pro + S_H_ion)
            S_ac_ion = self.K_a_ac * S_ac / (self.K_a_ac + S_H_ion)
            S_hco3_ion = self.K_a_co2 * S_IC / (self.K_a_co2 + S_H_ion)
            S_nh3 = self.K_a_IN * S_IN / (self.K_a_IN + S_H_ion)
            
            shdelta = (S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - 
                      S_ac_ion/64.0 - S_pro_ion/112.0 - S_bu_ion/160.0 - 
                      S_va_ion/208.0 - self.K_w/S_H_ion - S_anion)
            
            if abs(shdelta) < tol:
                break
                
            shgradeq = (1 + self.K_a_IN * S_IN / ((self.K_a_IN + S_H_ion)**2) +
                       self.K_a_co2 * S_IC / ((self.K_a_co2 + S_H_ion)**2) +
                       self.K_a_ac * S_ac / (64.0 * (self.K_a_ac + S_H_ion)**2) +
                       self.K_a_pro * S_pro / (112.0 * (self.K_a_pro + S_H_ion)**2) +
                       self.K_a_bu * S_bu / (160.0 * (self.K_a_bu + S_H_ion)**2) +
                       self.K_a_va * S_va / (208.0 * (self.K_a_va + S_H_ion)**2) +
                       self.K_w / (S_H_ion**2))
            
            S_H_ion = max(tol, S_H_ion - shdelta/shgradeq)
        
        # Update state with DAE solutions
        state[26] = S_H_ion
        state[27] = S_va_ion
        state[28] = S_bu_ion
        state[29] = S_pro_ion
        state[30] = S_ac_ion
        state[31] = S_hco3_ion
        state[32] = S_IC - S_hco3_ion  # S_co2
        state[33] = S_nh3
        state[34] = S_IN - S_nh3  # S_nh4_ion
        
        return state

    def _calculate_gas_flow(self, state):
        """Calculate gas flow rates"""
        S_gas_h2, S_gas_ch4, S_gas_co2 = state[35], state[36], state[37]
        
        p_gas_h2 = S_gas_h2 * self.R * self.T_op / 16
        p_gas_ch4 = S_gas_ch4 * self.R * self.T_op / 64
        p_gas_co2 = S_gas_co2 * self.R * self.T_op
        
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        q_gas = max(0, self.k_p * (p_gas - self.p_atm))
        
        if p_gas > 0:
            q_ch4 = q_gas * (p_gas_ch4 / p_gas)
        else:
            q_ch4 = 0
            
        return q_gas, q_ch4

    def _calculate_OLR(self, influent_row):
        """Calculate Organic Loading Rate"""
        total_cod = (influent_row['S_su'] + influent_row['S_aa'] + influent_row['S_fa'] +
                    influent_row['S_va'] + influent_row['S_bu'] + influent_row['S_pro'] +
                    influent_row['S_ac'] + influent_row['X_ch'] + influent_row['X_pr'] +
                    influent_row['X_li'])
        return total_cod * self.Q / self.V_liq

    def run(self):
        """Execute the full simulation process"""
        influent_df = self.run_codigestion()
        self.run_simulation(influent_df)

    def get_results(self):
        """Return process indicators for plotting"""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        return self.output_data

    def save_output(self, file_path="ADM1_Results.xlsx"):
        """Save results to Excel file"""
        if self.output_data is None:
            raise ValueError("Simulation has not been run yet. Call run() first.")
        
        with pd.ExcelWriter(file_path) as writer:
            self.simulate_results.to_excel(writer, sheet_name='ADM1_States', index=False)
            self.output_data.to_excel(writer, sheet_name='Process_Data', index=False)