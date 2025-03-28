import numpy as np
from scipy.optimize import minimize
import CoolProp.CoolProp as CP

def calc_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out):
    delta_T1 = T_hot_in - T_cold_out
    delta_T2 = T_hot_out - T_cold_in
    if np.isclose(delta_T1, delta_T2):
        return delta_T1
    return (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)

def overall_heat_trf_coeff(mdot_mix, rho_mix, mu_water_avg, A_tube, A_shell,
                                  k_wall, k_steam, k_mix,
                                  rho_steam, V_steam, mu_steam, r_tube_inner, r_tube_outer,
                                 L_shell_inner, L_shell_outer, L, Rf_hot, Rf_cold, Pr_mix, Pr_steam, N_tubes):

    # Calculates the overall heat transfer coefficient U for a shell-and-tube heat exchanger
    # with a square shell (inner side lengthL_shell_inner and outer side length L_shell_inner).

    # --- Tube-side calculations remain unchanged ---
    inner_area_tube = np.pi * (r_tube_inner ** 2)
    V_avg = (mdot_mix/N_tubes) / (inner_area_tube * rho_mix)
    Nu = 0.023 * ((rho_mix * V_avg * 2 * r_tube_inner / mu_water_avg) ** 0.8) * (Pr_mix ** 0.3)
    h_tube = (Nu * k_mix) / (2 * r_tube_inner)

    # --- Shell-side calculations for a square shell ---
    # Clear area (available for cross-flow) approximated as the inner square area.
    A_c = L_shell_inner ** 2
    # Wetted perimeter for a square is 4*L_inner.
    P_w = 4 *L_shell_inner
    # Hydraulic diameter for a square is then:
    D_h = 4 * A_c / P_w  # which simplifies toL_shell_inner

    # Typical Kern's method correlations:
    C = 0.36       # Empirical constant
    m_exp = 0.55   # Empirical exponent
    n_exp = 1/3   # Empirical exponent

    # h_shell_ideal based on square hydraulic diameter
    h_shell_ideal = C * (k_steam / D_h) * ((rho_steam * V_steam * D_h / mu_steam) ** m_exp) * (Pr_steam ** n_exp)

    # Apply baffle correction factors
    J_c = 0.9  # Cross‐flow Distribution Correction Factor
    J_l = 0.95  # Leakage Correction Factor
    J_b = 0.9  # Bundle Bypass Correction Factor
    J_s = 0.9  # Shell Pass Correction factor
    h_shell_corrected = h_shell_ideal * J_c * J_l * J_b * J_s

    # --- Overall resistance calculation ---
    R_tube = 1 / (h_tube * A_tube)  # Tube-side convective resistance
    R_shell = 1 / (h_shell_corrected * A_shell)  # Shell-side convective resistance
    R_wall = np.log(r_tube_outer / r_tube_inner) / (2 * np.pi * k_wall * L)  # Conduction resistance through tube wall
    R_f = Rf_hot + Rf_cold  # Combined fouling resistances

    U = 1 / (R_tube + R_shell + R_wall + R_f)
    return U

def effectiveness_NTU(U, A, C_cold, C_hot, F):
    C_min = min(C_cold, C_hot)
    C_max = max(C_cold, C_hot)
    Cr = C_min / C_max if C_max > 0 else 0.0

    NTU = (U * A) / C_min
    NTU_eff = F * NTU
    if np.isclose(Cr, 1.0):
        effectiveness = NTU_eff / (1 + NTU_eff)
    else:
        effectiveness = (1 - np.exp(-NTU_eff * (1 - Cr))) / (1 - Cr * np.exp(-NTU_eff * (1 - Cr)))
    return effectiveness

def get_darcy_factor(rho, V, D, mu):
    Re = (rho * V * D) / mu
    if Re <= 2100:
        f = 64 / Re
    else:
        f = 0.316 / (Re ** 0.25)
    return f

# Define mixture properties
mdot_CaNO3 = 738.45    # tons/hr
mdot_NH4NO3 = 39.16    # tons/hr
mdot_mix = 0.8*((mdot_CaNO3 + mdot_NH4NO3) * 1000 / 3600)/2   # kg/s
rho_water = CP.PropsSI('D', 'T', 25 + 273.15, 'P', 101325, 'Water')

V_CaNO3 = 369.23*1000*1000/2.504 + 369.23*1000*1000/1
V_NH4NO3 = 36.03*1000*1000/1.725 + 3.13*1000*1000/1
Vol_mix = ((V_CaNO3 + V_NH4NO3)/ (1000000*3600))/2 # m^3/s

rho_mix = mdot_mix / Vol_mix

# Estimation of the Specific Heat Capacity of the Mixture
cp_CaNO3_25C = 0.8836075564*1000                # J/kg·K
cp_CaNO3_150C = 0.9902498477*1000               # J/kg·K
cp_CaNO3_avg = (cp_CaNO3_150C + cp_CaNO3_150C)/2 # J/kg·K

cp_NH4NO3_avg = 1.74*1000   # J/kg·K

cp_water_25 = CP.PropsSI("Cpmass", "P", 101325, "T", 25 + 273.15, "Water")
cp_water_150 = CP.PropsSI("Cpmass", "P", 101325, "T", 150 + 273.15, "Water")
cp_water_avg = (cp_water_150 + cp_water_150)/2
cp_mix = (369.23*cp_CaNO3_avg + 36.03*cp_NH4NO3_avg + 372.36*cp_water_avg)/777.61

# Heat Capacity of Mixture
Cp_mix = mdot_mix * cp_mix

# Average Dynamic Viscosity of the Mixture (Assumed water because it is an aqueous solution)
mu_water_25C = CP.PropsSI('V', 'T', 25 + 273.15, 'P', 101325, 'Water')
mu_water_150C = CP.PropsSI('V', 'T', 150 + 273.15, 'P', 101325, 'Water')
mu_water_avg = (mu_water_150C + mu_water_150C)/2
#mu_water_avg = CP.PropsSI('V', 'T', 25 + 273.15, 'P', 101325, 'Water')

# Pressure and Temperature of the Mixture, constant temperature(enthalpy of vaporization and enthalpy of formation)
P_mix_1 = 101325  # edit for the height, Pa
T_mix_1 = 150 + 273.15  # K
T_mix_2 = 150 + 273.15  # K

# Heat needed to be gained by the mixture
Q_mix = 189830956.11/2 # J/s

# Thermal Conductivity Coefficients, Estimates
k_wall = 16  # From Material of Tube Wall, W/m·K
k_mix = 0.6  # W/m·K

Pr_mix = mu_water_avg*cp_mix/k_mix # Prandtl's Number

# Steam Properties, 6 bar saturated steam source, mdot_steam to be changed accordingly
steam_quality_1 = 1
mdot_steam = 60.4 + 1*0.01 # kg/s
P_steam_1 = 600000 # Pa
T_steam_1 = CP.PropsSI("T", 'P', P_steam_1, 'Q', steam_quality_1, 'Water')

rho_steam = CP.PropsSI('D', 'P', P_steam_1, 'Q', steam_quality_1, 'Water')
h_steam_1 = CP.PropsSI("H", 'P', P_steam_1, 'Q', steam_quality_1, 'Water')
Voldot_steam = mdot_steam / rho_steam # Volumetric Flow Rate of Steam

mu_steam = CP.PropsSI('V', 'P', P_steam_1, 'Q', steam_quality_1, 'Water')
mu_steam_corrected = 1.2*mu_steam # estimated correction

cp_steam = CP.PropsSI("C", "P", P_steam_1, "Q", steam_quality_1, "Water")
Cp_steam = mdot_steam * cp_steam # Heat Capacity of Steam
Pr_steam = CP.PropsSI('Prandtl', 'P', P_steam_1, 'Q', steam_quality_1, 'Water')

k_steam = CP.PropsSI('L', 'P', P_steam_1, 'Q', steam_quality_1, 'Water')  # W/m·K

# Fouling Factors
Rf_steam = 0.0002
Rf_mix = 0.0002

def total_tube_area(r_tube_inner, tube_length, N_tubes):
    # Calculate total tube inner surface area (proxy for cost/size)
    return np.pi * (2 * r_tube_inner) * tube_length * N_tubes

def total_shell_area(r_tube_outer, tube_length, N_tubes):
    # Calculate total tube outer surface area (proxy for shell area)
    return np.pi * (2 * r_tube_outer) * tube_length * N_tubes

def get_pressure_loss(Vol_mix, r_tube_inner, tube_length, N_tubes):
    inner_area_tube = np.pi * (r_tube_inner ** 2)
    V_avg = (Vol_mix/N_tubes) / inner_area_tube
    fric = get_darcy_factor(rho_mix, V_avg, 2 * r_tube_inner, mu_water_avg)

    major_loss = fric * (tube_length / (2 * r_tube_inner)) * ((0.5 * V_avg ** 2) /9.81)
    k_l_entry = 0.5
    k_l_exit = 1
    entry_loss = k_l_entry * (V_avg ** 2) / (2 * 9.81)
    exit_loss = k_l_exit * (V_avg ** 2) / (2 * 9.81)

    P_loss = rho_mix * 9.81 * (entry_loss + major_loss + exit_loss)
    return P_loss


def get_shell_pressure_loss(x):

    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    # 1. Basic geometry
    d_o = 2.0 * r_tube_outer  # [m] outer tube diameter
    P_t = norm_pitch  # [m] tube pitch (assumed square pitch)

    # 2. Equivalent diameter for square pitch crossflow
    #    (Note: must ensure (P_t^2) > (pi * d_o^2 / 4)!)
    D_e = (4.0 * (P_t ** 2 - 0.25 * np.pi * d_o ** 2)) / (np.pi * d_o)

    # 3. Estimate crossflow area S_m  (Kern uses typical net area around tubes).
    #    A very simplified approach:
    #    We assume the shell side is ~L_shell_inner wide,
    #    minus one column of tubes, times the baffle spacing.
    baffle_spacing = tube_length / N_baffles
    # Number of tube columns in crossflow (approx.)
    # E.g.  int(L_shell_inner / P_t), but we'll do a rough guess:
    N_col = max(1, int(np.floor(L_shell_inner / P_t)))

    # A guess for net free area in crossflow region:
    # (width minus the tubes) * baffle spacing
    # for each crossflow pass. Possibly multiply by # of passes or not.
    S_m = (L_shell_inner - N_col * d_o) * baffle_spacing

    if S_m <= 0:
        # If geometry is tight, we have to fallback or raise an error
        raise ValueError("Shell geometry leaves no free flow area!")

    # 4. Mass velocity in crossflow
    G_s = mdot_steam / S_m  # [kg/(m^2*s)]

    # 5. Shell-side Reynolds number
    Re_s = (G_s * D_e) / mu_steam

    # 6. Friction factor from a typical Kern correlation
    #    e.g. f = 0.079 Re^-0.25 for turbulent crossflow
    f_s = 0.079 * (Re_s ** -0.25)

    # 7. Number of crossflow passes
    #    (approx. equal to number of baffles, or N_baffles - 1, etc.)
    N_cf = N_baffles

    # 8. Pressure drop
    #    A typical Kern formula:
    #        dP = f_s * (G_s^2 / (2*rho_steam)) * N_cf
    #    Optionally multiply by (baffle_spacing / D_e) if desired
    #    or (N_cf * baffle_spacing / D_e).

    P_loss_shell = f_s * (G_s ** 2 / (2.0 * rho_steam)) * N_cf

    return P_loss_shell

def objective(x):
    # Objective: minimize total tube area (proxy for cost/size)
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x
    A = total_tube_area(r_tube_inner, tube_length, N_tubes)
    return A

def objective_shell(x):
    # Objective: minimize total shell area (proxy for cost/size)
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x
    A = total_shell_area(r_tube_outer, tube_length, N_tubes)
    return A

def constraint_heat_duty(x):
    # Constraint: computed heat transfer must equal required heat duty
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    A_tube = total_tube_area(r_tube_inner, tube_length, N_tubes)
    A_shell = total_shell_area(r_tube_outer, tube_length, N_tubes)
    V_steam = Voldot_steam / (np.pi * L_shell_inner ** 2)
    U = overall_heat_trf_coeff(mdot_mix, rho_mix, mu_water_avg, A_tube, A_shell,
                               k_wall, k_steam, k_mix, rho_steam, V_steam, mu_steam,
                               r_tube_inner, r_tube_outer, L_shell_inner, L_shell_outer,
                               tube_length, Rf_steam, Rf_mix, Pr_mix, Pr_steam, N_tubes)
    effect = effectiveness_NTU(U, A_tube, Cp_mix, Cp_steam, F = 1.0)

    h_steam_2 = h_steam_1 - Q_mix/(mdot_steam*effect)

    P_loss_shell = get_shell_pressure_loss(x)
    P_steam_2 = P_steam_1 - P_loss_shell

    # Check for physically valid outlet pressure:
    if P_steam_2 < 60000:
        # Return a large penalty if steam outlet pressure is below atmospheric pressure
        return -1e6

    T_steam_2 = CP.PropsSI('T', 'P', P_steam_2, 'H', h_steam_2, 'Water')
    LMTD = calc_LMTD(T_steam_1, T_steam_2, T_mix_1, T_mix_2)
    Q_computed = effect * U * A_tube * LMTD

    if Q_computed - Q_mix < 0:
        return -1e6
    else:
        return Q_computed - Q_mix  # must equal 0


def constraint_wall_thickness(x):
    # Ensure minimum wall thickness: (r_tube_outer - r_tube_inner) >= 0.001 m
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    if r_tube_outer - r_tube_inner - 0.0001 < 0:
        return -1e6
    else:
        return r_tube_outer - r_tube_inner - 0.001

def constraint_shell_thickness(x):
    # For a square shell, the shell thickness is the difference between the outer and inner side lengths.
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x
    if L_shell_outer - L_shell_inner < 0:
        return -1e6
    else:
        return L_shell_outer - L_shell_inner - 0.01905  # Must be >= 0

def constraint_shell_pressure_loss(x):
    # Unpack design variables;
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    d_o = 2.0 * r_tube_outer  # [m] outer tube diameter
    P_t = norm_pitch  # [m] tube pitch (assumed square pitch)

    # 2. Equivalent diameter for square pitch crossflow
    #    (Note: must ensure (P_t^2) > (pi * d_o^2 / 4)!)
    D_e = (4.0 * (P_t ** 2 - 0.25 * np.pi * d_o ** 2)) / (np.pi * d_o)

    # 3. Estimate crossflow area S_m  (Kern uses typical net area around tubes).
    #    A very simplified approach:
    #    We assume the shell side is ~L_shell_inner wide,
    #    minus one column of tubes, times the baffle spacing.
    baffle_spacing = tube_length / N_baffles
    # Number of tube columns in crossflow (approx.)
    # E.g.  int(L_shell_inner / P_t), but we'll do a rough guess:
    N_col = max(1, int(np.floor(L_shell_inner / P_t)))

    # A guess for net free area in crossflow region:
    # (width minus the tubes) * baffle spacing
    # for each crossflow pass. Possibly multiply by # of passes or not.
    S_m = (L_shell_inner - N_col * d_o) * baffle_spacing

    if S_m <= 0:
        # If geometry is tight, we have to fallback or raise an error
        raise ValueError("Shell geometry leaves no free flow area!")

    # 4. Mass velocity in crossflow
    G_s = mdot_steam / S_m  # [kg/(m^2*s)]

    # 5. Shell-side Reynolds number
    Re_s = (G_s * D_e) / mu_steam

    # 6. Friction factor from a typical Kern correlation
    #    e.g. f = 0.079 Re^-0.25 for turbulent crossflow
    f_s = 0.079 * (Re_s ** -0.25)

    # 7. Number of crossflow passes
    #    (approx. equal to number of baffles, or N_baffles - 1, etc.)
    N_cf = N_baffles

    # 8. Pressure drop
    #    A typical Kern formula:
    #        dP = f_s * (G_s^2 / (2*rho_steam)) * N_cf
    #    Optionally multiply by (baffle_spacing / D_e) if desired
    #    or (N_cf * baffle_spacing / D_e).

    P_loss_shell = f_s * (G_s ** 2 / (2.0 * rho_steam)) * N_cf

    # Enforce that P_loss_shell is > 0 Pa (Realistic).
    return P_loss_shell

def constraint_bundle_diameter(x):
    # Unpack the design variables; hereL_shell_inner is the inner side length of the square shell.
    r_tube_inner, r_tube_outer, tube_length, N_tubes,L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    # Calculate the effective bundle diameter.
    # (Assuming the bundle remains roughly circular, we use the same formula.)
    bundle_diameter = 2 * norm_pitch * np.sqrt(N_tubes / np.pi)

    # In a square shell, the largest circle that can be inscribed has a diameter equal to L_shell_inner.
    # Therefore, we require that:
    if L_shell_inner - bundle_diameter < 0:
        return -1e6
    else:
        return L_shell_inner - bundle_diameter

def constraint_pitch(x):
    # Unpack the design variables
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    # Pitch must not be lower than 2 * r_tube_outer
    if norm_pitch - 2 * r_tube_outer < 0:
        return -1e6
    else:
        return norm_pitch - 2 * r_tube_outer - 0.01 # Must be >= 0

def constraint_velocity(x):
    # ensure that the average velocity in the tubes remains above a threshold
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x
    inner_area_tube = np.pi * (r_tube_inner ** 2)

    V_avg = (mdot_mix/N_tubes) / (inner_area_tube * rho_mix)
    return V_avg - 0.1  # for example, require at least 1 m/s

def constraint_effectiveness(x):
    # Impose a lower bound on the effectiveness (or NTU) to ensure that the exchanger design achieves a minimum heat transfer performance
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x
    A_tube = total_tube_area(r_tube_inner, tube_length, N_tubes)
    A_shell = total_shell_area(r_tube_outer, tube_length, N_tubes)
    V_steam = Voldot_steam / (np.pi * L_shell_inner ** 2)
    U = overall_heat_trf_coeff(mdot_mix, rho_mix, mu_water_avg, A_tube, A_shell,
                               k_wall, k_steam, k_mix, rho_steam, V_steam, mu_steam,
                               r_tube_inner, r_tube_outer, L_shell_inner, L_shell_outer,
                               tube_length, Rf_steam, Rf_mix, Pr_mix, Pr_steam, N_tubes)
    effect = effectiveness_NTU(U, A_tube, Cp_mix, Cp_steam, F=1.0)
    # For example, require effectiveness to be at least 0.9
    return effect - 0.8

def upper_constraint_effectiveness(x):

    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x
    A_tube = total_tube_area(r_tube_inner, tube_length, N_tubes)
    A_shell = total_shell_area(r_tube_outer, tube_length, N_tubes)
    V_steam = Voldot_steam / (np.pi * L_shell_inner ** 2)
    U = overall_heat_trf_coeff(mdot_mix, rho_mix, mu_water_avg, A_tube, A_shell,
                               k_wall, k_steam, k_mix, rho_steam, V_steam, mu_steam,
                               r_tube_inner, r_tube_outer, L_shell_inner, L_shell_outer,
                               tube_length, Rf_steam, Rf_mix, Pr_mix, Pr_steam, N_tubes)
    effect = effectiveness_NTU(U, A_tube, Cp_mix, Cp_steam, F=1.0)
    # For example, require effectiveness to be lower than 1
    return 1 - effect

def constraint_shell_re(x):
    # For a square shell, assume the inner side length is L_shell_inner.
    # Unpack the design variables
    r_tube_inner, r_tube_outer, tube_length, N_tubes,L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    # Calculate the steam velocity in the square shell using its cross-sectional area.
    # For a square, area =L_shell_inner^2.
    V_steam_local = Voldot_steam / (L_shell_inner ** 2)

    # Use L_shell_inner as the characteristic length (the hydraulic diameter for a square channel isL_shell_inner).
    Re_shell = (rho_steam * V_steam_local *L_shell_inner) / mu_steam

    # For a minimum threshold (e.g., 5000), the constraint returns:
    return Re_shell - 5000

def constraint_pressure(x):
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    P_loss_tube = get_pressure_loss(Vol_mix,r_tube_inner, tube_length, N_tubes)
    P_mix_2 = P_mix_1 - P_loss_tube

    return P_mix_2 - 5000 # Must be 101325 Pa

def constraint_LMTD(x):
    # Constraint: LMTD must be higher
    r_tube_inner, r_tube_outer, tube_length, N_tubes, L_shell_inner, L_shell_outer, norm_pitch, N_baffles = x

    A_tube = total_tube_area(r_tube_inner, tube_length, N_tubes)
    A_shell = total_shell_area(r_tube_outer, tube_length, N_tubes)
    V_steam = Voldot_steam / (np.pi * L_shell_inner ** 2)
    U = overall_heat_trf_coeff(mdot_mix, rho_mix, mu_water_avg, A_tube, A_shell,
                               k_wall, k_steam, k_mix, rho_steam, V_steam, mu_steam,
                               r_tube_inner, r_tube_outer, L_shell_inner, L_shell_outer,
                               tube_length, Rf_steam, Rf_mix, Pr_mix, Pr_steam, N_tubes)
    effect = effectiveness_NTU(U, A_tube, Cp_mix, Cp_steam, F=1.0)

    h_steam_2 = h_steam_1 - Q_mix / (mdot_steam * effect)

    P_loss_shell = get_shell_pressure_loss(x)
    P_steam_2 = P_steam_1 - P_loss_shell

    # Check for physically valid outlet pressure:
    if P_steam_2 < 60000:
        # Return a large penalty if steam outlet pressure is below atmospheric pressure
        return -1e6

    T_steam_2 = CP.PropsSI('T', 'P', P_steam_2, 'H', h_steam_2, 'Water')
    LMTD = calc_LMTD(T_steam_1, T_steam_2, T_mix_1, T_mix_2)

    return LMTD - 40 # LMTD must be more than 40


bounds = [
    (0.015, 0.03),   # r_tube_inner [m]
    (0.018, 0.04),   # r_tube_outer [m] (ensuring a realistic wall thickness)
    (15, 25),         # tube_length [m] – longer tubes provide more surface area
    (1250, 2000),       # N_tubes – moderate tube count to keep the total cross‐section small enough for ~1 m/s
    (4.0, 10),       # L_shell_inner [m] – inner side length of square shell
    (5, 11),       # L_shell_outer [m] – outer side length; e.g., 5.0 m gives a shell thickness of ~0.5 m
    (0.10, 0.20),     # norm_pitch [m] – chosen so that clearance = norm_pitch - 2*r_tube_outer is positive
    (4, 12),          # N_baffles – enough baffles to generate proper steam-side turbulence
]
x0 = [
    0.025,    # r_tube_inner [m]
    0.035,    # r_tube_outer [m]
    23,     # tube_length [m]
    1800,      # N_tubes (yielding total cross-sectional area ≈ 200*π*(0.015)² ≈ 0.141 m², so V_tube ≈ 1.06 m/s)
    7,      # L_shell_inner [m]
    10,      # L_shell_outer [m]
    0.1,     # norm_pitch [m] (for r_tube_outer = 0.018 m, clearance = 0.07 - 2*0.018 = 0.034 m)
    8,       # N_baffles
]

# List of constraints: heat duty balance, wall thickness, tube-side pressure loss, and shell-side pressure loss.
constraints = [
    {'type': 'eq', 'fun': constraint_heat_duty},
    {'type': 'ineq', 'fun': constraint_wall_thickness},
    {'type': 'ineq', 'fun': constraint_shell_pressure_loss},
    {'type': 'ineq', 'fun': constraint_bundle_diameter},
    {'type': 'ineq', 'fun': constraint_pitch},
    {'type': 'ineq', 'fun': constraint_shell_thickness},
    {'type': 'ineq', 'fun': constraint_velocity},
    {'type': 'ineq', 'fun': constraint_effectiveness},
    {'type': 'ineq', 'fun': upper_constraint_effectiveness},
    {'type': 'ineq', 'fun': constraint_shell_re},
    {'type': 'eq', 'fun': constraint_pressure},
    {'type': 'ineq', 'fun':constraint_LMTD}
]

# Run the optimization using the tube area objective
solution = minimize(objective, x0, method = 'SLSQP', bounds = bounds, constraints = constraints, options = {'disp': False})
r_tube_inner_opt, r_tube_outer_opt, tube_length_opt, N_tubes_opt, L_shell_inner_opt, L_shell_outer_opt, norm_pitch_opt, N_baffles_opt = solution.x

# Calculating Overall Heat Transfer Coefficient and Effectiveness
N_tubes_opt = int(round(N_tubes_opt))
N_baffles_opt = int(round(N_baffles_opt))
t_wall_opt = r_tube_outer_opt - r_tube_inner_opt
A_tube_opt = total_tube_area(r_tube_inner_opt, tube_length_opt, N_tubes_opt)
A_shell_opt = total_shell_area(r_tube_outer_opt, tube_length_opt, N_tubes_opt)
V_steam_opt = Voldot_steam / (L_shell_inner_opt ** 2)
V_avg_opt = (Vol_mix/N_tubes_opt)/(np.pi*r_tube_inner_opt**2)
U_opt = overall_heat_trf_coeff(mdot_mix, rho_mix, mu_water_avg, A_tube_opt, A_shell_opt,
                               k_wall, k_steam, k_mix, rho_steam, V_steam_opt, mu_steam,
                               r_tube_inner_opt, r_tube_outer_opt, L_shell_inner_opt, L_shell_outer_opt,
                               tube_length_opt, Rf_steam, Rf_mix, Pr_mix, Pr_steam, N_tubes_opt)
effect_opt = effectiveness_NTU(U_opt, A_tube_opt, Cp_mix, Cp_steam, F = 1)
# Calculating Required Steam Output
h_steam_2_opt = h_steam_1 - Q_mix/(effect_opt*mdot_steam)
P_loss_shell_opt = get_shell_pressure_loss(solution.x)
P_steam_2_opt = P_steam_1 - P_loss_shell_opt
T_steam_2_opt = CP.PropsSI('T', 'P', P_steam_2_opt, 'H', h_steam_2_opt, 'Water')
steam_quality_2 = CP.PropsSI('Q', 'P', P_steam_2_opt, 'H', h_steam_2_opt, 'Water')

# Calculating the Heat Transfer from Heat Exchanger with Optimized Parameters
LMTD_opt = calc_LMTD(T_steam_1, T_steam_2_opt, T_mix_1, T_mix_2)
Q_final = effect_opt * U_opt * A_tube_opt * LMTD_opt
Q_lost_steam = mdot_steam*(h_steam_1 - h_steam_2_opt)

# Calculating Pressure Loss in tubes with optimal design parameters
P_loss_final = get_pressure_loss(Vol_mix, r_tube_inner_opt, tube_length_opt, N_tubes_opt)
P_mix_2 = P_mix_1 - P_loss_final

# For Steam Ejector
mdot_steam_2HX = mdot_steam
P_steam_2_2HX = P_steam_2_opt
steam_quality_2_2HX = steam_quality_2
h_steam_1_2HX = h_steam_1
mdot_steam_remain = (mdot_steam*2)*steam_quality_2
Q_final_2HX = Q_mix
Q_lost_steam_2HX = Q_lost_steam
T_steam_2_opt_2HX = T_steam_2_opt

print()
print("-------2nd Heat Exchanger(s)-------")
print("----Properties of the Fluids----")
print("  Initial Density of Mixture = {:.2f} kg/m^3".format(rho_mix))
print("  Volumetric Flow Rate of Mixture = {:.2f} m^3/s".format(Vol_mix))
print("  Mass Flow Rate of Mixture = {:.2f} kg/s".format(mdot_mix))
print("  Estimated Density of Mixture {:.2f} kg/m^3".format(rho_mix))
print("  Estimated Specific Heat Capacity of the Mixture = {:.2f} J/kg·K".format(cp_mix))
print("  Estimated Dynamic Viscosity of the Mixture = {:.8f} Pa·s".format(mu_water_avg))
print("  Velocity of mixture flow in tubes = {:.4f} m/s".format(V_avg_opt))
print("-------2nd Heat Exchanger(s)-------")
print("  Initial Pressure of Mixture = {:.2f} Pa".format(P_mix_1))
print("  Initial Temperature of Mixture = {:.2f} K".format(T_mix_1))
print("  Final Pressure of Mixture = {:.2f} Pa".format(P_mix_2))
print("  Final Temperature of Mixture = {:.2f} K".format(T_mix_2))
print("  Pressure Loss of Mixture = {:.2f} Pa".format(P_loss_final))
print("-------2nd Heat Exchanger(s)-------")
print("  Mass Flow rate of Steam = {:.2f} kg/s".format(mdot_steam_2HX))
print("  Initial Quality of Steam = {:.2f}".format(steam_quality_1))
print("  Initial Pressure of Steam = {:.2f} Pa".format(P_steam_1))
print("  Initial Temperature of Steam = {:.2f} K".format(T_steam_1))
print("  Initial Specific Heat Capacity of Steam = {:.2f} J/kg·K".format(cp_steam))
print("  Initial Specific Enthalpy of Steam = {:.2f} J/kg".format(h_steam_1))
print("  Initial Dynamic Viscosity of Steam = {:.10f} Pa·s".format(mu_steam))
print("  Initial Thermal Conductivity of Steam = {:.4f} W/m⋅K".format(k_steam))
print("-------2nd Heat Exchanger(s)-------")
print("  Final Quality of Steam = {:.2f}".format(steam_quality_2))
print("  Final Pressure of Steam = {:.2f} Pa".format(P_steam_2_opt))
print("  Final Temperature of Steam = {:.2f} K".format(T_steam_2_opt))
print("  Pressure Loss of Steam = {:.2f} Pa".format(P_loss_shell_opt))
print("  Final Specific Enthalpy of Steam = {:.2f} J/kg".format(h_steam_2_opt))
print("  Total Remaining Steam Vapour = {:.2f} kg/s".format(mdot_steam_remain))
print("-------2nd Heat Exchanger(s)-------")
print("Optimized Design Parameters (Heat Transfer Tube Area Objective):")
print("  r_tube_inner = {:.4f} m".format(r_tube_inner_opt))
print("  r_tube_outer = {:.4f} m".format(r_tube_outer_opt))
print("  Velocity of mixture flow in tubes = {:.4f} m/s".format(V_avg_opt))
print("  L_shell_inner = {:.4f} m".format(L_shell_inner_opt))
print("  L_shell_outer = {:.4f} m".format(L_shell_outer_opt))
print("  Tube length = {:.2f} m".format(tube_length_opt))
print("  Number of tubes = {}".format(N_tubes_opt))
print("  Number of Baffles = {}".format(N_baffles_opt))
print("  Optimal Normal Pitch = {} m".format(norm_pitch_opt))
print("  Tube wall thickness = {:.4f} m".format(t_wall_opt))
print("  Total tube area = {:.2f} m²".format(A_tube_opt))
print("----Resulting Heat Transfer Values----")
print("  Overall Heat Transfer Coefficient, U = {:.2f} W/m²·K".format(U_opt))
print("  Optimal Log Mean Temperature Difference, LMTD = {:.2f} K".format(LMTD_opt))
print("  Optimal Effectiveness = {:.2f}".format(effect_opt))
print("-------2nd Heat Exchanger(s)-------")
print("  Heat Transfer Area for Shell Side = {:.2f} m^2".format(A_shell_opt))
print("  Heat Transfer Area for Tube Side = {:.2f} m^2".format(A_tube_opt))
print("  Thermal Energy lost from steam, Q_loss_steam = {:.2f} MW".format(Q_lost_steam/1000000))
print("  Thermal Energy Required by mixture, Q_mix = {:.2f} MW".format(Q_mix/1000000))
print("  Actual Heat Transfer from Optimized Design Parameters, Q_final = {:.2f} MW".format(Q_final/1000000))
if (Q_mix - Q_final)/1000000 > 0:
    print("  Missing Heat Duty = -{:.2f} MW".format((Q_mix - Q_final) / 1000000))
else:
    print("  Excess Heat Duty = +{:.2f} MW".format(-(Q_mix - Q_final) / 1000000))

print("  Thermal Efficiency of Heat Exchanger = {:.2f}".format(Q_final/Q_lost_steam))

print()
print()
print()