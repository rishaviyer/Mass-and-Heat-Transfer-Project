import CoolProp.CoolProp as CP
import math
from Square_2nd_Heat_Exchanger import mdot_steam_remain, P_steam_2_2HX, T_steam_2_opt_2HX, steam_quality_2_2HX

# Estimated Value, Pressure Drop taken as the average
Pp0 = P_steam_2_2HX - (589.97 +476.89)/2
Qp0 = 1
Tp0 = CP.PropsSI('T', 'P', Pp0, 'Q', 1, 'Water')


Ps0 = 101325 #Pa
Ts0 = 423.026324 #K

def ejector_model():

    eta_n = 0.9  # Nozzle efficiency

    P1 = 1E5 #Pa

    sp0 = CP.PropsSI('S','P',Pp0,'Q', Qp0,'Water')  # Entropy at nozzle inlet (primary)
    hp0 = CP.PropsSI('H','P',Pp0,'Q', Qp0, 'Water')  # Enthalpy at nozzle inlet (primary)
    ss0 = CP.PropsSI('S','P',Ps0,'T', Ts0,'Water')  # Entropy at nozzle inlet (Secondary)
    hs0 = CP.PropsSI('H','P',Ps0,'T', Ts0, 'Water')  # Enthalpy at nozzle inlet (secondary)
    

    hp1s = CP.PropsSI('H','P',P1,'S',sp0,'Water')  # Enthalpy at nozzle exit (primary and constant entropy)
    up1 = math.sqrt(2 * eta_n * (hp0 - hp1s))
    

    hs1s = CP.PropsSI('H','P',P1,'S',ss0,'Water')  # Enthalpy at nozzle exit (secondary and constant entropy)
    us1 = math.sqrt(2 * eta_n * (hs0 - hs1s))
    

    rhop1 = CP.PropsSI('D','P',P1,'H',hp1s,'Water')  # Density of primary [kg/m³]
    rhos1 = CP.PropsSI('D','P',P1,'H',hs1s,'Water')  # Density of secondary [kg/m³]
    gammap = CP.PropsSI('CPMASS','P',P1,'H',hp1s,'Water') / CP.PropsSI('CVMASS','P',P1,'H',hp1s,'Water') # Specific heat ratio of primary
    gammas = CP.PropsSI('CPMASS','P',P1,'H',hs1s,'Water') / CP.PropsSI('CVMASS','P',P1,'H',hs1s,'Water') # Specific heat ratio of secondary

    Mp1 = math.sqrt(eta_n * (2 / (gammap - 1)) * ((Pp0 / P1) ** ((gammap - 1) / gammap) - 1))  # Mach number of primary flow
    Ms1 = math.sqrt(eta_n * (2 / (gammas - 1)) * ((Ps0 / P1) ** ((gammas - 1) / gammas) - 1))  # Mach number of secondary flow
    

    P2 = P1  # Pressure at mixing section (assumed constant)
    eta_m = 0.95  # Mixing efficiency
    m_dot_s = 297.73*1000/3600  # Mass flow rate of secondary [kg/s]

    Q4 = -1  # Initialize Q4 to a value greater than 1 to enter the loop
    #m_dot_p = mdot_steam_remain  # Initial mass flow rate of primary [kg/s] = 27.78 kg/s
    m_dot_p = mdot_steam_remain

    while Q4 == -1:

        omega = m_dot_s / m_dot_p  # entrainment ratio

        u2 = eta_m * ((up1 + omega * us1) / (1 + omega))  # Mixed flow velocity
        h2 = ((omega * hs1s + hp1s) / (1 + omega)) - ((u2 ** 2) / 2)  # Enthalpy of mixed flow

        Mp1_critical = math.sqrt((Mp1 ** 2) * (gammap + 1) / (2 + (gammap - 1) * (Mp1 ** 2)))  # Critical Mach number of primary flow
        Ms1_critical = math.sqrt((Ms1 ** 2) * (gammas + 1) / (2 + (gammas - 1) * (Ms1 ** 2)))  # Critical Mach number of secondary flow

        Tp = CP.PropsSI('T','P',P2,'H',hp1s,'Water')  # Temperature of mixed flow [K]
        Ts = CP.PropsSI('T','P',P2,'H',hs1s,'Water')  # Temperature of mixed flow [K]

        M2p_critical = (Mp1_critical + omega * Ms1_critical * math.sqrt(Ts / Tp)) / ((1 + omega) * (1 + omega * (Ts / Tp)))  # Critical Mach number of mixed flow

        gamma2 = CP.PropsSI('CPMASS','P',P2,'H',h2,'Water') / CP.PropsSI('CVMASS','P',P1,'H',h2,'Water')  # Specific heat ratio of mixed flow

        M2 = M2p_critical * math.sqrt(-2 / ((M2p_critical ** 2) * (gamma2 -1) - (gamma2 + 1)))  # Mach number of mixed flow
        rho2 = CP.PropsSI('D','P',P2,'H',h2,'Water')  # Density of mixed flow [kg/m³]
        c2 = math.sqrt(gamma2 * P2 / rho2)  # Speed of sound in mixed flow [m/s]

        c3 = c2  # Initial guess for speed of sound in mixed flow
        u3_new = c3 * math.sqrt(((u2 / c2) ** 2+(2 / (gamma2 - 1))) / ((u2 / c2) ** 2*((2 * gamma2) / (gamma2 - 1)) - 1)) 
        tolerance = 1e-6  # Define a tolerance for convergence
        u3_old = 0
        iteration_count = 0

        # Iterate until the change in u3 is below the tolerance
        while abs(u3_new - u3_old) > tolerance:
            u3_old = u3_new
            
            # Update the mixed flow properties based on the new u3
            h3 = h2 + (u2 ** 2 - u3_old ** 2) / 2  # Enthalpy of mixed flow at the exit
            rho3 = rho2 * (u2 / u3_old)  # Density of mixed flow at the exit
            P3 = CP.PropsSI('P','H',h3,'D',rho3,'Water')  # Pressure of mixed flow at the exit
            s3 = CP.PropsSI('S','P',P3,'H',h3,'Water')  # Entropy of mixed flow at the exit
            gamma3 = CP.PropsSI('CPMASS','P',P3,'H',h3,'Water') / CP.PropsSI('CVMASS','P',P3,'H',h3,'Water')  # Specific heat ratio of mixed flow
            c3 = math.sqrt(gamma3 * P3 / rho3)  # Speed of sound in mixed flow
            gamma_c = (gamma2 + gamma3) / 2  # Average specific heat ratio
            
            # Calculate the new u3
            u3_new = c3 * math.sqrt(((u2 / c2) ** 2+(2 / (gamma_c - 1))) / ((u2 / c2) ** 2*((2 * gamma_c) / (gamma_c - 1)) - 1)) 
            iteration_count += 1

        u3 = u3_new
        

        h3 = h2 + (u2 ** 2 - u3 ** 2) / 2  # Enthalpy of mixed flow at the exit
        rho3 = rho2 * (u2 / u3)  # Density of mixed flow at the exit
        P3 = CP.PropsSI('P','H',h3,'D',rho3,'Water')  # Pressure of mixed flow at the exit
        s3 = CP.PropsSI('S','P',P3,'H',h3,'Water')  # Entropy of mixed flow at the exit
        gamma3 = CP.PropsSI('CPMASS','P',P3,'H',h3,'Water') / CP.PropsSI('CVMASS','P',P3,'H',h3,'Water')  # Specific heat ratio of mixed flow
        gamma_c = (gamma2 + gamma3) / 2

        M3 = (M2 ** 2 + (2 /(gamma_c - 1))) / (((2 * gamma_c) / (gamma_c - 1)) * (M2 ** 2) - 1)  # Mach number of mixed flow at the exit
        P3_2 = P2 * ((1 + gamma_c * M2 ** 2)/(1 + gamma_c * M3 ** 2)) #Pressure in constant area section downstream of the shock in Pa

        """
        Diffuser Section
        """

        eta_d = 0.9  # Diffuser efficiency

        h4s = h3 + (u3 ** 2) / (2 * eta_d) # Enthalpy at outlet of diffuser assuming isentropic process
        h4 = h3 + (u3 ** 2) / 2 # Enthalpy at outlet of diffuser

        s4 = s3
        P4 = CP.PropsSI('P','H',h4,'S',s4,'Water')  # Pressure at outlet of diffuser
        T4 = CP.PropsSI('T','H',h4,'S',s4,'Water')  # Temperature at outlet of diffuser
        Q4 = CP.PropsSI('Q','H',h4,'S',s4,'Water')  # Quality at outlet of diffuser

        gamma4 = CP.PropsSI('CPMASS','P',P4,'H',h4,'Water') / CP.PropsSI('CVMASS','P',P4,'H',h4,'Water')  # Specific heat ratio at outlet of diffuser
        gamma_d = (gamma3 + gamma4) / 2  # Average specific heat ratio
        

        P4_2 = P3_2*(eta_d*((gamma_d-1)/2)*M3**2+1)**(gamma_d/(gamma_d-1)) #Pressure at diffuser outlet in Pa
        

        #m_dot_p += 0.1
        
    return m_dot_p, P4, P4_2, T4, Q4, h4, (m_dot_p + m_dot_s)
    

# Call the function
m_dot_p, P4, P4_mach, T4, Q4, h4, m_total_eject = ejector_model()
print("------------Steam Ejector------------")
print("Primary Mass flow rate of Steam = {:.2f} kg/s".format(m_dot_p))
print("Primary Steam Temperature = {:.2f} K".format(Tp0))
print("Primary Steam Pressure = {:.2f} Pa".format(Pp0))
print("Primary Steam Quality = {:.2f}".format(1))
print()
print("Combined Mass flow rate of Steam = {:.2f} kg/s".format(m_total_eject))
print("Pressure at diffuser outlet = {:.2f} Pa".format(P4))
print("Pressure at diffuser outlet by Mach Number approach = {:.2f} Pa".format(P4_mach))
print("Temperature at diffuser outlet = {:.2f} K".format(T4))
print("Quality at diffuser outlet = {:.8f}".format(Q4))
print("Enthalpy at diffuser outlet = {:.2f} J/kg".format(h4))
print("------------Steam Ejector------------")
print()
print()
print()


