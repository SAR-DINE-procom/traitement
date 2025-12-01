import numpy as np 
global k 
global T_0 
global c 

k  = 1.38*10e-23
T_0 = 290 
c = 3 * 10e8

def get_lbd(f): 
    return c / f
def log_2_lin(log): 
    return 10 ** (log / 10)

def lin_2_log(lin): 
    return 10 * np.log10(lin)

def estimate_chirp_duration(SNR: float, R: float, F_n: float, P_crete: float, G: float, lambd: float, ser: float) -> float: 
    """
        - SNR (linéaire): rapport S/B;  
        - R [m]: Distance max à la cible;   
        - F_n (linéaire): facteur de bruit du recepteur; 
        - P_crete [W]: Puissance d'émission;  
        - G (linéaire): Gain de l'antenne monostatique;  
        - lambd [m]: longueur d'onde; 
        ser [m²]: Surface équivalente radar. 
    """
    return (SNR * (4 * np.pi)**3 * R**4 * k * T_0 * F_n) / (P_crete * G ** 2 * lambd**2 * ser)


SNR = 10 
R = 5 
F_n = 3
P_crete = 5
G = 13 
f = 24e9 
ser = 1 
bw = 200e6

T_chirp = estimate_chirp_duration(log_2_lin(SNR), R, log_2_lin(F_n), 
                                  P_crete, log_2_lin(G), get_lbd(f), 
                                  ser)


print(T_chirp )
print(f"Rampe K: {bw / T_chirp}")