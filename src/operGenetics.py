import numpy as np
from scipy.stats import qmc
from src import geoBlade as gb
from src import aeroPropeller as ap
from src import electricFun as ef
import os
import json
import math

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):  
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
    
def initialPopulation(fileGen, N, Omega):
    D = len(Omega)
    Dh = 0
    var = []
    for j in range(D):
        if Omega[j,0]!=Omega[j,1]:
            Dh += 1
            var.append(j)
    sampler = qmc.LatinHypercube(d=Dh)
    sample = sampler.random(n=N)
    LHx = qmc.scale(sample, Omega[var,0], Omega[var,1])
    Pg = np.zeros((N, D))
    idg = []
    for i in range(N):
        idg.append(fileGen+'/blade_'+str(i))
        k = 0
        for j in range(D):
            if (j in var) == True:
                Pg[i,j] = LHx[i,k]
                k += 1
            else:
                Pg[i,j] = Omega[j,0]
    Pg[:,0] = np.round(Pg[:,0])
    return Pg, idg


def objectiveFunction(N, Pg, nameBlade, flow, paramBEMT, mass, motor):
    global GLOBAL_MODEL
    model = GLOBAL_MODEL
    
    nu = flow['nu']
    rho = flow['rho']
    Vaxial = flow['V_axial']
    Vtrans = flow['V_trans']
    mass_eng = mass['m_engine'] * 2
    rho_prop = mass['rho_prop']
    Vinf = np.sqrt(Vaxial**2 + Vtrans**2)
    alphaShaft = np.degrees(np.arctan2(Vtrans, Vaxial))
    
    W_elec_u_g = np.array([])
    W_elec_l_g = np.array([])
    mass_prop_g = np.array([])
    mass_bat_g = np.array([])
    mass_total_g = np.array([])
    T_g = np.array([])
    W_g = np.array([])
    T_u_g = np.array([])
    T_l_g = np.array([])
    Q_u_g = np.array([])
    Q_l_g = np.array([])
    FM_u_g = np.array([])
    FM_l_g = np.array([])
    for i in range(N):
        os.makedirs(nameBlade[i], exist_ok=True)
        dS, vol_up, vol_low = gb.paramBlade(Pg[i], nameBlade[i], paramBEMT['rR_min'],
                                        paramBEMT['rR_max'], paramBEMT['sect_body'], 
                                        paramBEMT['sect_tip'])
        mass_prop_up = vol_up * rho_prop
        mass_prop_low = vol_low * rho_prop
        
        CY_batch_up, CX_batch_up = ap.reconstruct_aero_fields(model, dS[:,33])
        CY_batch_low, CX_batch_low = ap.reconstruct_aero_fields(model, dS[:,37])
        aeroResults = ap.TWQCoaxial_Oblique_Vectorized_Geo(Pg[i][14], Pg[i][27], Pg[i][0], Pg[i][1], 
                                                           dS[:,27], dS[:,28], dS[:,29], dS[:,30], dS[:,31], 
                                                           CY_batch_up, CX_batch_up, CY_batch_low, CX_batch_low, 
                                                           Pg[i][1]*flow['Z_d'], rho, nu, 
                                                           V_inf=Vinf, alpha_shaft_deg=alphaShaft)
        
        T, W, _, _, _, Cw, f_wake, T_u, T_l, _, _, Q_u, Q_l, FM_u, FM_l, _, _, alpha_u, alpha_l, Re_u, Re_l, _, _, _, _ = aeroResults
        
        if math.isnan(T):
            W_res, T_res, Q_u_res, Q_l_res = 100000, 100000, 100000, 100000
            W_elec_u = 100000
            W_elec_l = 100000
            mass_bat = 100000
        else:
            W_res, T_res, Q_u_res, Q_l_res = W, T, Q_u, Q_l
            elecResults_u = ef.electricPow(Pg[i][14], Q_u_res, motor)
            elecResults_l = ef.electricPow(Pg[i][27], Q_l_res, motor)
            W_elec_u = elecResults_u['battery_power_W']
            W_elec_l = elecResults_l['battery_power_W']
            mass_bat = ef.battery_mass(W_elec_u, W_elec_l, mass['autonomy'], mass['gamma_bat'])
        
        mass_total = mass_eng + mass_bat + mass_prop_up + mass_prop_low
        
        info_prop = {
            'nameCase': nameBlade[i], 
            'Total Thrust': float(T_res), 
            'Thrust (upper)': float(T_u),
            'Thrust (lower)': float(T_l), 
            'Total Power': float(W_res), 
            'Torque (upper)': float(Q_u_res),
            'Torque (lower)': float(Q_l_res), 
            'FM (upper)': float(FM_u), 
            'FM (lower)': float(FM_l),
            'Cw': float(Cw), 
            'f_wake': float(f_wake),
            'Electric power (upper)': float(W_elec_u),
            'Electric power (lower)': float(W_elec_l),
            'Total electric power': float(W_elec_u+W_elec_l),
            'mass motors': float(mass_eng),
            'mass propeller (upper)': float(mass_prop_up),
            'mass propeller (lower)': float(mass_prop_low),
            'mass batteries': float(mass_bat),
            'Total mass': float(mass_total),
            'V_axial': flow['V_axial'], 
            'V_trans': flow['V_trans'],
            'design_parameters': {
                'B': int(Pg[i][0]), 
                'd': float(Pg[i][1]),
                'c_d_r_up': float(Pg[i][2]), 
                'c_d_m_up': float(Pg[i][3]), 
                'c_d_t_up': float(Pg[i][4]),
                'r_cdm_up': float(Pg[i][5]), 
                'yt_r_up': float(Pg[i][6]), 
                'yt_m_up': float(Pg[i][7]),
                'yt_t_up': float(Pg[i][8]), 
                'r_yt_up': float(Pg[i][9]), 
                'phi_r_up': float(Pg[i][10]), 
                'phi_m_up': float(Pg[i][11]), 
                'phi_t_up': float(Pg[i][12]),
                'r_phim_up': float(Pg[i][13]),
                'n_min_up': float(Pg[i][14]),
                'c_d_r_low': float(Pg[i][15]), 
                'c_d_m_low': float(Pg[i][16]), 
                'c_d_t_low': float(Pg[i][17]),
                'r_cdm_low': float(Pg[i][18]), 
                'yt_r_low': float(Pg[i][19]), 
                'yt_m_low': float(Pg[i][20]),
                'yt_t_low': float(Pg[i][21]), 
                'r_yt_low': float(Pg[i][22]),
                'phi_r_low': float(Pg[i][23]),
                'phi_m_low': float(Pg[i][24]), 
                'phi_t_low': float(Pg[i][25]), 
                'r_phim_low': float(Pg[i][26]),
                'n_min_low': float(Pg[i][27]),
            },
            'info_sections': {
                'r_R': dS[:,26].tolist(), 
                'r': dS[:,27].tolist(), 
                'c_up': dS[:,28].tolist(),
                'c_low': dS[:,29].tolist(),
                'phi_up': dS[:,30].tolist(), 
                'alpha_up': alpha_u.tolist(), 
                'phi_low': dS[:,31].tolist(),
                'alpha_low': alpha_l.tolist(), 
                'xt_up': dS[:,32].tolist(), 
                'yt_up': dS[:,33].tolist(),
                'xc_up': dS[:,34].tolist(), 
                'yc_up': dS[:,35].tolist(),
                'xt_low': dS[:,36].tolist(), 
                'yt_low': dS[:,37].tolist(),
                'xc_low': dS[:,38].tolist(), 
                'yc_low': dS[:,39].tolist(),
                'Re_u': Re_u.tolist(), 
                'Re_l': Re_l.tolist()
            }
        }
        
        with open(nameBlade[i] + '/info_prop.json', 'w') as outfile:
            json.dump(info_prop, outfile, indent=4, cls=NumpyEncoder)
            
        W_elec_u_g = np.append(W_elec_u_g, W_elec_u)
        W_elec_l_g = np.append(W_elec_l_g, W_elec_l)
        mass_prop_g = np.append(mass_prop_g, mass_prop_up+mass_prop_low)
        mass_bat_g = np.append(mass_bat_g, mass_bat)
        mass_total_g = np.append(mass_total_g, mass_total)
        T_g = np.append(T_g, T_res)
        W_g = np.append(W_g, W_res)
        T_u_g = np.append(T_u_g, T_u)
        T_l_g = np.append(T_l_g, T_l)
        Q_u_g = np.append(Q_u_g, Q_u_res)
        Q_l_g = np.append(Q_l_g, Q_l_res)
        FM_u_g = np.append(FM_u_g, FM_u)
        FM_l_g = np.append(FM_l_g, FM_l)
        
    TWQg = np.column_stack((T_g, W_g, Q_u_g, Q_l_g, T_u_g, T_l_g, FM_u_g, FM_l_g,
                            W_elec_u_g, W_elec_l_g, mass_prop_g, mass_bat_g, mass_total_g))
        
    return TWQg

def compute_fitness(mg, Tg, Tmin, Qg_u, Qg_l, Qtol):
    g1 = np.maximum(0, (Tmin - Tg) / Tmin)
    g2 = np.maximum(0, np.abs(Qg_u - Qg_l) - Qtol)
    cv = g1 + g2  
    feasible = cv <= 1e-12
    
    fitness = np.full_like(mg, np.inf, dtype=float)
    
    if np.any(feasible):
        fitness[feasible] = mg[feasible]
        
        if np.any(~feasible):
            worst_feasible_mg = np.max(mg[feasible])
            best_feasible_mg = np.min(mg[feasible])
            spread = worst_feasible_mg - best_feasible_mg
            if spread < 1e-6:
                spread = abs(worst_feasible_mg) * 0.1  # fallback
            
            cv_inf = cv[~feasible]
            max_cv = np.max(cv_inf)
            cv_norm = cv_inf / max_cv if max_cv > 1e-12 else np.zeros_like(cv_inf)
            
            fitness[~feasible] = worst_feasible_mg + spread + cv_norm * spread
    else:
        max_cv = np.max(cv)
        if max_cv > 1e-12:
            fitness = cv / max_cv
        else:
            fitness = np.zeros_like(mg)
    
    return fitness, cv

def newSizePop(N0, G, g, alpha=5, Nend=20):
    c = -np.log((Nend-alpha)/(N0+alpha))/G
    Ng = round((N0+alpha)*np.exp(-c*g)-alpha)
    if Ng < Nend:
        Ng = Nend
    if Ng%2 == 0:
        NE = 2
    else:
        NE = 3
    return Ng, NE

def tournament(N, fp, g, G, pt_min=0.6, pt_max=0.9):
    
    pt = pt_min + (pt_max - pt_min) * (g / G)
    
    xr = np.random.choice(np.arange(N), 2, replace=False)
    r = np.random.rand()
    
    if r < pt:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[0]
        else:
            p = xr[1]
    else:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[1]
        else:
            p = xr[0]
    return p

def SBX(Pg, N, fp, g, G, eta_c_min=5, eta_c_max=10):
    D = np.size(Pg, 1)
    C = np.empty([0, D])
    
    eta_c = eta_c_max - (eta_c_max - eta_c_min) * (1 - g / G)
    
    for i in range(int(N/2)):
        while 1 == 1:
            p1 = tournament(N, fp, g, G)
            p2 = tournament(N, fp, g, G)
            if p1 != p2:
                break
        P1 = Pg[p1]
        P2 = Pg[p2]
        C1 = np.zeros_like(P1)
        C2 = np.zeros_like(P2)
        for j in range(D):
            u = np.random.uniform()
            if u < 0.5:
                beta = (2*u)**(1/(eta_c+1))
            else:
                beta = 1/((2*(1-u))**(1/(eta_c+1)))
            C1[j] = 0.5*((1-beta)*P1[j] + (1+beta)*P2[j])
            C2[j] = 0.5*((1+beta)*P1[j] + (1-beta)*P2[j])
        C = np.row_stack((C, C1, C2))
    return C

def mutation(C, N, Omega, g, G, eta_m_min=5, eta_m_max=10):
    D = np.size(C, 1)
    Q = np.empty([0, D])
    
    pm = 1/D
    eta_m = eta_m_max - (eta_m_max - eta_m_min) * (1 - g / G)
    
    for i in range(N):
        p = C[i]
        m = np.copy(p) 
        for j in range(D):
            if np.random.rand() < pm:
                r = np.random.uniform()
                if r < 0.5:
                    delta = (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    
                m[j] = p[j] + delta * (Omega[j, 1] - Omega[j, 0])
                
        m = np.clip(m, Omega[:, 0], Omega[:, 1])
        
        m[0] = round(m[0]) 
        
        Q = np.vstack((Q, m))
    return Q

def elitism(Pg, fp, NE):
    sorted_indices = np.argsort(fp)[:NE]
    E = Pg[sorted_indices].copy()       
    return E

def updateMetrics(metrics, TWQg, fp):
    iopt = np.argmin(fp)
    metrics['T_opt'] = np.append(metrics['T_opt'], TWQg[iopt,0])
    metrics['W_opt'] = np.append(metrics['W_opt'], TWQg[iopt,1])
    metrics['Q_u_opt'] = np.append(metrics['Q_u_opt'], TWQg[iopt,2])
    metrics['Q_l_opt'] = np.append(metrics['Q_l_opt'], TWQg[iopt,3])
    metrics['T_u_opt'] = np.append(metrics['T_u_opt'], TWQg[iopt,4])
    metrics['T_l_opt'] = np.append(metrics['T_l_opt'], TWQg[iopt,5])
    metrics['FM_u_opt'] = np.append(metrics['FM_u_opt'], TWQg[iopt,6])
    metrics['FM_l_opt'] = np.append(metrics['FM_l_opt'], TWQg[iopt,7])
    metrics['W_e_u_opt'] = np.append(metrics['W_e_u_opt'], TWQg[iopt,8])
    metrics['W_e_l_opt'] = np.append(metrics['W_e_l_opt'], TWQg[iopt,9])
    metrics['m_prop_opt'] = np.append(metrics['m_prop_opt'], TWQg[iopt,10])
    metrics['m_battery_opt'] = np.append(metrics['m_battery_opt'], TWQg[iopt,11])
    metrics['m_total_opt'] = np.append(metrics['m_total_opt'], TWQg[iopt,12])
    metrics['Fitness_opt'] = np.append(metrics['Fitness_opt'], min(fp))
    return metrics

def check_convergence(W_history, threshold=0.001, patience=5):

    if len(W_history) < 2:
        return False
    
    consecutive = 0
    
    for i in range(len(W_history) - 1, 0, -1):
        curr = W_history[i]
        prev = W_history[i - 1]
        
        if prev == 0:
            variation = 0
        else:
            variation = abs(curr - prev) / abs(prev)
        
        if variation < threshold:
            consecutive += 1
        else:
            break  
    
    return consecutive >= patience
