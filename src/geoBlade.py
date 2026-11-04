import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simpson
from math import factorial
from src import geoAirfoil as ga

def BezierN(p, n):
    nc = 20
    t = np.linspace(0, 1, nc)
    B = 0
    for i in range(n+1):
        B += p[i]*t**i*(1-t)**(n-i)*(factorial(n)/(factorial(i)*factorial(n-i)))
    return B

def chordCurve(p0, p1, p2, p3, d, r, nameBlade):
    R = d/2    
    pR_r = np.array([r[0], 0.5*(p3*R+r[0]), p3*R])
    pc_r = np.array([p0*d, p1*d, p1*d])
    pR_t = np.array([p3*R, (r[-1]+2*p3*R)/3, (2*r[-1]+p3*R)/3, r[-1]])
    pc_t = np.array([p1*d, p1*d, (p2+5*p1)*d/6, p2*d])
    BR_r = BezierN(pR_r, 2)
    Bc_r = BezierN(pc_r, 2)
    BR_t = BezierN(pR_t, 3)
    Bc_t = BezierN(pc_t, 3)
    BR = np.append(BR_r[:-1], BR_t)
    Bc = np.append(Bc_r[:-1], Bc_t)
    FC = InterpolatedUnivariateSpline(BR, Bc, k=4)
    c = FC(r)
    return c

def phiCurve(p0, p1, p2, p3, r, R, nameBlade):
    pR_r = np.array([r[0], (p3*R+2*r[0])/3, (2*p3*R+r[0])/3, p3*R])    
    pphi_r = np.array([p0, 0.8*p0, 1.1*p1, p1])    
    pR_t = np.array([p3*R, 0.5*(r[-1]+p3*R), r[-1]])
    pphi_t = np.array([p1, p1, p2]) 
    
    BR_r = BezierN(pR_r, 3)
    Bphi_r = BezierN(pphi_r, 3)
    BR_t = BezierN(pR_t, 2)
    Bphi_t = BezierN(pphi_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Bphi = np.append(Bphi_r[:-1], Bphi_t)
    Fphi = InterpolatedUnivariateSpline(BR, Bphi, k=4)
    phi = Fphi(r)
    
    return phi


def airfoilThickness(p0, p1, p2, p3, r, R, nameBlade):    
    pR_r = np.array([r[0], 0.5*(p3*R+r[0]), p3*R])
    pyt_r = np.array([p0, p1, p1])
    pR_t = np.array([p3*R, (r[-1]+R*p3)*0.5, r[-1]])
    pyt_t = np.array([p1, p1, p2])
    BR_r = BezierN(pR_r, 2)
    Byt_r = BezierN(pyt_r, 2)
    BR_t = BezierN(pR_t, 2)
    Byt_t = BezierN(pyt_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Byt = np.append(Byt_r[:-1], Byt_t)
    Fyt = InterpolatedUnivariateSpline(BR, Byt, k=4)
    ytMax = Fyt(r)

    return ytMax

def airfoil_area(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
        
    # Fórmula de la Lazada (Shoelace) usando operaciones vectorizadas de NumPy.
    # np.roll(y, 1) desplaza el array para emparejar (x_i, y_{i+1}) y (y_i, x_{i+1}),
    # incluyendo el salto implícito del último elemento al primero.
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def paramBlade(Pi, nameBlade, rR_min, rR_max, sect_body, sect_tip):
    r_R1 = np.linspace(rR_min, 0.75, sect_body)
    r_R2 = np.linspace(0.776, rR_max, sect_tip)
    r_R = np.append(r_R1, r_R2)
    R = Pi[1]/2
    r = r_R*R
    
    c_up = chordCurve(Pi[2], Pi[3], Pi[4], Pi[5], Pi[1], r, nameBlade)
    c_low = chordCurve(Pi[15], Pi[16], Pi[17], Pi[18], Pi[1], r, nameBlade)
    ytmax_up = airfoilThickness(Pi[6], Pi[7], Pi[8], Pi[9], r, R, nameBlade)
    ytmax_low = airfoilThickness(Pi[19], Pi[20], Pi[21], Pi[22], r, R, nameBlade)
    phi_up = phiCurve(Pi[10], Pi[11], Pi[12], Pi[13], r, R, nameBlade)
    phi_low = phiCurve(Pi[23], Pi[24], Pi[25], Pi[26], r, R, nameBlade)
        
    A_up = np.empty([0,13])
    xt_up = np.array([])
    yt_up = np.array([])
    xc_up = np.array([])
    yc_up = np.array([])
    for s in range(len(r)):
        As = ga.creatorAirfoil(ytmax_up[s])
        A_up = np.row_stack((A_up, As))
        X, YU, YL = ga.cstN5(As[:-1], TE=As[-1])
        _, YT, YC = ga.cst_ST_SC(As[:-1], TE=As[-1])
        xt_up = np.append(xt_up, X[np.argmax(YT)])
        yt_up = np.append(yt_up, max(YT))
        xc_up = np.append(xc_up, X[np.argmax(YC)])
        yc_up = np.append(yc_up, max(YC))
    
    np.save(nameBlade+'/A_up.npy', A_up)
    
    A_low = np.empty([0,13])
    xt_low = np.array([])
    yt_low = np.array([])
    xc_low = np.array([])
    yc_low = np.array([])
    for s in range(len(r)):
        As = ga.creatorAirfoil(ytmax_low[s])
        A_low = np.row_stack((A_low, As))
        X, YU, YL = ga.cstN5(As[:-1], TE=As[-1])
        _, YT, YC = ga.cst_ST_SC(As[:-1], TE=As[-1])
        xt_low = np.append(xt_low, X[np.argmax(YT)])
        yt_low = np.append(yt_low, max(YT))
        xc_low = np.append(xc_low, X[np.argmax(YC)])
        yc_low = np.append(yc_low, max(YC))
    
    np.save(nameBlade+'/A_low.npy', A_low)
    
    dS = np.column_stack((A_up, A_low))
    dS = np.column_stack((dS, r_R))
    dS = np.column_stack((dS, r))
    dS = np.column_stack((dS, c_up))
    dS = np.column_stack((dS, c_low))
    dS = np.column_stack((dS, phi_up))
    dS = np.column_stack((dS, phi_low))
    dS = np.column_stack((dS, xt_up))
    dS = np.column_stack((dS, yt_up))
    dS = np.column_stack((dS, xc_up))
    dS = np.column_stack((dS, yc_up))
    dS = np.column_stack((dS, xt_low))
    dS = np.column_stack((dS, yt_low))
    dS = np.column_stack((dS, xc_low))
    dS = np.column_stack((dS, yc_low))
    
    sections_scale_up = []
    for s in range(len(r)):
        X, YU, YL = ga.cstN5(A_up[s,:12], TE=A_up[s,-1])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*c_up[s]
        Yscale = Ya*c_up[s]
        sections_scale_up.append((Xscale, Yscale))
        
    areas_up = [airfoil_area(x, y) for x, y in sections_scale_up]
    volume_blade_up = simpson(areas_up, x=r)
    volume_prop_up = volume_blade_up * Pi[0]
    
    sections_scale_low = []
    for s in range(len(r)):
        X, YU, YL = ga.cstN5(A_low[s,:12], TE=A_low[s,-1])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*c_low[s]
        Yscale = Ya*c_low[s]
        sections_scale_low.append((Xscale, Yscale))
        
    areas_low = [airfoil_area(x, y) for x, y in sections_scale_low]
    volume_blade_low = simpson(areas_low, x=r)
    volume_prop_low = volume_blade_low * Pi[0]
        
    return dS, volume_prop_up, volume_prop_low

