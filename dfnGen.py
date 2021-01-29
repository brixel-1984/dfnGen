# -*- coding: utf-8 -*-

# _____________________
#
#
# _____________________

import os, sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import scipy as sp 
import sympy as sym
from sympy.solvers import solve
import math

def randomNumberGenerator(distr,*params,length):
    """Generates random numbers based on several probability distributions.
    
    Arguments:
    distr -- Statistical distribution (uniform, normal, lognormal, etc.)
    *params -- Parameters of statistical distributions
    length -- Number of values generated

    Returns:
    RndNum -- Generated random numbers
    """

    if distr == 'Uniform':
        loVal = params[0]
        hiVal = params[1]
        RndNum = np.random.uniform(low=loVal,high=hiVal,size=length)

    elif distr == 'Normal':
        Mean = params[0]
        StdDev = params[1]
        RndNum = np.random.normal(loc=Mean,scale=StdDev,size=length)

    elif distr == 'Exponential':
        Mean = params
        RndNum = np.random.exponential(scale=Mean,size=length)
    
    elif distr == 'Lognormal':
        Mean = params[0]
        StdDev = params[1]
        Mu = np.log10((Mean**2)/np.sqrt(StdDev**2+Mean**2))
        Sigma = np.sqrt(np.log10(StdDev**2/Mean**2+1))
        RndNum = np.random.lognormal(mean=Mu,sigma=Sigma,size=length)

    elif distr == 'Gamma':
        Mean = params[0]
        StdDev = params[1]
        A = Mean**2/StdDev**2
        B = StdDev**2/Mean
        RndNum = np.random.gamma(shape=A,scale=B,size=length)

    return RndNum

def fractalDfnGen(a,D,alpha,lmin,L):
    """Generates 2-D fractal DFNs

    Arguments:
    a -- Density parameter
    D -- Fractal dimension
    alpha -- Power-law exponent
    lmin -- Minimum fracture length
    L -- Maximum fracture length

    Returns:
    rho -- fracture density
    p21 -- fracture intensity
    p -- percolation parameter
    w -- number of intersection/fracture
    """
    
    Nbin = 50
    
    # Domain 
    Xmin, Xmax = 0, L
    Ymin, Ymax = 0, L
    Width, Height = L, L 

    # Length (powerlaw)
    MaxLen = np.sqrt((Xmax-Xmin)**2+(Ymax-Ymin)**2)*100
    Interval = np.logspace(np.log10(lmin), np.log10(MaxLen), Nbin+1)
    binMidLen = np.zeros(Nbin)
    powerLen = []

    for i in range(0,Nbin):
        binLeft   = Interval[i]
        binRight  = Interval[i+1]
        binMidLen[i] = 10**((np.log10(binLeft)+np.log10(binRight))/2)
        density   = alpha*(L**D)*binMidLen[i]**(-a)
        binFracNo = int(np.round(density*(binRight-binLeft)))
        binFracLen= randomNumberGenerator('Uniform',binLeft,binRight,length=binFracNo)      
        powerLen  = np.append(powerLen, binFracLen)

    TotFracNo = len(powerLen)
    powerLen  = powerLen[np.random.permutation(TotFracNo)]

    # Orientation
    #fracOrien = np.random.uniform(low=0,high=np.pi,size=TotFracNo)
    fracOrien = randomNumberGenerator('Uniform',0,np.pi,length=TotFracNo)

    # Barycentres
    if D == 2:
        centX = randomNumberGenerator('Uniform',Xmin,Xmax,length=TotFracNo)
        centY = randomNumberGenerator('Uniform',Ymin,Ymax,length=TotFracNo)

        BaryCenter = np.vstack((centX,centY))
    
    else:
        # Probability set
        sp1 = 0.2
        sp2 = 0.3
        p3, p4 = sym.symbols('p3,p4')
        eq1 = sym.Eq(sp1+sp2+p3+p4,1)
        eq2 = sym.Eq(sp1**2+sp2**2+p3**2+p4**2,0.5**2)

        sp3, sp4 = sym.solve([eq1,eq2],[p3,p4])
        Probset = [sp1,sp2,sp3[0],sp4[0]]

        # Cascade iteration
        CascIter = math.floor(np.log(L/lmin)/np.log(2))

        # Probability field
        for k in range(0,CascIter):
            print(k)

            # Current field
            Curfield = np.zeros(2**k)

            if k == 1:
                # Permute probability set
                PermProbSet = Probset[np.random.permutation(4)]

                for i in range(0,1):
                    for j in range(0,1):
                        Curfield[i,j] = PermProbSet[(i-1)*2+j]
            else:
                PreGridNum = 2**(k)

                for m in range(0,PreGridNum):
                    for n in range(0,PreGridNum):
                        PermProbSet = Probset[np.random.permutation(4)]

                        for i in range(0,1):
                            for j in range(0,1):
                                Curfield[2*(m-1)+i,2*(n-1)+j] = PermProbSet[(i-1)*2+j]*Prefield[m,n]
            Prefield = Curfield
        Cascfield = Curfield

        # Create Barycenters
        
    # Generate fracture geometry
    EndPt1X = BaryCenter[0]+powerLen*np.cos(fracOrien)/2
    EndPt2X = BaryCenter[0]-powerLen*np.cos(fracOrien)/2
    EndPt1Y = BaryCenter[1]+powerLen*np.sin(fracOrien)/2
    EndPt2Y = BaryCenter[1]-powerLen*np.sin(fracOrien)/2

    # Trim traces
    for j in range(0,TotFracNo):
        Pt1X = EndPt1X[j]
        Pt1Y = EndPt1Y[j]
        Pt2X = EndPt2X[j]
        Pt2Y = EndPt2Y[j]

        # Pt1X > Xmax (1st point outside)
        if Pt1X > Xmax:
            Pt1Y = Pt2Y-(Pt2Y-Pt1Y)/(Pt2X-Pt1X)*(Pt2X-Xmax)
            Pt1X = Xmax
        
        elif Pt1X < Xmin:
            Pt1Y = Pt2Y-(Pt2Y-Pt1Y)/(Pt2X-Pt1X)*(Pt2X-Xmin)
            Pt1X = Xmin

        elif Pt1Y > Ymax:
            Pt1X = Pt2X-(Pt2X-Pt1X)/(Pt2Y-Pt1Y)*(Pt2Y-Ymax)
            Pt1Y = Ymax
            
        elif Pt1Y < Ymin:
            Pt1X = Pt2X-(Pt2X-Pt1X)/(Pt2Y-Pt1Y)*(Pt2Y-Ymin)
            Pt1Y = Ymin

        # Pt2X > Xmax (2nd point outside)
        if Pt2X > Xmax:
            Pt2Y = Pt1Y-(Pt1Y-Pt2Y)/(Pt1X-Pt2X)*(Pt1X-Xmax)
            Pt2X = Xmax
        
        elif Pt2X < Xmin:
            Pt2Y = Pt1Y-(Pt1Y-Pt2Y)/(Pt1X-Pt2X)*(Pt1X-Xmin)
            Pt2X = Xmin

        elif Pt2Y > Ymax:
            Pt2X = Pt1X-(Pt1X-Pt2X)/(Pt1Y-Pt2Y)*(Pt1Y-Xmax)
            Pt2Y = Ymax

        elif Pt2Y < Ymin:
            Pt2X = Pt1X-(Pt1X-Pt2X)/(Pt1Y-Pt2Y)*(Pt1Y-Xmin)
            Pt2Y = Ymin
        
        EndPt1X[j] = Pt1X
        EndPt1Y[j] = Pt1Y
        EndPt2X[j] = Pt2X
        EndPt2Y[j] = Pt2Y

    PolylineX = np.zeros((TotFracNo,4))
    PolylineY = np.zeros((TotFracNo,4))

    for j in range(0,TotFracNo):
        Pt1X = EndPt1X[j]
        Pt1Y = EndPt1Y[j]
        Pt2X = EndPt2X[j]
        Pt2Y = EndPt2Y[j]
        fracLen = np.sqrt((Pt1X-Pt2X)**2+(Pt1Y-Pt2Y)**2)
        a = 5e-3
        X1, Y1 = Pt1X, Pt1Y
        X2 = (Pt1X+Pt2X)/2+a/2*np.sin(fracOrien[j])
        Y2 = (Pt1Y+Pt2Y)/2-a/2*np.cos(fracOrien[j])
        X3, Y3 = Pt2X, Pt2Y
        X4 = (Pt1X+Pt2X)/2-a/2*np.sin(fracOrien[j])
        Y4 = (Pt1Y+Pt2Y)/2+a/2*np.cos(fracOrien[j])
        PolylineX[j] = [X1,X2,X3,X4]
        PolylineY[j] = [Y1,Y2,Y3,Y4]

    ## Colorcode traces (orientation)
    #colorparams = fracOrien
    #colormap = cm.winter
    #normalize = mcolors.Normalize(vmin=np.min(colorparams), #vmax=np.max(colorparams))
     
    fig, ax = plt.subplots(figsize=(5,5))
    for j in range(0,TotFracNo):
        #color = colormap(normalize(fracOrien[j]))
        color = '#1f77b4'
        ax.plot(np.append(PolylineX[j],PolylineX[j,0]),np.append(PolylineY[j],PolylineY[j,0]),c=color,lw=0.8)
    ax.set_aspect('equal')
    ax.set_xlim(Xmin,Xmax)
    ax.set_ylim(Ymin,Ymax)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('dfn1.png',dpi=500)

    rho = TotFracNo/(L**2)
    print(r'Density =', rho)

    return rho

L = 10
a = 3
alpha = 1.5
D = 2
lmin = L/50
lmax = L*50

fractalDfnGen(a,D,alpha,lmin,L)
