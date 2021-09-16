# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May  6 19:10:11 2021
@reference: [8] Weszka, A Comparative Study of Texture Measures for Terrain Classification
==============================================================================
A.3 Gray Level Difference Statistics
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - mask:     int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                0 else
    - Dx:       array with X-coordinates of vectors denoting orientation
    - Dy:       array with Y-coordinates of vectors denoting orientation
Outputs:
    - features: 1) Contrast, 2)Angular Second Moment, 3)Entropy, 4)Mean
==============================================================================
"""

import numpy as np

def glds(f, mask, dx, dy, Ng):
    
    N1, N2 = f.shape
    
    # Calculate f_d(x,y)
    f_d = np.zeros((N1,N2), np.double) 
    mask_d = np.zeros((N1,N2), np.double) 
    for x in range(N1):
        for y in range(N2):
            if (x+dx < N1) & (y+dy < N2) & (x+dx >= 0) & (y+dy >= 0):
                f_d[x,y] = abs(f[x,y] - f[x+dx,y+dy])  
                mask_d[x+dx,y+dy] = 1
    #mask_d[mask_d>1] = 1       
            
    # Calculate pd_(i)
    f_d_ravel = f_d.ravel()
    mask_d_ravel = mask_d.ravel()
    roi = f_d_ravel[mask_d_ravel.astype(bool)]
    p_d = np.histogram(roi, bins=Ng, range=(0,Ng-1))[0] # histogram of f_d in ROI
        
    return f_d, p_d
        
def glds_features(f, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1]):
    
    # 1) Labels
    labels =  ["GLDS_Homogeneity","GLDS_Contrast",
               "GLDS_ASM","GLDS_Entopy","GLDS_Mean"]
    
    # 2) Parameters
    f = f.astype(np.double)
    mask = mask.astype(np.double)
    Dx = np.array(Dx)
    Dy = np.array(Dy)
    Ng = 256    
    
    # 3) Loop over Dx, Dy values to calculate feats
    features = []  
    for ii in range(Dx.shape[0]):
        
        dx = Dx[ii]
        dy = Dy[ii]  
        
        f_d, p_d = glds(f, mask, dx, dy, Ng)
            
        feats = np.zeros(5,np.double)
        i = np.arange(Ng)
        i2 = i ** 2
        feats[0] = sum(np.divide(p_d,i2+1))
        feats[1] = sum(np.multiply(p_d, i2))
        feats[2] = sum(np.multiply(p_d,p_d))
        feats[3] = -sum(np.multiply(p_d,np.log(p_d+1e-16)))
        feats[4] = sum(np.multiply(p_d,i))
        features.append(feats) 
      
    # 4) Calculate Features: mean of feats
    features = np.array(features)
    features = features.mean(axis=0)  
        
    return features, labels