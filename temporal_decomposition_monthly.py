#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lollier

Keerthi detrending 
Contrasted Contribution of Intraseasonal Time Scales to Surface Chlorophyll Variations in a Bloom and an Oligotrophic Regime
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JC015701

"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                               Temporal decomposition                                  | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import xarray as xr
from tqdm import tqdm
from scipy import signal

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                       Implement a derived X11 method for detrending                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #




class temporal_decomposition_monthly():
    """
    Create an object from a temporal series with Trend, Seasonality and IntraSeasonnal components.
    The frequency sample needs to be monthly. Any kind of shape is allowed but the time component needs to be the first dimension.
    If the dataset is sufficiently long, it's better to exclude the first and last year from further analysis after decomposition,
    the method is less robust at the boundaries because of intrinsic convolutions properties. 
    
    TODO:
        - Separate T into Trend and Interannual
        - Verify with Keerthi the use of S3 or S2 at step 5/6
        - Linear interpolation for now, consider adding cubic or other methods
    """
    
    def __init__(self,time_serie):
        """
        Parameters
        ----------
        time_serie : np.array 
            data that will be decompose, time axis needs to be the first dim
            
        -------
        """
        self.time_serie=time_serie
        
        self.ts_shape=self.time_serie.shape
        self.len=time_serie.shape[0]
        
        self.T=np.empty(self.ts_shape)
        self.S=np.empty(self.ts_shape)
        self.I=np.empty(self.ts_shape)
        
        self.build_decomposition()
        
    def build_decomposition(self):
        
        #loop over all ts
        for ts_index,_ in (np.ndenumerate(self.time_serie[0])):
            
            #building the slice for the right index
            ts_slice=[]
            for dim,size in enumerate(self.ts_shape):
                if dim==0: #temporal axis
                    ts_slice.append(slice(0,size))
                else :
                    ts_slice.append(ts_index[dim-1])
            
            ts_slice=tuple(ts_slice)
            if np.isnan(self.time_serie[ts_slice]).sum()==self.len:
                
                self.T[ts_slice]=self.time_serie[ts_slice]
                self.S[ts_slice]=self.time_serie[ts_slice]
                self.I[ts_slice]=self.time_serie[ts_slice]
                
            else :
                t,s,i=self.decompose(self.interpolate(self.time_serie[ts_slice]))
                self.T[ts_slice]=t*(1+(np.isnan(self.time_serie[ts_slice])*self.time_serie[ts_slice]))
                self.S[ts_slice]=s*(1+(np.isnan(self.time_serie[ts_slice])*self.time_serie[ts_slice]))
                self.I[ts_slice]=i*(1+(np.isnan(self.time_serie[ts_slice])*self.time_serie[ts_slice]))
            
    #linear interoplation
    def interpolate(self, ts):
        axis=np.arange(0,self.len)
        return np.interp(axis, axis[~np.isnan(ts)], ts[~np.isnan(ts)])

    def moving_average(self, x):

        
        w=np.ones(13)*2
        w[0],w[-1]=1,1
        
        #padding same instead of duplication of series' boundaries.
        res=np.convolve(x, w, 'same') / (12*2)

        return res

    def seasonal_running_mean(self, x):
        res=np.zeros((x.shape[0]))

        #no duplications anymore, the seasonal running mean for the boundaries is only computed with the available years.
        for i in range(12,x.shape[0]-12):
            res[i]=0.25*(x[i-12]+(2*x[i])+x[i+12])
        
        for i in range(0,12):
            res[i]=((2*x[i])+x[i+12])/3
            
        for i in range(x.shape[0]-12,x.shape[0]):
            res[i]=(x[i-12]+(2*x[i]))/3

        return res
            
        
    def henderson_filter_13(self, x):

        H_wts=np.array([-0.0193, -0.0279,  0.000,  0.0655,  0.1474, 0.2143, 0.2401,  0.2144,  0.1474,  0.0655,  0.000, -0.0279, -0.0193])
        
        res=np.convolve(x, H_wts, 'same')#/H_wts.sum() ? la somme des H_wts fait pas exactement 1 probablement du à un pb sur la gestion des float
        return res
    
    def band_pass(self, x):
        
        # filt=np.zeros(x.shape)
        # filt[:12]=np.ones((12)) #12*8=88
        
        # fft_x=sc.fft.fft(x)
        
        # return sc.fft.ifft(fft_x*filt)    
        Te=30 
        # Fréquence d'échantillonnage
        fe = 1/Te  # Hz
        
        # Fréquence de nyquist
        f_nyq = fe / 2.  # Hz
        
        # Fréquence de coupure
        fc = 1/88  # Hz
        
        # Préparation du filtre de Butterworth en passe bande
        b, a = signal.butter(4, fc/f_nyq, 'high', analog=False)
        
        return signal.filtfilt(b, a, x)

    def decompose(self, ts):
        
        #1        
        T1=self.moving_average(ts)
        Z=ts-T1
        
        #2
        S0=self.seasonal_running_mean(Z)
        MS0=self.moving_average(S0)
        S1=S0-MS0
        Z1=ts-S1
        
        #3
        T2=self.henderson_filter_13(Z1)
        Z2=ts-T2
        
        #4
        S2=self.seasonal_running_mean(Z2)
        MS2=self.moving_average(S2)
        S3=S2-MS2
        Z3=ts-S3
        
        
        #5
        I1=self.band_pass(S2) #keert utilise S2 mais c'est peut être S3 ? ça change un peu I et S mais c'est très léger

        #6
        T=self.henderson_filter_13(Z3)

        #7
        S=S2-I1 #see above
        
        #8
        I=I1+(ts-S-T)
        
        return T,S,I


if __name__=="__main__": 
    
    print('')














