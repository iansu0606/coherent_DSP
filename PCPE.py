import math
import numpy as np
import os
from scipy.fftpack import fft,ifft
import scipy.io as sio
from subfunction.DataNormalize import *
from subfunction.seq2bit import *
from subfunction.BERcount import *
from subfunction.SNR import *
from subfunction.SNR2BER import *
from subfunction.bit2seq import *
import mpl_scatter_density
import sys
from PCPEBPS import *
from torch.distributions import Normal
from subfunction.loadParameter import *
from subfunction.ExpTxencode import *
from subfunction.corr import *
from model.Conv1dmodel import *
from PLL import *
# from CMA import *
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
# import torch.utils.data as Data
import random
from scipy.signal import resample_poly as resample
import torch.nn.functional as F
from subfunction.scatter_plot import *
from tqdm import tqdm
from model.Conv1dreal import *
from model.GoogLeNet import *
from model.conv1dResnetModel import *
from PLL import *
simu = True
#%%#%%
eye      = 0
PCPEtaps = 2
BPStaps  = 31

# data = np.load('80kmaftermodel.npz')
# xr,xi,yr,yi = data['xr'],data['xi'],data['yr'],data['yi']

para = loadParameters(simu)
RxXIi,RxXQi = para.RxXIi, para.RxXQi
RxYIi,RxYQi = para.RxYIi, para.RxYQi
TxXIi,TxXQi = para.TxXIi, para.TxXQi
TxYIi,TxYQi = para.TxYIi, para.TxYQi
downsample_num = para.resamplenumber
upsample_num = para.upsamplenum
PAM_order = para.pamorder
RxXIi,RxXQi = DataNormalize(RxXIi,RxXQi,PAM_order)
RxYIi,RxYQi = DataNormalize(RxYIi,RxYQi,PAM_order)

TxXIi,TxXQi = DataNormalize(TxXIi,TxXQi,PAM_order)
TxYIi,TxYQi = DataNormalize(TxYIi,TxYQi,PAM_order)
    
RxXI = resample(RxXIi,up = upsample_num,down=1)
RxYI = resample(RxYIi,up = upsample_num,down=1)
RxXQ = resample(RxXQi,up = upsample_num,down=1)
RxYQ = resample(RxYQi,up = upsample_num,down=1)
#%%
for eye in range(para.resamplenumber):
    RxXI = RxXIi[eye::downsample_num]
    RxYI = RxYIi[eye::downsample_num]
    RxXQ = RxXQi[eye::downsample_num]
    RxYQ = RxYQi[eye::downsample_num]
    
    TxX = seq2bit(TxXIi[eye::downsample_num],TxXQi[eye::downsample_num],PAM_order)
    TxY = seq2bit(TxYIi[eye::downsample_num],TxYQi[eye::downsample_num],PAM_order)
    # TxX = seq2bit(resample(TxXIi[eye:],up = 1,down=downsample_num),resample(TxXQi[eye:],up = 1,down=downsample_num),PAM_order)
    # TxY = seq2bit(resample(TxYIi[eye:],up = 1,down=downsample_num),resample(TxYQi[eye:],up = 1,down=downsample_num),PAM_order)
    Tx_real = bit2seq(TxX,PAM_order)[:,0]
    Tx_img  = bit2seq(TxX,PAM_order)[:,1]
    Ty_real = bit2seq(TxY,PAM_order)[:,0]
    Ty_img  = bit2seq(TxY,PAM_order)[:,1]
    xr,xi,yr,yi = RxXI,RxXQ,RxYI,RxYQ
    scatter_plot(xr,xi,'X-polarization original data',xlabel = 'real',ylabel = 'imag')
    scatter_plot(yr,yi,'Y-polarization original data',xlabel = 'real',ylabel = 'imag')
    inputx = np.vstack((xr,xi)).T
    inputy = np.vstack((yr,yi)).T
    datar,datai   = PCPE16QAM(inputx,PCPEtaps)
    scatter_plot(datar,datai,'x-pol PCPE',xlabel = 'real',ylabel = 'imag')
    outr,outi = MstageBPS(datar,datai,11,2,BPStaps)
    scatter_plot(outr,outi,'eye=%d output'%(eye),xlabel = 'real',ylabel = 'imag')
    datar,datai   = PCPE16QAM(inputy,PCPEtaps)
    scatter_plot(datar,datai,'y-pol PCPE',xlabel = 'real',ylabel = 'imag')
    outr,outi = MstageBPS(datar,datai,11,2,BPStaps)
    scatter_plot(outr,outi,'eye=%d output'%(eye),xlabel = 'real',ylabel = 'imag')
#%%

