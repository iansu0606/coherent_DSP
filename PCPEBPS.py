import numpy as np
import math
from subfunction.seq2bit import *
from subfunction.bit2seq import *
# from subfunction.scatter_plot import *
PAM_order = 4
def PCPE16QAM(input,taps):
    rr,ri = input[:,0],input[:,1]
    datar,datai,phirecord = [],[],[]
    half = taps//2
    rr, ri = np.pad(rr,(half,half)),  np.pad(ri,(half,half))
    phikr  = np.zeros(rr.size)
    phiadj = np.zeros(rr.size)
    for i in range(half,rr.shape[0]-half):
        tmpr,tmpi = rr[i-half:i+half+1],ri[i-half:i+half+1]
        xxx       = (tmpr+1j*tmpi)**2
        tmpr      = xxx.real
        tmpi      = xxx.imag
        Ak = np.vstack((tmpr.reshape((1,-1)),tmpi.reshape((1,-1))))
        # print(Ak)
        Ck = np.matmul(Ak, Ak.T)
        # print(Ck)
        # print('CKshape:',Ck.shape)
        if i == half:
            vk    = np.array([1,0]).T
            for p in range(3):
                vk   = np.matmul(Ck,vk)
                vk  /= np.sqrt(vk[0]**2+vk[1]**2)
                phik = 1/2*math.atan(vk[1]/vk[0])-np.pi/4
                if p == 1:
                    philast = phik
        else:
            vk   = np.matmul(Ck,vk)
            vk  /= np.sqrt(vk[0]**2+vk[1]**2)
            phik = 1/2*math.atan(vk[1]/vk[0])-np.pi/4 
        # phik    += np.pi/2*(1/2+(philast-phik)/np.pi*2)
        phij     = phik+np.pi/2*np.floor(1/2+(philast-phik)/np.pi*2)
        tmp      = (rr[i] + 1j*ri[i])*np.exp(-1j*phij)#
        philast  = phij
        phirecord.append(phik)
        datar.append(tmp.real)
        datai.append(tmp.imag)
    datar = np.array(datar)
    datai = np.array(datai)
    return datar,datai

def harddecision(x):
    'Find closest symbol and return position'
    xr  = x.real
    xi  = x.imag
    out = bit2seq(seq2bit(xr,xi,PAM_order),PAM_order)
    ans = out[:,0]+1j*out[:,1]
    return ans


def BPS(datar,datai,b,taps):
    rr,ri     = datar,datai
    ita       = 1/b
    half      = taps//2
    # rr, ri =  np.pad(rr,(half,half)),  np.pad(ri,(half,half))
    phi = lambda p:ita*np.pi*((2*p-1)/4/b-1/4)#p*ita*np.pi/2#
    outr,outi = [],[]
    for i in range(half,rr.size-half):
        tmp_r = rr[i-half:i+half+1]
        tmp_i = ri[i-half:i+half+1]
        ek    = []
        lmse      = 1e10
        bestphi   = 0
        for p in range(1,b+1):
            phib  = phi(p)
            est = (tmp_r+1j*tmp_i)*np.exp(-1j*phib)
            err = est - harddecision(est)
            err = np.mean(abs(err)**2)
            ek.append(err)
            if p == 1: lmse = err
            elif err < lmse:
                lmse = err
                bestphi = phib
            # print(bestphi)
    
        bestp = int(((bestphi/ita/np.pi+1/4)*4*b+1)/2)#int(bestphi/ita*2/np.pi)-1#
        # if bestp>0 and bestp<b-1:  
        #     dkp   = ek[bestp-1] - ek[bestp]
        #     dkl   = ek[bestp+1] - ek[bestp]
        #     out   = (rr[i]+1j*ri[i])*\
        #     np.exp(-1j*(bestphi+abs(phi(bestp-1)-phi(bestp+1))/2*(dkp+dkl)/(dkp-dkl)))
        # else: 
        out   = (rr[i]+1j*ri[i])*np.exp(-1j*bestphi)    
        outr.append(out.real)
        outi.append(out.imag)
    return np.array(outr),np.array(outi)

def MstageBPS(rr,ri,b,M,taps):
    for i in range(M):
        rr,ri = BPS(rr,ri,b,taps)
        # scatter_plot(rr,ri,'BPS stage %d'%(i+1),xlabel = 'real',ylabel = 'imag')
    return rr,ri

def testPCPE16QAM(input,taps):
    rr,ri = input[:,0],input[:,1]
    datar,datai = [],[]
    half = taps//2
    for i in range(0,input.shape[0]-1,taps):
        tmpr,tmpi = rr[i:i+taps],ri[i:i+taps]
        xxx       = (tmpr+1j*tmpi)**2
        tmpr      = xxx.real
        tmpi      = xxx.imag
        Ak = np.vstack((tmpr.reshape((1,-1)),tmpi.reshape((1,-1))))
        Ck = np.matmul(Ak, Ak.T)
        if i == 0:
            vk    = np.array([1,0]).T
            philast = 0
            for _ in range(3):
                vk   = np.matmul(Ck,vk)
                vk  /= np.sqrt(vk[0]**2+vk[1]**2)
                phik = 1/2*math.atan(vk[1]/vk[0])-np.pi/4
        else:
            vk   = np.matmul(Ck,vk)
            vk  /= np.sqrt(vk[0]**2+vk[1]**2)
            phik = 1/2*math.atan(vk[1]/vk[0])-np.pi/4 
        phik    += np.pi/2*(1/2+(philast-phik)/np.pi*2)
        tmp      = (tmpr + 1j*tmpi)*np.exp(-1j*(phik+np.pi/2*(1/2+(philast-phik)/np.pi*2)))
        philast  = phik
        datar.extend(tmp.real)
        datai.extend(tmp.imag)

    tmpr,tmpi = rr[i:],ri[i:]
    xxx       = (tmpr+1j*tmpi)**2
    tmpr      = xxx.real
    tmpi      = xxx.imag
    Ak = np.vstack((tmpr.reshape((1,-1)),tmpi.reshape((1,-1))))
    Ck = np.matmul(Ak, Ak.T)
    vk   = np.matmul(Ck,vk)
    vk  /= np.sqrt(vk[0]**2+vk[1]**2)
    phik = 1/2*math.atan(vk[1]/vk[0])-np.pi/4 
    phik    += np.pi/2*(1/2+(philast-phik)/np.pi*2)
    tmp      = (tmpr + 1j*tmpi)*np.exp(-1j*(phik+np.pi/2*(1/2+(philast-phik)/np.pi*2)))
    philast  = phik
    datar.extend(tmp.real)
    datai.extend(tmp.imag)
    
    datar = np.array(datar)
    datai = np.array(datai)
    return datar,datai