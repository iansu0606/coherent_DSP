import numpy as np
import cmath
from subfunction.DD import *
from subfunction.Histogram2D import *
from subfunction.Histogram2D_Hot import *
import matplotlib.pyplot as plt
class Phaserecovery:
    def __init__(self, isig, taps=101):
        self.isig = np.array(isig)
        self.tapscan = [3, 5, 19, 21, 31]
        self.taps = taps
        self.center = int((self.taps-1)/2)

    def V_Valg(self):
        if np.shape(self.isig) == (len(self.isig), ):
            print("reshape input data")
            self.isig = np.reshape(self.isig, (1, len(self.isig)))
        batchnum = np.shape(self.isig)[0]
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((batchnum, datalength), dtype="complex_")
        for batch in range(batchnum):
            phase = np.zeros((datalength, 1))
            phaseadj = np.zeros((datalength, 1))
            ak = np.zeros((datalength, 1))
            for indx in range(self.center, datalength-self.center):
                phase[indx] = (cmath.phase(
                    np.sum(self.isig[batch][indx - self.center:indx + self.center + 1] ** 4)) - cmath.pi) / 4
                ak[indx] = ak[indx - 1] + np.floor(0.5 - 4 * ((phase[indx] - phase[indx - 1]) / (2 * cmath.pi)))
                phaseadj[indx] = phase[indx] + ak[indx] * 2 * cmath.pi / 4
                self.rx_recovery[batch][indx] = self.isig[batch][indx] * cmath.exp(-1j * phaseadj[indx])
        self.t = ak
        self.y = phase
        self.rx_recovery = self.rx_recovery[:, self.center:-1-self.center+1]
        return self.rx_recovery


    def PLL(self):
        bandwidth = 0.005
        dampingfactor = 0.707
        theta = bandwidth / (dampingfactor + 1 / (4 * dampingfactor))
        d = 1 + 2 * dampingfactor * theta + theta ** 2
        Kp = 2
        K0 = 1
        g1 = 4 * theta ** 2 / (K0 * Kp * d)
        gp = 4 * dampingfactor * theta / (K0 * Kp * d)
        if np.shape(self.isig) == (len(self.isig),):
            print("reshape input data")
            self.isig = np.reshape(self.isig, (1, len(self.isig)))
        batchnum = np.shape(self.isig)[0]
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((batchnum, datalength), dtype="complex_")
        for batch in range(batchnum):
            err = np.zeros((datalength, 1))
            weight = np.zeros((datalength, 1))
            lamb = np.zeros((datalength, 1))
            self.rx_recovery[batch][0] = self.isig[batch][0] * cmath.exp(-1j * lamb[0])
            err[0] = np.sign(np.real(self.rx_recovery[batch][0])) * np.imag(self.rx_recovery[batch][0]) - np.sign(
                np.imag(self.rx_recovery[batch][0])) * np.real(self.rx_recovery[batch][0])
            weight[0] = err[0] * g1
            for it in range(10):
                for indx in range(1, datalength):
                    lamb[indx] = gp * err[indx - 1] + weight[indx - 1] + lamb[indx - 1]
                    self.rx_recovery[batch][indx] = self.isig[batch][indx] * cmath.exp(-1j * lamb[indx])
                    err[indx] = np.sign(np.real(self.rx_recovery[batch][indx])) * np.imag(
                        self.rx_recovery[batch][indx]) - np.sign(np.imag(self.rx_recovery[batch][indx])) * np.real(
                        self.rx_recovery[batch][indx])
                    weight[indx] = g1 * err[indx] + weight[indx - 1]
        self.rx_recovery = self.rx_recovery[:, 1:]
        return self.rx_recovery

    def DD_PLL(self):
        if np.shape(self.isig) == (len(self.isig),):
            print("reshape input data")
            self.isig = np.reshape((1, len(self.isig)))
        batchnum = np.shape(self.isig)[0]
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((batchnum, datalength), dtype="complex_")
        stepsize = 1e-5
        for batch in range(batchnum):
            err = np.zeros((datalength, 1), dtype='complex_')
            lamb = np.zeros((datalength, 1), dtype='float32')
            z = np.zeros((datalength, 1), dtype='complex_')
            z[0] = self.isig[batch][0] * cmath.exp(-1j * lamb[0])
            err[0] = z[0] - DD(z[0], 4)
            self.rx_recovery[batch][0] = DD(z[0], 4)
            for it in range(3):
                for indx in range(1, datalength):
                    lamb[indx] = lamb[indx - 1] - stepsize * np.imag(z[indx - 1] * err[indx - 1])
                    z[indx] = self.isig[batch][indx] * cmath.exp(-1j * lamb[indx])
                    DD_z = DD(z[indx], 4)
                    err[indx] = z[indx] - DD_z
                    self.rx_recovery[batch][indx] = DD_z
        self.rx_recovery = self.rx_recovery[:, 1:]
        return self.rx_recovery

    def FFT_FOE(self, Rs=56e9, Rres=1e3):   #Too small Rres may lead to memory error
        N = int(Rs / Rres)
        rx = self.isig
        rx_4th = np.array(rx) ** 4
        length = len(rx)
        dft_f = abs(np.fft.fftshift(np.fft.fft(rx_4th, N)))


        # plt.figure()
        # a = np.linspace(0,4000, 4000)
        # xx = np.argmax(dft_f)
        # b = dft_f[xx-2000:xx+2000]
        # plt.plot(a, b)
        # plt.show()

        f_arg = np.argmax(dft_f) - N / 2 - 1
        f_offset = Rs * f_arg / (4 * N)
        print("Frequeucy Offset = {:.5f}GHz".format(f_offset / 1e9))
        rx = rx * np.exp(-f_offset * np.linspace(1, length, length) * 2 * np.pi * 1j / Rs)


        # plt.figure()
        # ss = abs(np.fft.fftshift(np.fft.fft(np.array(rx) ** 4, N)))
        # qq = 2000
        # b = ss[qq-2000:qq+2000]
        # plt.plot(a,b)
        # plt.show()
        return rx

    def QPSK_partition_VVPE(self, path_image,inner = 2.3, outter = 3.8):
        if np.shape(self.isig) == (len(self.isig), ):
            print("reshape input data")
            self.isig = np.reshape(self.isig, (1, len(self.isig)))
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((0, datalength), dtype="complex_")
        rx_partition_c1c3, rx_part = [], []
        for indx in range(0, datalength):
            if abs(self.isig[0][indx]) >= outter or abs(self.isig[0][indx]) <= inner:
                rx_partition_c1c3.append(self.isig[0][indx])
                rx_part.append(self.isig[0][indx])
            else:
                rx_partition_c1c3.append(np.nan)
        rx_partition_c1c3 = np.array(rx_partition_c1c3)
        Histogram2D_Hot("Partition data_outer"+str(outter), np.array(rx_part), path=path_image)
        phase = np.zeros((datalength, 1))
        phaseadj = np.zeros((datalength, 1))
        adj = np.zeros((datalength, 1))
        for indx in range(self.center, datalength-self.center):
            rx_c1c3_4th = np.ma.masked_equal(rx_partition_c1c3[indx-self.center:indx+self.center+1], np.nan) ** 4
            phase[indx] = (np.angle(
                np.sum(rx_c1c3_4th / abs(rx_c1c3_4th))) - cmath.pi) / 4
            adj[indx] = adj[indx - 1] + np.floor(0.5 - 4 * (phase[indx] - phase[indx - 1]) / (2 * cmath.pi))
            phaseadj[indx] = phase[indx] + adj[indx] * 2 * cmath.pi / 4
            self.isig[0][indx] = self.isig[0][indx] * cmath.exp(-1j * phaseadj[indx])
        Histogram2D_Hot('Phase_fourth result', rx_c1c3_4th, path=path_image)
        Histogram2D_Hot('QPSK_Partition_outer'+str(outter), self.isig[0], path=path_image)
        return self.isig[0]

    def VVPE_eight(self, path_image):
        if np.shape(self.isig) == (len(self.isig),):
            print("reshape input data")
            self.isig = np.reshape(self.isig, (1, len(self.isig)))
        batchnum = np.shape(self.isig)[0]
        datalength = np.shape(self.isig)[1]
        self.rx_recovery = np.zeros((batchnum, datalength), dtype="complex_")
        for batch in range(batchnum):
            phase = np.zeros((datalength, 1))
            phaseadj = np.zeros((datalength, 1))
            ak = np.zeros((datalength, 1))
            for indx in range(self.center, datalength - self.center):
                phase[indx] = (cmath.phase(
                    np.sum(self.isig[batch][indx - self.center:indx + self.center + 1] ** 8)) - cmath.pi) / 8
                ak[indx] = ak[indx - 1] + np.floor(0.5 - 8 * ((phase[indx] - phase[indx - 1]) / (2 * cmath.pi)))
                phaseadj[indx] = phase[indx] + ak[indx] * 2 * cmath.pi / 8
                self.rx_recovery[batch][indx] = self.isig[batch][indx] * cmath.exp(-1j * phaseadj[indx])
        self.rx_recovery = self.rx_recovery[:, self.center:-1 - self.center + 1].reshape(-1)
        return self.rx_recovery


    def CT(self, rx, path_image, taps = 51):
        center = int((taps-1)/2)
        datalength = len(rx)
        rx_adj = []
        for indx in range(datalength):
            rx_real = np.real(rx[indx]) - np.sign(np.real(rx[indx]) - 2 * np.sign(np.real(rx[indx])))
            rx_imag = np.imag(rx[indx]) - np.sign(np.imag(rx[indx]) - 2 * np.sign(np.imag(rx[indx])))
            rx_adj.append(rx_real + 1j * rx_imag)
        rx_adj = np.array(rx_adj)
        Histogram2D_Hot('C transform', rx_adj, path=path_image)
        phase = np.zeros((datalength, 1))
        phaseadj = np.zeros((datalength, 1))
        adj = np.zeros((datalength, 1))
        for indx in range(center, datalength - center):
            phase[indx] = (cmath.phase(
                np.sum(rx_adj[indx - center:indx + center + 1] ** 4)) - cmath.pi) / 4
            adj[indx] = adj[indx - 1] + np.floor(0.5 - 4 * (phase[indx] - phase[indx - 1]) / (2 * cmath.pi))
            phaseadj[indx] = phase[indx] + adj[indx] * 2 * cmath.pi / 4
            rx[indx] = rx[indx] * cmath.exp(-1j * phaseadj[indx])
        Histogram2D_Hot("CT", rx, path=path_image)
        return rx

    def ML(self, rx, path_image, taps=51):
        datalength = len(rx)
        center = int((taps-1)/2)
        h = np.zeros((datalength, 1), dtype='complex')
        phaseML = np.zeros((datalength, 1))
        for indx in range(center, datalength-center):
            X = rx[indx-center:indx+center+1]
            Y_DD = DD(X, 4)
            h[indx] = np.sum(X * np.conj(Y_DD))
            phaseML[indx] = np.arctan(np.imag(h[indx])/np.real(h[indx]))
            rx[indx] = rx[indx] * cmath.exp(-1j * phaseML[indx])
        Histogram2D_Hot("ML", rx, path=path_image)
        return rx

    def Enhanced_ML(self, rx, path_image, taps=51):
        delta = 0.1
        center = int((taps-1)/2)
        datalength = len(rx)
        phase = np.zeros((datalength, 1))
        ak = np.zeros((datalength, ), dtype='complex')
        phaseML = np.zeros((datalength, 1))
        for indx in range(center, datalength-center):
            X = rx[indx-center:indx+center+1]
            Y_DD = DD(X, 4)
            ak[indx-center:indx+center+1] = X * np.conj(Y_DD)
            ak_tap = np.angle(ak[indx - center:indx + center + 1])
            phase[indx] = np.sum(ak_tap)/taps
            low = phase[indx] - delta
            high = phase[indx] + delta
            ak[indx-center:indx+center+1] = np.where(low < ak_tap, ak[indx-center:indx+center+1], 0)
            ak[indx-center:indx+center+1] = np.where(ak_tap < high, ak[indx-center:indx+center+1], 0)
            H = np.sum(ak[indx-center:indx+center+1])
            phaseML[indx] = np.arctan(np.imag(H)/np.real(H))
            rx[indx] = rx[indx] * cmath.exp(-1j * phaseML[indx])
        Histogram2D_Hot("E_ML", rx, path=path_image)
        return rx