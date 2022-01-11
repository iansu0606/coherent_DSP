import numpy as np
import cmath
import math
from tqdm import trange, tqdm
from subfunction.Histogram2D_Hot import *

class CMA:
    def __init__(self, rx_x, rx_y, taps=[21, 21], iter=30):
        self.rx_x_single = np.array(rx_x)
        self.rx_y_single = np.array(rx_y)
        self.rx_x = np.array(rx_x)
        self.rx_y = np.array(rx_y)
        self.datalength = len(rx_x)
        #                      0    1      2    3     4        5      6     7      8
        self.stepsizelist = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 6.409e-6, 1e-6, 5e-5, 8e-7,
                             #      9    10    11    12      13    14     15
                             1e-7, 1e-8, 1e9, 1e-10, 5e-13, 1e-14, 1e-25]
        # self.batchsize = 32767*2
        self.batchsize = self.datalength
        self.overhead = 1
        self.trainlength = round(self.batchsize * self.overhead)
        self.cmataps = taps
        self.center = int(np.max(self.cmataps)//2)
        self.batchnum = int(self.datalength/self.batchsize)
        self.iterator = iter
        self.earlystop = 0.0001
        self.stepsizeadjust = 0.9
        self.rx_x.resize((self.batchnum, self.batchsize), refcheck=False)
        self.rx_y.resize((self.batchnum, self.batchsize), refcheck=False)
        self.stepsize = [self.stepsizelist[4], self.stepsizelist[4]]
        self.stepsize_x = self.stepsize[0]
        self.stepsize_y = self.stepsize[1]

    def run(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize-self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize-self.center), dtype="complex_")
        for batch in range(self.batchnum):
            #initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    errx[indx] = (10 - (np.abs(exout[indx]))**2)
                    erry[indx] = (10 - (np.abs(eyout[indx]))**2)
                    hxx = hxx + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * exout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * eyout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:])**2)
                self.costfuny[batch][it] = np.mean((erry[self.center:])**2)
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                            self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))
            for indx in range(self.center, self.batchsize - self.center):  #taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def CMA_butterfly(self, is_twostage=0):
        #initialize H
        if is_twostage:
            R = 10
            print("========Successful setting first stage R========")
        else:
            R = 2
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1 + 0j
        hyy[self.center] = 1 + 0j
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="float32")
        erry = np.zeros(self.datalength, dtype="float32")
        stepx, stepy = self.stepsize[0] ,self.stepsize[1]
        for it in trange(self.iterator):
            for indx in range(self.center, self.datalength-self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                errx[indx] = (np.abs(exout[indx]))**2 - R
                erry[indx] = (np.abs(eyout[indx]))**2 - R
                hxx = hxx - stepx * errx[indx] * exout[indx] * np.conj(inputrx[
                                                                       indx - self.center:indx + self.center + 1])
                hxy = hxy - stepx * errx[indx] * exout[indx] * np.conj(inputry[
                                                                       indx - self.center:indx + self.center + 1])
                hyx = hyx - stepy * erry[indx] * eyout[indx] * np.conj(inputrx[
                                                                       indx - self.center:indx + self.center + 1])
                hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conj(inputry[
                                                                       indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:])**2)
            self.costfuny[0][it] = np.mean((erry[self.center:])**2)
            if it > 20:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < 0.001:
                    stepx *= self.stepsizeadjust
                    print('Stepsize adjust to {:.2E}'.format(stepx))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < 0.001:
                    stepy *= self.stepsizeadjust
                    print('Stepsize adjust to {:.2E}'.format(stepy))
            print('X cost:{:.3f}'.format(self.costfunx[0][it]), 'Y cost:{:.3f}'.format(self.costfuny[0][it]))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def CMA_single(self):
        #initialize H
        self.costfunx = np.zeros((1, self.iterator), dtype="complex")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex")
        hyy = np.zeros(self.cmataps, dtype="complex")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex")
        eyout = np.zeros(self.datalength, dtype="complex")
        errx = np.zeros(self.datalength, dtype="complex")
        erry = np.zeros(self.datalength, dtype="complex")
        stepx, stepy = self.stepsize[0] ,self.stepsize[1]
        for it in trange(self.iterator):
            for indx in range(self.center, self.datalength-self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                errx[indx] = (np.abs(exout[indx])) ** 2 - 2
                erry[indx] = (np.abs(eyout[indx])) ** 2 - 2
                hxx = hxx - stepx * exout[indx] * errx[indx] * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy - stepy * eyout[indx] * erry[indx] * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < 0.001:
                    stepx *= self.stepsizeadjust
                    print('Stepsize adjust to {:.2E}'.format(stepx))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < 0.001:
                    stepy *= self.stepsizeadjust
                    print('Stepsize adjust to {:.2E}'.format(stepy))
            print('X cost:{:.3f}'.format(self.costfunx[0][it]), 'Y cost:{:.3f}'.format(self.costfuny[0][it]))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def run_16qam_single(self):
        # initialize H
        self.rx_x_cma = np.zeros((1, self.datalength), dtype="complex_")
        self.rx_y_cma = np.zeros((1, self.datalength), dtype="complex_")
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1 + 0j
        hyy[self.center] = 1 + 0j
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        radx = np.zeros(self.datalength, dtype="complex_")
        rady = np.zeros(self.datalength, dtype="complex_")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                exin = inputrx[indx - self.center:indx + self.center + 1]
                eyin = inputry[indx - self.center:indx + self.center + 1]
                rad_aaa = np.real(exin) - 2 * np.sign(np.real(exin)) + 1j * (
                        np.imag(exin) - 2 * np.sign(np.imag(exin)))
                rad_bbb = np.real(eyin) - 2 * np.sign(np.real(eyin)) + 1j * (
                        np.imag(eyin) - 2 * np.sign(np.imag(eyin)))
                radx[indx] = np.mean(np.abs(rad_aaa ** 4)) / np.mean(np.abs(rad_aaa ** 2))
                rady[indx] = np.mean(np.abs(rad_bbb ** 4)) / np.mean(np.abs(rad_bbb ** 2))
                exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                        np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                        np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                errx[indx] = ((np.abs(exout_adj)) ** 2 - radx[indx])
                erry[indx] = ((np.abs(eyout_adj)) ** 2 - rady[indx])
                hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])
                hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                    inputry[indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.stepsizeadjust * \
                        self.costfuny[0][it]:
                    self.stepsize *= 0.5
                    print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_cma = exout[:-self.center]
        self.rx_y_cma = eyout[:-self.center]

    def run_16qam(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                                np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                    eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                            np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    errx[indx] = ((np.abs(exout_adj)) ** 2 - 2)
                    erry[indx] = ((np.abs(eyout_adj)) ** 2 - 2)
                    hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:]) ** 2)
                self.costfuny[batch][it] = np.mean((erry[self.center:]) ** 2)
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    # if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                    #         self.costfunx[batch][it] and np.abs(
                    #     self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                    #         self.costfuny[batch][it]:
                    #     self.stepsize *= 0.5
                    #     print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def run_16qam_single_ch(self):
        self.costfunx = np.zeros((1, self.iterator), dtype="complex")
        self.rx_x_cma = np.zeros((1, self.datalength), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                            np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                errx[indx] = ((np.abs(exout_adj)) ** 2 - 10)
                hxx = hxx - self.stepsize * errx[indx] * exout[indx] * np.conj(inputrx[indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and it > 20:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                        self.costfunx[0][it]:
                    self.stepsize *= 0.7
                    print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_single = exout[self.center:-self.center]

    def run_16qam_butterfly(self):
        self.rx_x_cma = np.zeros((1, self.datalength), dtype="complex_")
        self.rx_y_cma = np.zeros((1, self.datalength), dtype="complex_")
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1 + 0j
        hyy[self.center] = 1 + 0j
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(np.conj(hxx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conj(hxy), inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(np.conj(hyx), inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    np.conj(hyy), inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                        np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                        np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                errx[indx] = ((np.abs(exout_adj)) ** 2 - 10)
                erry[indx] = ((np.abs(eyout_adj)) ** 2 - 10)
                hxx = hxx - self.stepsize * np.conj(errx[indx] * exout_adj2) * inputrx[indx - self.center:indx + self.center + 1]
                hxy = hxy - self.stepsize * np.conj(errx[indx] * exout_adj2) * inputry[indx - self.center:indx + self.center + 1]
                hyx = hyx - self.stepsize * np.conj(erry[indx] * eyout_adj2) * inputrx[indx - self.center:indx + self.center + 1]
                hyy = hyy - self.stepsize * np.conj(erry[indx] * eyout_adj2) * inputry[indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                #         self.costfunx[0][it] and np.abs(
                #     self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.stepsizeadjust * \
                #         self.costfuny[0][it]:
                #     self.stepsize *= 0.5
                #     print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_cma = exout[self.center:-self.center]
        self.rx_y_cma = eyout[self.center:-self.center]

    def RDE_CMA_butterfly(self):
        stepx, stepy = self.stepsize[0] ,self.stepsize[1]
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        tapx, tapy = self.cmataps[0], self.cmataps[1]
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(tapx, dtype="complex_")
        hxy = np.zeros(tapy, dtype="complex_")
        hyx = np.zeros(tapx, dtype="complex_")
        hyy = np.zeros(tapy, dtype="complex_")
        centerx, centery = tapx//2, tapy//2
        hxx[centerx] = 1
        hyy[centery] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        R = [2, 10, 18]
        for it in trange(self.iterator):
            if it < 7:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(np.conjugate(hxx),
                                            inputrx[indx - centerx:indx + centerx + 1]) + np.matmul(
                        np.conjugate(hxy), inputry[indx - centery:indx + centery + 1])
                    eyout[indx] = np.matmul(np.conjugate(hyx),
                                            inputrx[indx - centerx:indx + centerx + 1]) + np.matmul(
                        np.conjugate(hyy), inputry[indx - centery:indx + centery + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    errx[indx] = ((np.abs(exout[indx])) ** 2 - 10)
                    erry[indx] = ((np.abs(eyout[indx])) ** 2 - 10)
                    hxx = hxx - stepx * np.conjugate(errx[indx] * exout[indx]) * inputrx[
                                                                                         indx - centerx:indx + centerx + 1]
                    hxy = hxy - stepx * np.conjugate(errx[indx] * exout[indx]) * inputry[
                                                                                         indx - centery:indx +centery + 1]
                    hyx = hyx - stepy * np.conjugate(erry[indx] * eyout[indx]) * inputrx[
                                                                                         indx - centerx:indx + centerx + 1]
                    hyy = hyy - stepy * np.conjugate(erry[indx] * eyout[indx]) * inputry[
                                                                                         indx - centery:indx + centery + 1]
                    # Histogram2D_Hot("1st stsge", exout[self.center:-self.center])
            else:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(np.conj(hxx), inputrx[indx - centerx:indx + centerx + 1]) + np.matmul(
                        np.conj(hxy), inputry[indx - centery:indx + centery + 1])
                    eyout[indx] = np.matmul(np.conj(hyx), inputrx[indx - centerx:indx + centerx + 1]) + np.matmul(
                        np.conj(hyy), inputry[indx - centery:indx + centery + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    xdistance = [np.abs(np.abs(exout[indx]) - R[0] ** 0.5), np.abs(np.abs(exout[indx]) - R[1] ** 0.5),
                                 np.abs(np.abs(exout[indx]) - R[2] ** 0.5)]
                    ydistance = [np.abs(np.abs(eyout[indx]) - R[0] ** 0.5), np.abs(np.abs(eyout[indx]) - R[1] ** 0.5),
                                 np.abs(np.abs(eyout[indx]) - R[2] ** 0.5)]
                    errx[indx] = np.abs(exout[indx])**2 - R[np.argmin(xdistance)]
                    erry[indx] = np.abs(eyout[indx])**2 - R[np.argmin(ydistance)]
                    hxx = hxx - stepx * np.conj(errx[indx] * exout[indx]) * inputrx[
                                                                                    indx - centerx:indx + centerx + 1]
                    hxy = hxy - stepx * np.conj(errx[indx] * exout[indx]) * inputry[
                                                                                    indx - centery:indx + centery + 1]
                    hyx = hyx - stepy * np.conj(erry[indx] * eyout[indx]) * inputrx[
                                                                                    indx - centerx:indx + centerx + 1]
                    hyy = hyy - stepy * np.conj(erry[indx] * eyout[indx]) * inputry[
                                                                                    indx - centery:indx + centery + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print('X_Pol Cost:{:.2f}'.format(self.costfunx[0][it]), 'Y_pol Cost:{:.2f}'.format(self.costfuny[0][it]))
            if it > 7:
                # if(it ==5 and self.costfunx[0][it] < 21 and self.costfuny[0][it] < 21):
                #     print("can't solve")
                #     break
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < 0.001:
                    stepx *= self.stepsizeadjust
                    if stepx <= 1E-8:
                        stepx = 1E-8
                    print('X Stepsize adjust to {:.2E}'.format(stepx))

                if np.abs(self.costfuny[0][it] - costmeany) < 0.001:
                    stepy *= self.stepsizeadjust
                    if stepy <= 1E-8:
                        stepy = 1E-8
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def RDE_CMA_single(self):    #good snr = 15.3
        stepx, stepy = self.stepsize[0] ,self.stepsize[1]
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        R = [2, 10, 18]
        for it in trange(self.iterator):
            if it < 20:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    errx[indx] = ((np.abs(exout[indx])) ** 2 - 10)
                    erry[indx] = ((np.abs(eyout[indx])) ** 2 - 10)
                    hxx = hxx - stepx * errx[indx] * exout[indx] * np.conj(inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conj(inputry[indx - self.center:indx + self.center + 1])
            else:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    xdistance = [np.abs(np.abs(exout[indx]) - R[0] ** 0.5), np.abs(np.abs(exout[indx]) - R[1] ** 0.5),
                                 np.abs(np.abs(exout[indx]) - R[2] ** 0.5)]
                    ydistance = [np.abs(np.abs(eyout[indx]) - R[0] ** 0.5), np.abs(np.abs(eyout[indx]) - R[1] ** 0.5),
                                 np.abs(np.abs(eyout[indx]) - R[2] ** 0.5)]
                    errx[indx] = np.abs(exout[indx])**2 - R[np.argmin(xdistance)]
                    erry[indx] = np.abs(eyout[indx])**2 - R[np.argmin(ydistance)]
                    hxx = hxx - stepx * errx[indx] * exout[indx] * np.conj(inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conj(inputry[indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print('X_Pol Cost:{:2f}'.format(self.costfunx[0][it]), 'Y_pol Cost:{:.2f}'.format(self.costfuny[0][it]))

            if it > 7:
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < 0.01:
                    stepx *= self.stepsizeadjust
                    if stepx <= 1E-8:
                        stepx = 1E-8
                    print('X Stepsize adjust to {:.2E}'.format(stepx))

                if np.abs(self.costfuny[0][it] - costmeany) < 0.01:
                    stepy *= self.stepsizeadjust
                    if stepy <= 1E-8:
                        stepy = 1E-8
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def RDE_butterfly(self):
        self.record = np.zeros((100, self.iterator), dtype="complex")
        self.prec = np.zeros((self.iterator, 1), dtype="float32")
        stepx, stepy = self.stepsize[0] ,self.stepsize[1]
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        R = [2, 10, 18]
        for it in trange(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                xdistance = [np.abs(np.abs(exout[indx]) - R[0] ** 0.5), np.abs(np.abs(exout[indx]) - R[1] ** 0.5),
                             np.abs(np.abs(exout[indx]) - R[2] ** 0.5)]
                ydistance = [np.abs(np.abs(eyout[indx]) - R[0] ** 0.5), np.abs(np.abs(eyout[indx]) - R[1] ** 0.5),
                             np.abs(np.abs(eyout[indx]) - R[2] ** 0.5)]
                errx[indx] = np.abs(exout[indx])**2 - R[np.argmin(xdistance)]
                erry[indx] = np.abs(eyout[indx])**2 - R[np.argmin(ydistance)]
                hxx = hxx - stepx * errx[indx] * exout[indx] * np.conj(inputrx[
                                                                                indx - self.center:indx + self.center + 1])
                hxy = hxy - stepx * errx[indx] * exout[indx] * np.conj(inputry[
                                                                                indx - self.center:indx + self.center + 1])
                hyx = hyx - stepy * erry[indx] * eyout[indx] * np.conj(inputrx[
                                                                                indx - self.center:indx + self.center + 1])
                hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conj(inputry[
                                                                                indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            for re in range(0, 10):
                self.record[re][it] = exout[re+100]
            self.prec[it] = np.abs(cmath.phase(exout[3000])-cmath.phase(exout[9000]))
            print('X_Pol Cost:{:.2f}'.format(self.costfunx[0][it]), 'Y_pol Cost:{:.2f}'.format(self.costfuny[0][it]))
            if it > 1:
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < 0.001:
                    stepx *= self.stepsizeadjust
                    if stepx <= 1E-8:
                        stepx = 1E-8
                    print('X Stepsize adjust to {:.2E}'.format(stepx))

                if np.abs(self.costfuny[0][it] - costmeany) < 0.001:
                    stepy *= self.stepsizeadjust
                    if stepy <= 1E-8:
                         stepy = 1E-8
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
            # hyy = np.conj(hxx)
            # hxy = np.conj(hyx)
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def RDE_single(self):    #good snr = 15.3
        stepx, stepy = self.stepsize[0] ,self.stepsize[1]
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        R = [2, 10, 18]
        costmeanx, costmeany = 0, 0
        for it in trange(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                xdistance = [np.abs(np.abs(exout[indx]) - R[0] ** 0.5), np.abs(np.abs(exout[indx]) - R[1] ** 0.5),
                             np.abs(np.abs(exout[indx]) - R[2] ** 0.5)]
                ydistance = [np.abs(np.abs(eyout[indx]) - R[0] ** 0.5), np.abs(np.abs(eyout[indx]) - R[1] ** 0.5),
                             np.abs(np.abs(eyout[indx]) - R[2] ** 0.5)]
                errx[indx] = np.abs(exout[indx])**2 - R[np.argmin(xdistance)]
                erry[indx] = np.abs(eyout[indx])**2 - R[np.argmin(ydistance)]
                hxx = hxx - stepx * errx[indx] * exout[indx] * np.conjugate(inputrx[
                                                                                indx - self.center:indx + self.center + 1])
                hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conjugate(inputry[
                                                                                indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print('X_Pol Cost:{:2f}'.format(self.costfunx[0][it]), 'Y_pol Cost:{:.2f}'.format(self.costfuny[0][it]))

            if it > 1:
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < 0.001:
                    stepx *= self.stepsizeadjust
                    if stepx <= 1E-8:
                        stepx = 1E-8
                    print('X Stepsize adjust to {:.2E}'.format(stepx))

                if np.abs(self.costfuny[0][it] - costmeany) < 0.001:
                    stepy *= self.stepsizeadjust
                    if stepy <= 1E-8:
                        stepy = 1E-8
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def MCMA_single(self):
        stepx = self.stepsize
        stepy = self.stepsize
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        costmeanx, costmeany = 0, 0
        for it in range(self.iterator):
            if it < 10:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    errx[indx] = ((np.abs(exout[indx])) ** 2 - 10)
                    erry[indx] = ((np.abs(eyout[indx])) ** 2 - 10)
                    hxx = hxx - stepx * errx[indx] * exout[indx] * np.conj(inputrx[
                                                                                         indx - self.center:indx + self.center + 1])
                    hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conj(inputry[
                                                                                         indx - self.center:indx + self.center + 1])
            else:
                for indx in range(self.center, self.datalength - self.center):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exout_adj = np.real(exout[indx]) - 2 * np.sign(np.real(exout[indx])) + 1j * (
                            np.imag(exout[indx]) - 2 * np.sign(np.imag(exout[indx])))
                    eyout_adj = np.real(eyout[indx]) - 2 * np.sign(np.real(eyout[indx])) + 1j * (
                            np.imag(eyout[indx]) - 2 * np.sign(np.imag(eyout[indx])))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    errx[indx] = ((np.abs(exout_adj)) ** 2 - 10)
                    erry[indx] = ((np.abs(eyout_adj)) ** 2 - 10)
                    hxx = hxx - stepx * errx[indx] * exout[indx] * np.conj(inputrx[
                                                                           indx - self.center:indx + self.center + 1])
                    hyy = hyy - stepy * erry[indx] * eyout[indx] * np.conj(inputry[
                                                                           indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print('X_Pol Cost:{}'.format(self.costfunx[0][it]), 'Y_pol Cost:{}'.format(self.costfuny[0][it]))

            if it > 10:
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < self.stepsizeadjust * \
                        self.costfunx[0][it]:
                    stepx *= 0.5
                    if stepx <= 1E-7:
                        stepx = 1E-7
                    print('X Stepsize adjust to {:.2E}'.format(stepx))
                if np.abs(self.costfuny[0][it] - costmeany) < self.stepsizeadjust * \
                        self.costfuny[0][it]:
                    stepy *= 0.5
                    if stepy <= 1E-7:
                        stepy = 1E-7
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def run_CIAEMCMA(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            radx = np.zeros(self.trainlength + self.center, dtype="complex_")
            rady = np.zeros(self.trainlength + self.center, dtype="complex_")
            stepsizex = 1e-4
            stepsizey = 1e-4
            maxstepsize = 0.08
            minstepsize = 1e-6
            gama = 0.1
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exin = inputrx[indx - self.center:indx + self.center + 1]
                    eyin = inputry[indx - self.center:indx + self.center + 1]
                    rad_aaa = np.real(exin) - 2 * np.sign(np.real(exin)) + 1j * (
                            np.imag(exin) - 2 * np.sign(np.imag(exin)))
                    rad_bbb = np.real(eyin) - 2 * np.sign(np.real(eyin)) + 1j * (
                            np.imag(eyin) - 2 * np.sign(np.imag(eyin)))
                    radx[indx] = np.mean(np.abs(rad_aaa ** 4)) / np.mean(np.abs(rad_aaa ** 4))
                    rady[indx] = np.mean(np.abs(rad_bbb ** 4)) / np.mean(np.abs(rad_bbb ** 4))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    errx[indx] = ((np.abs(exout_adj2)) ** 2 - radx[indx])
                    erry[indx] = ((np.abs(eyout_adj2)) ** 2 - rady[indx])

                    px = math.exp(-np.abs(errx[indx] - errx[indx - 1]))+math.exp(-indx/gama)
                    py = math.exp(-np.abs(erry[indx] - erry[indx - 1]))+math.exp(-indx/gama)
                    stepsizex = stepsizex * px
                    stepsizey = stepsizey * py
                    if stepsizex > maxstepsize:
                        stepsizex = maxstepsize
                    elif stepsizex < minstepsize:
                        stepsizex = minstepsize
                    if stepsizey > maxstepsize:
                        stepsizey = maxstepsize
                    elif stepsizey < minstepsize:
                        stepsizey = minstepsize
                    hxx = hxx + stepsizex * errx[indx] * exout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + stepsizex * errx[indx] * exout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + stepsizey * erry[indx] * eyout[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + stepsizey * erry[indx] * eyout[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:]) ** 2)
                self.costfuny[batch][it] = np.mean((erry[self.center:]) ** 2)
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def MCMA(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            cost_x = np.zeros(self.trainlength + self.center, dtype="complex_")
            # cost_xq = np.zeros(self.trainlength + self.center, dtype="complex_")
            cost_y = np.zeros(self.trainlength + self.center, dtype="complex_")
            # cost_yq = np.zeros(self.trainlength + self.center, dtype="complex_")
            radxi = np.zeros(self.trainlength + self.center, dtype="complex_")
            radxq = np.zeros(self.trainlength + self.center, dtype="complex_")
            radyi = np.zeros(self.trainlength + self.center, dtype="complex_")
            radyq = np.zeros(self.trainlength + self.center, dtype="complex_")
            ZR, ZI = 2, 2
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                    exq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                    eyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                    eyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    rad_xi = exi - 4 * np.sign(exi) - 2 * np.sign(exi - 4 * np.sign(exi))
                    rad_xq = exq - 2 * np.sign(exq) - 2 * np.sign(exq - 4 * np.sign(exq))
                    rad_yi = eyi - 2 * np.sign(eyi) - 2 * np.sign(eyi - 4 * np.sign(eyi))
                    rad_yq = eyq - 2 * np.sign(eyq) - 2 * np.sign(eyq - 4 * np.sign(eyq))
                    disxi = np.min([np.abs(np.real(exout[indx]) - 7.5), np.abs(np.real(exout[indx]) - 2.5), np.abs(np.real(exout[indx]) + 2.5),
                                    np.abs(np.real(exout[indx]) + 7.5)])
                    disxq = np.min([np.abs(np.imag(exout[indx]) - 7.5), np.abs(np.imag(exout[indx]) - 2.5), np.abs(np.imag(exout[indx]) + 2.5),
                                    np.abs(np.imag(exout[indx]) + 7.5)])
                    disyi = np.min([np.abs(np.real(eyout[indx]) - 7.5), np.abs(np.real(eyout[indx]) - 2.5), np.abs(np.real(eyout[indx]) + 2.5),
                                    np.abs(np.real(eyout[indx]) + 7.5)])
                    disyq = np.min([np.abs(np.imag(eyout[indx]) - 7.5), np.abs(np.imag(eyout[indx]) - 2.5), np.abs(np.imag(eyout[indx]) + 2.5),
                                    np.abs(np.imag(eyout[indx]) + 7.5)])
                    if (disxi) < ZR:
                        radxi[indx] = 10
                    else:
                        radxi[indx] = np.mean(np.abs(rad_xi) ** 4) / np.mean(np.abs(rad_xi) ** 2)
                    if (disxq) < ZI:
                        radxq[indx] = 10
                    else:
                        radxq[indx] = np.mean(np.abs(rad_xq) ** 4) / np.mean(np.abs(rad_xq) ** 2)
                    if(disyi) < ZR:
                        radyi[indx] = 10
                    else:
                        radyi[indx] = np.mean(np.abs(rad_yi) ** 4) / np.mean(np.abs(rad_yi) ** 2)
                    if (disyq) < ZI:
                        radyq[indx] = 10
                    else:
                        radyq[indx] = np.mean(np.abs(rad_yq) ** 4) / np.mean(np.abs(rad_yq) ** 2)
                    exout_adj_i = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                       np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                    exout_adj_q = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                       np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                    eyout_adj_i = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                       np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                    eyout_adj_q = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                       np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))
                    errx[indx] = np.real(exout[indx]) * (radxi[indx] - np.abs(exout_adj_i) ** 2) + \
                                 1j * np.imag(exout[indx]) * (radxq[indx] - np.abs(exout_adj_q) ** 2)
                    erry[indx] = np.real(eyout[indx]) * (radyi[indx] - np.abs(eyout_adj_i) ** 2) + \
                                 1j * np.imag(eyout[indx]) * (radyq[indx] - np.abs(eyout_adj_q) ** 2)
                    hxx = hxx + self.stepsize * errx[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy + self.stepsize * errx[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx + self.stepsize * erry[indx] * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy + self.stepsize * erry[indx] * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    cost_x[indx] = exout_adj_i ** 2 - radxi[indx] + 1j * (
                            exout_adj_q ** 2 - radxq[indx])
                    cost_y[indx] = eyout_adj_i ** 2 - radyi[indx] + 1j * (
                            eyout_adj_q ** 2 - radyq[indx])
                self.costfunx[batch][it] = -1 * (np.mean(np.real(cost_x[self.center:])) + np.mean(np.imag(cost_x[self.center:])))
                self.costfuny[batch][it] = -1 * (np.mean(np.real(cost_y[self.center:])) + np.mean(np.imag(cost_y[self.center:])))
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            np.abs(self.costfunx[batch][it]) and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            np.abs(self.costfuny[batch][it]):
                        print("Earlybreak at iterator {}".format(it))
                        break
                if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                        np.abs(self.costfunx[batch][it]) and np.abs(
                    self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                        np.abs(self.costfuny[batch][it]):
                    self.stepsize *= 0.5
                    print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def MCMA_2_single(self):
        stepx = self.stepsize
        stepy = self.stepsize
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength, dtype="complex_")
        cost_y = np.zeros(self.datalength, dtype="complex_")
        radxi = np.zeros(self.datalength, dtype="complex_")
        radxq = np.zeros(self.datalength, dtype="complex_")
        radyi = np.zeros(self.datalength, dtype="complex_")
        radyq = np.zeros(self.datalength, dtype="complex_")
        ZR, ZI = 0.5, 0.5
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exi = np.real(inputrx[indx - self.center:indx + self.center + 1])
                exq = np.imag(inputrx[indx - self.center:indx + self.center + 1])
                eyi = np.real(inputry[indx - self.center:indx + self.center + 1])
                eyq = np.imag(inputry[indx - self.center:indx + self.center + 1])
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                rad_xi = exi - 4 * np.sign(exi) - 2 * np.sign(exi - 4 * np.sign(exi))
                rad_xq = exq - 2 * np.sign(exq) - 2 * np.sign(exq - 4 * np.sign(exq))
                rad_yi = eyi - 2 * np.sign(eyi) - 2 * np.sign(eyi - 4 * np.sign(eyi))
                rad_yq = eyq - 2 * np.sign(eyq) - 2 * np.sign(eyq - 4 * np.sign(eyq))
                disxi = np.min([np.abs(np.real(exout[indx]) - 3), np.abs(np.real(exout[indx]) - 1), np.abs(np.real(exout[indx]) + 1),
                                np.abs(np.real(exout[indx]) + 3)])
                disxq = np.min([np.abs(np.imag(exout[indx]) - 3), np.abs(np.imag(exout[indx]) - 1), np.abs(np.imag(exout[indx]) + 1),
                                np.abs(np.imag(exout[indx]) + 3)])
                disyi = np.min([np.abs(np.real(eyout[indx]) - 3), np.abs(np.real(eyout[indx]) - 1), np.abs(np.real(eyout[indx]) + 1),
                                np.abs(np.real(eyout[indx]) + 3)])
                disyq = np.min([np.abs(np.imag(eyout[indx]) - 3), np.abs(np.imag(eyout[indx]) - 1), np.abs(np.imag(eyout[indx]) + 1),
                                np.abs(np.imag(eyout[indx]) + 3)])
                if (disxi) < ZR:
                    radxi[indx] = 2
                else:
                    radxi[indx] = np.mean(np.abs(rad_xi) ** 4) / np.mean(np.abs(rad_xi) ** 2)
                if (disxq) < ZI:
                    radxq[indx] = 2
                else:
                    radxq[indx] = np.mean(np.abs(rad_xq) ** 4) / np.mean(np.abs(rad_xq) ** 2)
                if(disyi) < ZR:
                    radyi[indx] = 2
                else:
                    radyi[indx] = np.mean(np.abs(rad_yi) ** 4) / np.mean(np.abs(rad_yi) ** 2)
                if (disyq) < ZI:
                    radyq[indx] = 2
                else:
                    radyq[indx] = np.mean(np.abs(rad_yq) ** 4) / np.mean(np.abs(rad_yq) ** 2)
                exout_adj_i = np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])) - 2 * \
                                   np.sign(np.real(exout[indx]) - 4 * np.sign(np.real(exout[indx])))
                exout_adj_q = np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])) - 2 * \
                                   np.sign(np.imag(exout[indx]) - 4 * np.sign(np.imag(exout[indx])))
                eyout_adj_i = np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])) - 2 * \
                                   np.sign(np.real(eyout[indx]) - 4 * np.sign(np.real(eyout[indx])))
                eyout_adj_q = np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])) - 2 * \
                                   np.sign(np.imag(eyout[indx]) - 4 * np.sign(np.imag(eyout[indx])))
                errx[indx] = np.real(exout[indx]) * (np.abs(exout_adj_i) ** 2 - radxi[indx]) + \
                             1j * np.imag(exout[indx]) * (np.abs(exout_adj_q) ** 2 - radxq[indx])
                erry[indx] = np.real(eyout[indx]) * (np.abs(eyout_adj_i) ** 2 - radyi[indx]) + \
                             1j * np.imag(eyout[indx]) * (np.abs(eyout_adj_q) ** 2 - radyq[indx])
                hxx = hxx - self.stepsize * errx[indx] * np.conj(inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy - self.stepsize * erry[indx] * np.conj(inputry[indx - self.center:indx + self.center + 1])
                cost_x[indx] = exout_adj_i ** 2 - radxi[indx] + 1j * (
                        exout_adj_q ** 2 - radxq[indx])
                cost_y[indx] = eyout_adj_i ** 2 - radyi[indx] + 1j * (
                        eyout_adj_q ** 2 - radyq[indx])
            self.costfunx[0][it] = -1 * (np.mean(np.real(cost_x[self.center:])) + np.mean(np.imag(cost_x[self.center:])))
            self.costfuny[0][it] = -1 * (np.mean(np.real(cost_y[self.center:])) + np.mean(np.imag(cost_y[self.center:])))
            print(self.costfunx[0][it])
            if it > 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
                        np.abs(self.costfunx[0][it]) and np.abs(
                    self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.earlystop * \
                        np.abs(self.costfuny[0][it]):
                    print("Earlybreak at iterator {}".format(it))
                    break
            if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.stepsizeadjust * \
                    np.abs(self.costfunx[0][it]) or np.abs(
                self.costfuny[0][it] - self.costfuny[0][it - 1]) < self.stepsizeadjust * \
                    np.abs(self.costfuny[0][it]):
                self.stepsize *= 0.7
                print('Stepsize adjust to {}'.format(self.stepsize))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def run_RLS(self):
        self.costfunx = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((self.batchnum, self.iterator), dtype="complex_")
        self.rx_x_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        self.rx_y_cma = np.zeros((self.batchnum, self.batchsize - self.center), dtype="complex_")
        for batch in range(self.batchnum):
            # initialize H
            inputrx = self.rx_x[batch]
            inputry = self.rx_y[batch]
            hxx = np.zeros(self.cmataps, dtype="complex_")
            hxy = np.zeros(self.cmataps, dtype="complex_")
            hyx = np.zeros(self.cmataps, dtype="complex_")
            hyy = np.zeros(self.cmataps, dtype="complex_")
            hxx[self.center] = 1
            hyy[self.center] = 1
            exout = np.zeros(self.trainlength + self.center, dtype="complex_")
            eyout = np.zeros(self.trainlength + self.center, dtype="complex_")
            errx = np.zeros(self.trainlength + self.center, dtype="complex_")
            erry = np.zeros(self.trainlength + self.center, dtype="complex_")
            radx = np.zeros(self.trainlength + self.center, dtype="complex_")
            rady = np.zeros(self.trainlength + self.center, dtype="complex_")
            for it in range(self.iterator):
                for indx in range(self.center, self.center + self.trainlength):
                    exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hxy, inputry[indx - self.center:indx + self.center + 1])
                    eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                        hyy, inputry[indx - self.center:indx + self.center + 1])
                    if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                        raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                    exin = inputrx[indx - self.center:indx + self.center + 1]
                    eyin = inputry[indx - self.center:indx + self.center + 1]
                    rad_aaa = np.real(exin) - 2 * np.sign(np.real(exin)) + 1j * (
                            np.imag(exin) - 2 * np.sign(np.imag(exin)))
                    rad_bbb = np.real(eyin) - 2 * np.sign(np.real(eyin)) + 1j * (
                            np.imag(eyin) - 2 * np.sign(np.imag(eyin)))
                    radx[indx] = np.mean(np.abs(rad_aaa ** 8)) / np.mean(np.abs(rad_aaa ** 8))
                    rady[indx] = np.mean(np.abs(rad_bbb ** 8)) / np.mean(np.abs(rad_bbb ** 8))
                    exout_adj2 = exout[indx] - 2 * np.sign(np.real(exout[indx])) - 2j * (np.sign(np.imag(exout[indx])))
                    eyout_adj2 = eyout[indx] - 2 * np.sign(np.real(eyout[indx])) - 2j * (np.sign(np.imag(eyout[indx])))
                    errx[indx] = ((np.abs(exout_adj2)) ** 4 - radx[indx]) * np.abs(exout_adj2) ** 2
                    erry[indx] = ((np.abs(eyout_adj2)) ** 4 - rady[indx]) * np.abs(exout_adj2) ** 2
                    hxx = hxx - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hxy = hxy - self.stepsize * errx[indx] * exout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                    hyx = hyx - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputrx[indx - self.center:indx + self.center + 1])
                    hyy = hyy - self.stepsize * erry[indx] * eyout_adj2 * np.conjugate(
                        inputry[indx - self.center:indx + self.center + 1])
                self.costfunx[batch][it] = np.mean((errx[self.center:]) ** 2)
                self.costfuny[batch][it] = np.mean((erry[self.center:]) ** 2)
                print(self.costfunx[batch][it])
                if it > 1:
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.earlystop * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.earlystop * \
                            self.costfuny[batch][it]:
                        print("Earlybreak at iterator {}".format(it))
                        break
                    if np.abs(self.costfunx[batch][it] - self.costfunx[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfunx[batch][it] and np.abs(
                        self.costfuny[batch][it] - self.costfuny[batch][it - 1]) < self.stepsizeadjust * \
                            self.costfuny[batch][it]:
                        self.stepsize *= 0.5
                        print('Stepsize adjust to {}'.format(self.stepsize))

            for indx in range(self.center, self.batchsize - self.center):  # taps-1 point overhead
                self.rx_x_cma[batch][indx] = np.matmul(hxx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                self.rx_y_cma[batch][indx] = np.matmul(hyx,
                                                       inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])

    def RDE(self):
        stepx = self.stepsize
        stepy = self.stepsize
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        R = [2, 10, 18]
        costmeanx, costmeany = 0, 0
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(np.conj(hxx), inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(np.conj(hyy), inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                xdistance = [np.abs(np.abs(exout[indx]) - R[0]), np.abs(np.abs(exout[indx]) - R[1]),
                             np.abs(np.abs(exout[indx]) - R[2])]
                ydistance = [np.abs(np.abs(eyout[indx]) - R[0]), np.abs(np.abs(eyout[indx]) - R[1]),
                             np.abs(np.abs(eyout[indx]) - R[2])]
                errx[indx] = np.abs(exout[indx])**2 - R[np.argmin(xdistance)]
                erry[indx] = np.abs(eyout[indx])**2 - R[np.argmin(ydistance)]
                hxx = hxx - stepx * np.conj(errx[indx] * exout[indx]) * inputrx[
                                                                                indx - self.center:indx + self.center + 1]
                hyy = hyy - stepy * np.conj(erry[indx] * eyout[indx]) * inputry[
                                                                                indx - self.center:indx + self.center + 1]
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print('X_Pol Cost:{}'.format(self.costfunx[0][it]), 'Y_pol Cost:{}'.format(self.costfuny[0][it]))

            if it > 15:
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < self.stepsizeadjust * \
                        self.costfunx[0][it]:
                    stepx *= 0.5
                    if stepx <= 1E-7:
                        stepx = 1E-7
                    print('X Stepsize adjust to {:.2E}'.format(stepx))

                if np.abs(self.costfuny[0][it] - costmeany) < self.stepsizeadjust * \
                        self.costfuny[0][it]:
                    stepy *= 0.5
                    if stepy <= 1E-7:
                        stepy = 1E-7
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def MCMA_tt(self):
        stepx = self.stepsize
        stepy = self.stepsize
        self.costfunx = np.zeros((1, self.iterator), dtype="complex_")
        self.costfuny = np.zeros((1, self.iterator), dtype="complex_")
        # initialize H
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")
        errx = np.zeros(self.datalength, dtype="complex_")
        erry = np.zeros(self.datalength, dtype="complex_")
        radxi = np.zeros(self.datalength, dtype="complex_")
        radxq = np.zeros(self.datalength, dtype="complex_")
        radyi = np.zeros(self.datalength, dtype="complex_")
        radyq = np.zeros(self.datalength, dtype="complex_")
        costmeanx, costmeany = 0, 0
        ZR, ZI = 0.5, 0.5
        R = [-3, -1, 1, 3]
        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
                disxi = np.min([np.abs(np.real(exout[indx]) - 3), np.abs(np.real(exout[indx]) - 1), np.abs(np.real(exout[indx]) + 1),
                                np.abs(np.real(exout[indx]) + 3)])
                disxiind = np.argmin([np.abs(np.real(exout[indx]) - 3), np.abs(np.real(exout[indx]) - 1), np.abs(np.real(exout[indx]) + 1),
                                np.abs(np.real(exout[indx]) + 3)])
                disxq = np.min([np.abs(np.imag(exout[indx]) - 3), np.abs(np.imag(exout[indx]) - 1), np.abs(np.imag(exout[indx]) + 1),
                                np.abs(np.imag(exout[indx]) + 3)])
                disxqind = np.argmin([np.abs(np.imag(exout[indx]) - 3), np.abs(np.imag(exout[indx]) - 1), np.abs(np.imag(exout[indx]) + 1),
                                np.abs(np.imag(exout[indx]) + 3)])
                disyi = np.min([np.abs(np.real(eyout[indx]) - 3), np.abs(np.real(eyout[indx]) - 1), np.abs(np.real(eyout[indx]) + 1),
                                np.abs(np.real(eyout[indx]) + 3)])
                disyiind = np.argmin([np.abs(np.real(eyout[indx]) - 3), np.abs(np.real(eyout[indx]) - 1), np.abs(np.real(eyout[indx]) + 1),
                                np.abs(np.real(eyout[indx]) + 3)])
                disyq = np.min([np.abs(np.imag(eyout[indx]) - 3), np.abs(np.imag(eyout[indx]) - 1), np.abs(np.imag(eyout[indx]) + 1),
                                np.abs(np.imag(eyout[indx]) + 3)])
                disyqind = np.argmin([np.abs(np.imag(eyout[indx]) - 3), np.abs(np.imag(eyout[indx]) - 1), np.abs(np.imag(eyout[indx]) + 1),
                                np.abs(np.imag(eyout[indx]) + 3)])
                if (disxi) < ZR and it >10:
                    radxi[indx] = R[disxiind]**2
                else:
                    radxi[indx] = 10
                if (disxq) < ZI and it >10:
                    radxq[indx] = R[disxqind]**2
                else:
                    radxq[indx] = 10
                if (disyi) < ZR and it >10:
                    radyi[indx] = R[disyiind]**2
                else:
                    radyi[indx] = 10
                if (disyq) < ZI and it >10:
                    radyq[indx] = R[disyqind]**2
                else:
                    radyq[indx] = 10
                errx[indx] = np.real(exout[indx]) * (np.abs(np.real(exout[indx])) ** 2 - radxi[indx]) + \
                             1j * np.imag(exout[indx]) * (np.abs(np.imag(exout[indx])) ** 2 - radxq[indx])
                erry[indx] = np.real(eyout[indx]) * (np.abs(np.real(eyout[indx])) ** 2 - radyi[indx]) + \
                             1j * np.imag(eyout[indx]) * (np.abs(np.imag(eyout[indx])) ** 2 - radyq[indx])
                hxx = hxx - stepx * errx[indx] * np.conj(inputrx[
                                                          indx - self.center:indx + self.center + 1])
                hyy = hyy - stepy * erry[indx] * np.conj(inputry[
                                                          indx - self.center:indx + self.center + 1])
            self.costfunx[0][it] = np.mean((errx[self.center:]) ** 2)
            self.costfuny[0][it] = np.mean((erry[self.center:]) ** 2)
            print('X_Pol Cost:{}'.format(self.costfunx[0][it]), 'Y_pol Cost:{}'.format(self.costfuny[0][it]))

            if it > 10:
                costmeanx = np.mean(self.costfunx[0][it - 1:it])
                costmeany = np.mean(self.costfuny[0][it - 1:it])
                if np.abs(self.costfunx[0][it] - costmeanx) < self.earlystop * \
                        self.costfunx[0][it] and np.abs(
                    self.costfuny[0][it] - costmeany) < self.earlystop * \
                        self.costfuny[0][it]:
                    print("Earlybreak at iterator {}".format(it))
                    break
                if np.abs(self.costfunx[0][it] - costmeanx) < self.stepsizeadjust * \
                        self.costfunx[0][it]:
                    stepx *= 0.5
                    if stepx <= 1E-7:
                        stepx = 1E-7
                    print('X Stepsize adjust to {:.2E}'.format(stepx))
                if np.abs(self.costfuny[0][it] - costmeany) < self.stepsizeadjust * \
                        self.costfuny[0][it]:
                    stepy *= 0.5
                    if stepy <= 1E-7:
                        stepy = 1E-7
                    print('Y Stepsize adjust to {:.2E}'.format(stepy))
        self.rx_x_single = exout[self.center:-self.center]
        self.rx_y_single = eyout[self.center:-self.center]

    def qam_4_butter_RD(self,stage,radii):  # A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS
        self.type = 'butter_RD'
        self.costfunx = np.zeros((1, self.iterator))
        self.costfuny = np.zeros((1, self.iterator))
        inputrx = self.rx_x_single
        inputry = self.rx_y_single
        hxx = np.zeros(self.cmataps, dtype="complex_")
        hxy = np.zeros(self.cmataps, dtype="complex_")
        hyx = np.zeros(self.cmataps, dtype="complex_")
        hyy = np.zeros(self.cmataps, dtype="complex_")
        hxx[self.center] = 1
        hyy[self.center] = 1
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")

        # errx = np.zeros(self.datalength, dtype="complex_")
        cost_x = np.zeros(self.datalength)
        cost_y = np.zeros(self.datalength)
        if stage == 2:
            radius = [2 ** 0.5, 10 ** 0.5, 18 ** 0.5]
        elif stage == 1:
            radius = [radii ** 0.5]
        # QPSK is sqrt(2), 16QAM is sqrt(10)

        HardDecision_X = np.zeros(len(radius))
        HardDecision_Y = np.zeros(len(radius))

        for it in range(self.iterator):
            for indx in range(self.center, self.datalength - self.center):
                exout[indx] = np.matmul(hxx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hxy, inputry[indx - self.center:indx + self.center + 1])
                eyout[indx] = np.matmul(hyx, inputrx[indx - self.center:indx + self.center + 1]) + np.matmul(
                    hyy, inputry[indx - self.center:indx + self.center + 1])
                if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                    raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

                for i in range(len(radius)):
                    HardDecision_X[i] = radius[i] - np.abs(exout[indx])
                    HardDecision_Y[i] = radius[i] - np.abs(eyout[indx])

                errx = (radius[np.argmin(abs(HardDecision_X))] ** 2 - np.abs(exout[indx]) ** 2) * (exout[indx])
                erry = (radius[np.argmin(abs(HardDecision_Y))] ** 2 - np.abs(eyout[indx]) ** 2) * (eyout[indx])

                hxx = hxx + self.stepsize_x * errx * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hxy = hxy + self.stepsize_x * errx * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])
                hyx = hyx + self.stepsize_y * erry * np.conj(
                    inputrx[indx - self.center:indx + self.center + 1])
                hyy = hyy + self.stepsize_y * erry * np.conj(
                    inputry[indx - self.center:indx + self.center + 1])

                cost_x[indx] = (abs(exout[indx])) ** 2 - radius[np.argmin(abs(HardDecision_X))] ** 2
                cost_y[indx] = (abs(eyout[indx])) ** 2 - radius[np.argmin(abs(HardDecision_Y))] ** 2

            self.costfunx[0][it] = np.mean(cost_x) * -1
            self.costfuny[0][it] = np.mean(cost_y) * -1

            print('iteration = {}'.format(it))
            print(self.costfunx[0][it])
            print(self.costfuny[0][it])
            print('-------')

            if it >= 1:
                if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < (self.earlystop * 100):
                    self.stepsize_x *= self.stepsizeadjust
                    print('Stepsize_x adjust to {}'.format(self.stepsize_x))
                if np.abs(self.costfuny[0][it] - self.costfuny[0][it - 1]) < (self.earlystop * 100):
                    self.stepsize_y *= self.stepsizeadjust
                    print('Stepsize_y adjust to {}'.format(self.stepsize_y))

        self.rx_x_cma = exout
        self.rx_y_cma = eyout