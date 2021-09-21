import scipy.io as sio
import numpy as np
import pandas as pd
import os


class Parameter:
    def __init__(self, simulation=False, upsample=32):
        self.path = os.path.abspath(os.getcwd())
        if simulation == True:
            self.datafolder = '20210108_FINISAR_16QAM/20210108_80KM_16QAM_14dB.mat'
            self.symbolRate = 56e9
            self.sampleRate = 32 * self.symbolRate
            self.upsamplenum = 1
            self.samplepersymbol = self.sampleRate / self.symbolRate
            self.resamplenumber = int(self.samplepersymbol * self.upsamplenum)
            self.timespan = 1000 #ps
            self.pamorder = 4
            self.Prbsnum = 13
            self.TxXI_prbs = np.reshape(
                np.append(pd.read_table(self.path + '/PRBS13/LogTxXI1.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                          pd.read_table(self.path + '/PRBS13/LogTxXI2.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
                order='F')
            self.TxXQ_prbs = np.reshape(
                np.append(pd.read_table(self.path + '/PRBS13/LogTxXQ1.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                          pd.read_table(self.path + '/PRBS13/LogTxXQ2.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
                order='F')
            self.TxYI_prbs = np.reshape(
                np.append(pd.read_table(self.path + '/PRBS13/LogTxYI1.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                          pd.read_table(self.path + '/PRBS13/LogTxYI2.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
                order='F')
            self.TxYQ_prbs = np.reshape(
                np.append(pd.read_table(self.path + '/PRBS13/LogTxYQ1.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                          pd.read_table(self.path + '/PRBS13/LogTxYQ2.txt', names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
                order='F')
            # self.TxXI_prbs = np.reshape(
            #     np.append(pd.read_table(self.path + '/PRBS13/0104/LogTxXI1.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy(),
            #               pd.read_table(self.path + '/PRBS13/0104/LogTxXI2.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            #     order='F')
            # self.TxXQ_prbs = np.reshape(
            #     np.append(pd.read_table(self.path + '/PRBS13/0104/LogTxXQ1.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy(),
            #               pd.read_table(self.path + '/PRBS13/0104/LogTxXQ2.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            #     order='F')
            # self.TxYI_prbs = np.reshape(
            #     np.append(pd.read_table(self.path + '/PRBS13/0104/LogTxYI1.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy(),
            #               pd.read_table(self.path + '/PRBS13/0104/LogTxYI2.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            #     order='F')
            # self.TxYQ_prbs = np.reshape(
            #     np.append(pd.read_table(self.path + '/PRBS13/0104/LogTxYQ1.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy(),
            #               pd.read_table(self.path + '/PRBS13/0104/LogTxYQ2.txt', names=['time', 'PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            #     order='F')
            # self.TxXI_prbs = self.TxXI_prbs[::self.resamplenumber, :]
            # self.TxXQ_prbs = self.TxXQ_prbs[::self.resamplenumber, :]
            # self.TxYI_prbs = self.TxYI_prbs[::self.resamplenumber, :]
            # self.TxYQ_prbs = self.TxYQ_prbs[::self.resamplenumber, :]
            self.datalength = int(self.symbolRate/1e9 * self.samplepersymbol * self.timespan)
            self.RxXI = pd.read_table(self.datafolder + 'RxXI.txt', names=['RxXI'])['RxXI'].tolist()[-self.datalength:]
            self.RxXQ = pd.read_table(self.datafolder + 'RxXQ.txt', names=['RxXQ'])['RxXQ'].tolist()[-self.datalength:]
            self.RxYI = pd.read_table(self.datafolder + 'RxYI.txt', names=['RxYI'])['RxYI'].tolist()[-self.datalength:]
            self.RxYQ = pd.read_table(self.datafolder + 'RxYQ.txt', names=['RxYQ'])['RxYQ'].tolist()[-self.datalength:]
            # self.TxXI = pd.read_table(self.datafolder + 'TxXI.txt', names=['TxXI'])['TxXI'].tolist()[-self.datalength:]
            # self.TxXQ = pd.read_table(self.datafolder + 'TxXQ.txt', names=['TxXQ'])['TxXQ'].tolist()[-self.datalength:]
            # self.TxYI = pd.read_table(self.datafolder + 'TxYI.txt', names=['TxYI'])['TxYI'].tolist()[-self.datalength:]
            # self.TxYQ = pd.read_table(self.datafolder + 'TxYQ.txt', names=['TxYQ'])['TxYQ'].tolist()[-self.datalength:]
        else:
            self.symbolRate = 28.125e9
            self.sampleRate = 50e9
            self.pamorder = 4
            self.upsamplenum = 9  #1024/225
            self.datapath = ''.join([self.path, '/', r'20210108_FINISAR_16QAM/00KM_16QAM.mat'])
            # self.txpath = self.path + r'\data\20210108_FINISAR_16QAM\00KM_16QAM.mat'
            self.txpath = self.datapath
            self.loadfile = sio.loadmat(self.datapath)
            self.loadtx = sio.loadmat(self.txpath)
            self.Prbsnum = 15
            # self.PRBS = pd.read_table(self.path + '/PRBS_TX15.txt', names=['PRBS'])['PRBS'].tolist()
            # self.samplepersymbol = self.sampleRate / self.symbolRate
            # self.resamplenumber = int(self.samplepersymbol * self.upsamplenum)
            self.resamplenumber = 16
            self.keysightdownnum = 5
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                               Tektronix Data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            self.RxXI = self.loadfile["Vblock"]["Values"][0][0][0]
            self.RxXQ = self.loadfile["Vblock"]["Values"][0][1][0]
            self.RxYI = self.loadfile["Vblock"]["Values"][0][2][0]
            self.RxYQ = self.loadfile["Vblock"]["Values"][0][3][0]
            self.Xsoft = self.loadfile["zXSym"]["Values"][0][0][0]  #tek DSP
            self.Ysoft = self.loadfile["zYSym"]["Values"][0][0][0]  #tek DSP
            if self.pamorder == 4:
                self.TxXI = self.loadtx["zXSym"]["SeqRe"][0][0][:]
                self.TxXI = np.reshape(np.reshape(self.TxXI, -1), (-1, 2), order='F')
                self.TxXQ = self.loadtx["zXSym"]["SeqIm"][0][0][:]
                self.TxXQ = np.reshape(np.reshape(self.TxXQ, -1), (-1, 2), order='F')
                self.TxYI = self.loadtx["zYSym"]["SeqRe"][0][0][:]
                self.TxYI = np.reshape(np.reshape(self.TxYI, -1), (-1, 2), order='F')
                self.TxYQ = self.loadtx["zYSym"]["SeqIm"][0][0][:]
                self.TxYQ = np.reshape(np.reshape(self.TxYQ, -1), (-1, 2), order='F')
            else:
                self.TxXI = self.loadfile["zXSym"]["SeqRe"][0][0][0].tolist()
                self.TxXQ = self.loadfile["zXSym"]["SeqIm"][0][0][0].tolist()
                self.TxYI = self.loadfile["zYSym"]["SeqRe"][0][0][0].tolist()
                self.TxYQ = self.loadfile["zYSym"]["SeqIm"][0][0][0].tolist()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                               keysight Data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#             #row data
#             self.RxX1 = np.reshape(self.loadfile["Y1_4"], -1)
#             self.RxX2 = np.reshape(self.loadfile["Y2_4"], -1)
#             self.RxY1 = np.reshape(self.loadfile["Y3_4"], -1)
#             self.RxY2 = np.reshape(self.loadfile["Y4_4"], -1)
#             self.RxXI = np.real(self.RxX1).tolist()
#             self.RxXQ = np.real(self.RxX2).tolist()
#             self.RxYI = np.real(self.RxY1).tolist()
#             self.RxYQ = np.real(self.RxY2).tolist()
#             # self.TxX = pd.read_table(self.path + '/data/20210323_KEYsight_28.125_16QAM/Keysight/exp/qpsk/err1.txt', names=['seq', 'Tx_sym'])['Tx_sym'].tolist()
#             # self.TxY = pd.read_table(self.path + '/data/20210323_KEYsight_28.125_16QAM/Keysight/exp/qpsk/err2.txt', names=['seq', 'Tx_sym'])['Tx_sym'].tolist()
#             #Keysight Ref data(Tx)
#             self.keysight_X = pd.read_table(self.path + '/data/20210420_data/qpsk/no algorithm/ch1_ref.txt', names=['X_real', 'X_imag'])#, skiprows=129)
#             self.keysightXI_ref = self.keysight_X['X_real'].to_numpy()
#             self.keysightXQ_ref = self.keysight_X['X_imag'].to_numpy()
#             self.keysight_Y = pd.read_table(self.path + '/data/20210420_data/qpsk/no algorithm/ch3_ref.txt', names=['Y_real', 'Y_imag'])#, skiprows=129)
#             self.keysightYI_ref = self.keysight_Y['Y_real'].to_numpy()
#             self.keysightYQ_ref = self.keysight_Y['Y_imag'].to_numpy()
#             #Keysight meas data
#             self.keysight_X = pd.read_table(self.path + '/data/20210420_data/qpsk/no algorithm/ch1.txt', names=['X_real', 'X_imag'])#, skiprows=129)
#             self.keysightXI_meas = self.keysight_X['X_real'].to_numpy()
#             self.keysightXQ_meas = self.keysight_X['X_imag'].to_numpy()
#             self.keysight_Y = pd.read_table(self.path + '/data/20210420_data/qpsk/no algorithm/ch3.txt', names=['Y_real', 'Y_imag'])#, skiprows=129)
#             self.keysightYI_meas = self.keysight_Y['Y_real'].to_numpy()
#             self.keysightYQ_meas = self.keysight_Y['Y_imag'].to_numpy()