import scipy.io as sio
import numpy as np
import pandas as pd
import os
import yaml


class Parameter:
    def __init__(self, proc_folder, proc_file, high_rate=False, simulation=False, tektronic=True):
        self.path = os.path.abspath(os.getcwd())
        self.config = yaml.safe_load(open('./behave_config.yml'))
        if self.config['is_simulation']:
            self.config = self.config['Simulationset']
        else:
            self.config = self.config['Setting']
        if simulation == True:
            self.datafolder = ''.join(['../exp_data/', proc_folder])
            self.symbolRate = float(self.config['symbolRate'])
            self.sampleRate = 32 * self.symbolRate
            self.upsamplenum = 1
            self.samplepersymbol = self.sampleRate / self.symbolRate
            self.resamplenumber = int(self.samplepersymbol * self.upsamplenum)
            self.timespan = 1000  # ps
            self.pamorder = 4 # 16QAM
            self.Prbsnum = 13
            self.datalength = int(self.symbolRate / 1e9 * self.samplepersymbol * self.timespan)
            self.LogTxXI1 = pd.read_table('../exp_data/PRBS13/' + 'LogTxXI1.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxXI2 = pd.read_table('../exp_data/PRBS13/' + 'LogTxXI2.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxXQ1 = pd.read_table('../exp_data/PRBS13/' + 'LogTxXQ1.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxXQ2 = pd.read_table('../exp_data/PRBS13/' + 'LogTxXQ2.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxYI1 = pd.read_table('../exp_data/PRBS13/' + 'LogTxYI1.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxYI2 = pd.read_table('../exp_data/PRBS13/' + 'LogTxYI2.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxYQ1 = pd.read_table('../exp_data/PRBS13/' + 'LogTxYQ1.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.LogTxYQ2 = pd.read_table('../exp_data/PRBS13/' + 'LogTxYQ2.txt', names=['L1'], skiprows=9)['L1'].to_numpy().astype(np.int8)
            self.RxXI = pd.read_table(self.datafolder + 'RxXI.txt', names=['RxXI'], skiprows=9)['RxXI'].to_numpy()[
                        -self.datalength:]
            self.RxXQ = pd.read_table(self.datafolder + 'RxXQ.txt', names=['RxXQ'], skiprows=9)['RxXQ'].to_numpy()[
                        -self.datalength:]
            self.RxYI = pd.read_table(self.datafolder + 'RxYI.txt', names=['RxYI'], skiprows=9)['RxYI'].to_numpy()[
                        -self.datalength:]
            self.RxYQ = pd.read_table(self.datafolder + 'RxYQ.txt', names=['RxYQ'], skiprows=9)['RxYQ'].to_numpy()[
                        -self.datalength:]
        else:
            self.datapath = ''.join(['../exp_data/', proc_folder, proc_file])
            self.loadfile = sio.loadmat(self.datapath)
            self.Prbsnum = 15
            is_QPSK = self.config['is_QPSK']
            if high_rate:
                self.symbolRate = 53.125e9  # keysight
            else:
                self.symbolRate = 26.5625e9 # keysight
            if is_QPSK:
                self.pamorder = 2
                self.txpath = ''.join(['../exp_data/', r'PRBS15_X.mat'])  # QPSK
            else:
                self.pamorder = 4
                self.txpath = ''.join(['../exp_data/', r'PRBSQ_15_X.mat'])  # 16QAM
            self.loadtx = sio.loadmat(self.txpath)
            # --------------------------------------------------------------------------------------------------------------------------------
            #                               Tektronic Data
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if tektronic:
                self.sampleRate = 50e9
                self.symbolRate = 26.5625e9
                self.resamplenumber = 32
                self.upsamplenum = 17
                self.RxXI = self.loadfile["Vblock"]["Values"][0][0][0]
                self.RxXQ = self.loadfile["Vblock"]["Values"][0][1][0]
                self.RxYI = self.loadfile["Vblock"]["Values"][0][2][0]
                self.RxYQ = self.loadfile["Vblock"]["Values"][0][3][0]
                self.Xsoft = self.loadfile["zXSym"]["Values"][0][0][0]  # tek DSP
                self.Ysoft = self.loadfile["zYSym"]["Values"][0][0][0]  # tek DSP
                # --------------------------------------------------------------------------------------------------------------------------------
    #                               keysight Data
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                #row data
            else:
                self.sampleRate = 92e9  # keysight
                if high_rate:
                    self.upsamplenum = self.config['keysight_upsample_high']  # keysight
                    self.downsample_number = self.config['first_stage_downsample_high']
                else:
                    self.upsamplenum = self.config['keysight_upsample']  # keysight
                    self.downsample_number = self.config['first_stage_downsample']
                self.RxX1 = np.reshape(self.loadfile["Y1"], -1)
                self.RxX2 = np.reshape(self.loadfile["Y1"], -1)
                self.RxY1 = np.reshape(self.loadfile["Y2"], -1)
                self.RxY2 = np.reshape(self.loadfile["Y2"], -1)
                self.RxXI = np.real(self.RxX1)
                self.RxXQ = np.imag(self.RxX2)
                self.RxYI = np.real(self.RxY1)
                self.RxYQ = np.imag(self.RxY2)
            if self.pamorder == 4:
                if tektronic:
                    self.TxXI = self.loadfile["zXSym"]["SeqRe"][0][0][:]
                    self.TxXI = np.reshape(np.reshape(self.TxXI, -1), (-1, 2), order='F')
                    self.TxXQ = self.loadfile["zXSym"]["SeqIm"][0][0][:]
                    self.TxXQ = np.reshape(np.reshape(self.TxXQ, -1), (-1, 2), order='F')
                    self.TxYI = self.loadfile["zYSym"]["SeqRe"][0][0][:]
                    self.TxYI = np.reshape(np.reshape(self.TxYI, -1), (-1, 2), order='F')
                    self.TxYQ = self.loadfile["zYSym"]["SeqIm"][0][0][:]
                    self.TxYQ = np.reshape(np.reshape(self.TxYQ, -1), (-1, 2), order='F')
                else:
                    self.TxXI = self.loadtx["zXSym"]["SeqRe"][0][0][:]
                    self.TxXI = np.reshape(np.reshape(self.TxXI, -1), (-1, 2), order='F')
                    self.TxXQ = self.loadtx["zXSym"]["SeqIm"][0][0][:]
                    self.TxXQ = np.reshape(np.reshape(self.TxXQ, -1), (-1, 2), order='F')
                    self.txpath = ''.join(['../exp_data/', r'PRBSQ_15_Y.mat'])
                    self.loadtx = sio.loadmat(self.txpath)
                    self.TxYI = self.loadtx["zXSym"]["SeqRe"][0][0][:]
                    self.TxYI = np.reshape(np.reshape(self.TxYI, -1), (-1, 2), order='F')
                    self.TxYQ = self.loadtx["zXSym"]["SeqIm"][0][0][:]
                    self.TxYQ = np.reshape(np.reshape(self.TxYQ, -1), (-1, 2), order='F')
            else:
                if tektronic:
                    self.TxXI = self.loadfile["zXSym"]["SeqRe"][0][0][0].tolist()
                    self.TxXQ = self.loadfile["zXSym"]["SeqIm"][0][0][0].tolist()
                    self.TxYI = self.loadfile["zYSym"]["SeqRe"][0][0][0].tolist()
                    self.TxYQ = self.loadfile["zYSym"]["SeqIm"][0][0][0].tolist()
                else:
                    self.TxXI = self.loadtx["zXSym"]["SeqRe"][0][0][0].tolist()
                    self.TxXQ = self.loadtx["zXSym"]["SeqIm"][0][0][0].tolist()
                    self.txpath = ''.join(['../exp_data/', r'PRBS15_Y.mat'])
                    self.loadtx = sio.loadmat(self.txpath)
                    self.TxYI = self.loadtx["zYSym"]["SeqRe"][0][0][0].tolist()
                    self.TxYQ = self.loadtx["zYSym"]["SeqIm"][0][0][0].tolist()