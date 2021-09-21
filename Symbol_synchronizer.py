import numpy as np
import scipy.io as sio
import matlab.engine

# path = r'D:\Ian\QPSK\QPSKmat-master\20210517\Synchronize_data\SS.mat'
# loadfile = sio.loadmat(path)
# SSx = loadfile['SSx']
# SSy = loadfile['SSy']
class Symbol_synchronizer:
    def __init__(self,X,Y,sps=2):
        self.eng = matlab.engine.start_matlab()
        self.eng.workspace['X'] = matlab.double(X.tolist(),is_complex=True)
        self.eng.workspace['Y'] = matlab.double(Y.tolist(),is_complex=True)
        self.eng.eval('X = transpose(X)',nargout=0)
        self.eng.eval('Y = transpose(Y)',nargout=0)
        self.sps = sps
    def Gardner(self):
        self.eng.workspace['symbolSync'] = \
            self.eng.comm.SymbolSynchronizer('TimingErrorDetector','Gardner (non-data-aided)',\
                                             'SamplesPerSymbol',self.sps,'DampingFactor',.5)
        self.X_comp = self.eng.eval('symbolSync(X)')
        self.Y_comp = self.eng.eval('symbolSync(Y)')
        self.eng.quit()
        return np.array(self.X_comp).squeeze(), np.array(self.Y_comp).squeeze()
    def Early_Late(self):
        self.eng.workspace['symbolSync'] = \
            self.eng.comm.SymbolSynchronizer('TimingErrorDetector','Early-Late (non-data-aided)',\
                                             'SamplesPerSymbol',self.sps,'DampingFactor',1.)
        self.X_comp = self.eng.eval('symbolSync(X)')
        self.Y_comp = self.eng.eval('symbolSync(Y)')
        self.eng.quit()
        return np.array(self.X_comp).squeeze(), np.array(self.Y_comp).squeeze()    
