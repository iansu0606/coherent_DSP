"""
@author: Ian Su
Contact: ensu.ee08g@nycu.edu.tw
"""
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path
import os
import yaml
from tqdm import trange
from numba import jit
# @jit(nopython=True,cache=True)
def sample_more(input_signal, rate):
    signal_slice = input_signal[1000:2000]
    signal_slice -= np.mean(signal_slice)
    signal_slice *= (0.5/np.mean(np.abs(signal_slice)))
    # the first 8 is because we have to keep integer
    signal_eye = signal.resample_poly(signal_slice, up=(26.5625 * 16 * rate), down=92* 16)
    return signal_eye

def scan_eye(input_signal, rate):
    snr_list = np.array([])
    # print("mean = {}".format(np.mean(np.abs(input_signal[k::rate]))))
    for offset in range(0, rate):
        temp = hard_decision(input_signal[offset::rate])
        snr_list = np.append(snr_list, temp)

    index = np.where(snr_list == np.max(snr_list))
    # Transfer from tuple to integer
    index = int(index[0])
    return index, snr_list

def hard_decision(input_signal):
    # Please insert the resampled input array
    level3 = input_signal[input_signal > 2]
    # print("level 3 = {}".format(level3))
    temp = input_signal[input_signal < 2]
    level1 = temp[temp > 0]
    # print("level 1 = {}".format(level1))
    temp = input_signal[input_signal < 0]
    level_1 = temp[temp > -2]
    # print("level -1 = {}".format(level_1))
    level_3 = input_signal[input_signal < -2]
    # print("level -3 = {}".format(level_3))
    bias = np.array([])
    bias = np.append(bias, level3 / (level3 - 3))
    bias = np.append(bias, level1 / (level1 - 1))
    bias = np.append(bias, level_1 / (level_1 + 1))
    bias = np.append(bias, level_3 / (level_3 + 3))
    # print("bias = {}".format(bias))
    evm = np.mean(np.abs(bias))
    #print("evm = {}".format(evm))
    # print("evm = {}".format(evm))
    snr = 20 * np.log10(evm)
    return snr


#@jit(nopython=True,cache=True)
def data_extract(input_file):
    datum = None
    if input_file.endswith(".txt"):
        datum = np.genfromtxt(input_file)[:, 0]
    elif input_file.endswith(".mat"):
        import scipy.io as sio
        datum = sio.loadmat(input_file)
    return datum

def eye_plot(input_signal, offset, rate):
    for iterate in range(0, 150):
        signal_draw = input_signal[offset+iterate*rate*2:offset+(iterate+1)*rate*2]
        plt.plot(signal_draw, 'y')
    ax = plt.gca()
    ax.set(ylim=(-2, 2))
    ax.set_facecolor('k')
    ax.grid('w')
    ax.set_title('Eye diagram of Data: offset = {}'.format(offset), FontSize=20, FontWeight="bold")
    ax.set_xlabel('time (ps)', FontSize=16, FontWeight="bold")
    ax.set_ylabel('Amplitude (a.u.)', FontSize=16, FontWeight="bold")
    return

if __name__ == '__main__':
    # input Control pane
    parser = OptionParser()
    parser.add_option('--input', dest="input", help="input file", default='./behave_config.yml')
    (options, args) = parser.parse_args()
    # for object control, please go to behave.cfg
    # lg_cfg = yaml.safe_load(open('./behave_config.yml'))['Eye']
    lg_cfg = yaml.safe_load(open(options.input))['Eye']
    file = lg_cfg['file']
    sample_per_symbol = lg_cfg['sample_per_symbol']
    is_mat = lg_cfg['is_mat']
    is_fig = lg_cfg['is_fig']
    is_video = lg_cfg['is_video']
    is_pam4 = lg_cfg['is_pam4']
    is_keysight = lg_cfg['is_keysight']

    data = data_extract(file)
    # print(np.real(data['Y1']))
    data_XI = np.real(data['Y1'])
    data_XQ = np.imag(data['Y1'])
    data_YI = np.real(data['Y2'])
    data_YQ = np.imag(data['Y2'])
    sig = sample_more(data_XI, sample_per_symbol)
    eye_plot(sig, offset=2, rate=32)
    plt.show()