from optparse import OptionParser
import numpy as np
# from eyediagram.mpl import eyediagram
# from eyediagram.demo_data import demo_data
import matplotlib.pyplot as plt
import scipy.signal as signal


def eye_diagram(input_signal):
    signal_slice = input_signal[499:1699]
    #plt.stem(signal_slice[0:100])
    print("This slice bias ={}".format(np.mean(signal_slice)))
    signal_slice -= np.mean(signal_slice)
    print("abs average ={}".format(np.mean(np.abs(signal_slice))))
    signal_slice *= (2/np.mean(np.abs(signal_slice)))
    # the first 8 is because we have to keep integer
    # the second 8 is because we need to resample at 8 times in order to draw the eye diagram
    signal_eye = signal.resample_poly(signal_slice,up = (53.125 * 8 * 8) , down = 92* 8  )
    return signal_eye

def scan_eye(input_signal):
    snr_list = np.array([])
    for k in range(0,8):
        hard_decision(input_signal[k::8],snr_list)

    #index = np.where(snr_list==np.max(snr_list))
    return index

def scatter(input_signal,index):
    plt.scatter([i for i in range(len(input_signal[index::8]))],input_signal[index::8])
    ax = plt.gca()
    ax.set(ylim=(-4, 4))
    plt.yticks(np.arange(-4, 4, step=1))
    plt.show()

    return


def hard_decision(input_signal,snr_list):
        # Please insert the resampled input array
        level3 = input_signal[input_signal>2]
        temp = input_signal[input_signal<2]
        level1 = temp[temp>0]
        temp = input_signal[input_signal<0]
        level_1 = temp[temp>-2]
        level_3 = input_signal[input_signal<-2]
        bias = np.array([])
        bias = np.append(bias,np.abs((level3-3)/level3))
        bias = np.append(bias,np.abs((level1-1)/level1))
        bias = np.append(bias,np.abs((level_1+1)/level_1))
        bias = np.append(bias,np.abs((level_3+3)/level_3))
        evm = np.average(bias)
        snr = 20 * np.log10(evm)
        snr_list = np.append(snr_list,snr)
        return

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input', dest="input", help="input file name")
    parser.add_option('--sam_per_sym',dest="sym_sample",help="sample per symbol")
    (options, args) = parser.parse_args()
    data = np.genfromtxt(options.input)
    data = data[:,0]
    sig = eye_diagram(data)
    print(scan_eye(sig))
    #scatter(sig,scan_eye(sig))