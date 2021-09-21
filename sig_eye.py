"""
@author: Ian Su
Contact: suanyu@amazon.com
"""
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path
import os
import yaml
from tqdm import trange

# TODO: add more function
def sample_more(input_signal, rate):
    signal_slice = input_signal[1000:2000]
    signal_slice -= np.mean(signal_slice)
    signal_slice *= (2/np.mean(np.abs(signal_slice)))
    # the first 8 is because we have to keep integer
    signal_eye = signal.resample_poly(signal_slice, up=(53.125 * 8 * rate), down=92* 8)
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


def scatter(input_signal, index, rate):
    plt.scatter([i for i in range(len(input_signal[index::rate]))], input_signal[index::rate])
    ax = plt.gca()
    ax.set(ylim=(-4, 4))
    plt.yticks(np.arange(-4, 4, step=1))
    ax.set_title('Scatter Diagram', FontSize=20, FontWeight="bold")
    ax.set_xlabel('time (ps)', FontSize=16, FontWeight="bold")
    ax.set_ylabel('Amplitude (a.u.)', FontSize=16, FontWeight="bold")
    return


def eye_plot(input_signal, offset, rate):
    for iterate in range(0, 150):
        signal_draw = input_signal[offset+iterate*rate*2:offset+(iterate+1)*rate*2]
        plt.plot(signal_draw, 'y')
    ax = plt.gca()
    ax.set(ylim=(-5, 5))
    ax.set_facecolor('k')
    ax.grid('w')
    ax.set_title('Eye diagram of Data: offset = {}'.format(offset), FontSize=20, FontWeight="bold")
    ax.set_xlabel('time (ps)', FontSize=16, FontWeight="bold")
    ax.set_ylabel('Amplitude (a.u.)', FontSize=16, FontWeight="bold")
    return


def hard_bitstream(input_signal):
    bitstream = np.ones(len(input_signal))
    padding = np.zeros(len(input_signal))
    padding[input_signal > -2] = 1
    # Padding as a filter
    bitstream[np.where(input_signal[np.where(padding)]<0)] = -1
    bitstream[input_signal > 2] = 3
    bitstream[input_signal < -2] = -3
    return bitstream


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


def video_maker(path_chart):
    import cv2
    image_folder = path_chart
    video_name = 'video.avi'
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")],
                    key=lambda x: (int(x[0:2])))
    # print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    return


def optimize_plot(sig, sample_per_symbol, index, path_chart):
    plt.figure(1)
    eye_plot(sig, index, sample_per_symbol)
    plt.savefig(path_chart + '/' + "eye_optimized" + ".png")
    plt.close(2)
    plt.figure(2)
    scatter(sig, index, sample_per_symbol)
    plt.savefig(path_chart + '/' + "scatter_optimized" + ".png",
                facecolor='w', edgecolor='w')
    plt.close(2)
    return


def scan(sig, sample_per_symbol, index, path_chart):
    for k in range(1, sample_per_symbol):
        plt.figure(k)
        eye_plot(sig, k, sample_per_symbol)
        plt.savefig(path_chart + '/' + "eye_" + str(k) + ".png")
        plt.close(k)

    for k in range(1, sample_per_symbol):
        plt.figure(k)
        scatter(sig, index, sample_per_symbol)
        plt.savefig(path_chart + '/' + str(k) + " _scatter" + ".png",
                    facecolor='w', edgecolor='w')
        plt.close(k)
    return


def data_extract(input_file):
    datum = None
    if input_file.endswith(".txt"):
        datum = np.genfromtxt(input_file)[:, 0]
    elif input_file.endswith(".mat"):
        import scipy.io as sio
        datum = sio.loadmat(input_file)
    return datum


def pam4_bitcombiner(refer_bitstream):
    # Find the Gray encoding level
    bitstream = np.ones(len(refer_bitstream))
    index3 = [i for i, j in enumerate(refer_bitstream) if np.all(j==[1,0])]
    index1 = [i for i, j in enumerate(refer_bitstream) if np.all(j==[1,1])]
    index_1 = [i for i, j in enumerate(refer_bitstream) if np.all(j==[0,1])]
    index_3 = [i for i, j in enumerate(refer_bitstream) if np.all(j==[0,0])]
    # Transfer them into amplitude level
    bitstream[index3] = 3
    bitstream[index_1] = -1
    bitstream[index_3] = -3
    return bitstream



def correlation_extract(referenced_signal, input_signal):
    from numpy import corrcoef
    from numpy import append as app
    from numpy import max as maxim
    #, window_length, similarity, prbs_num
    # Reference,input signal reshape to one-dimensional
    referenced_signal = np.reshape(referenced_signal, -1)
    input_signal = np.reshape(input_signal, -1)
    received_bitstream = hard_bitstream(input_signal)
    """
    try:
        assert len(referenced_signal) == len(input_signal), '參考する長さ[{0}], 入力長さ[{1}]'\
            .format(len(referenced_signal),len(input_signal) )
    except AssertionError as err:
        print('長さは同じはず', err)
    """
    correlation_list = np.array([])
    offset = 1
    while 1:
        for window in trange(0,32000,1):
            corr_temp = corrcoef(received_bitstream[offset:500+offset],
                             referenced_signal[window:500+window])
            if corr_temp[0][1] > 0.9:
                print("window = : {}".format(window))
                break
            correlation_list = app(correlation_list,corr_temp[0][1])
        print("correlation: {}".format(maxim(correlation_list)))
        if maxim(correlation_list)>0.9:
            print("offset = : {}".format(offset))
            break
        offset +=1
        correlation_list = np.array([])
    return correlation_list

def progressing(file, sample_per_symbol, is_mat, is_fig, is_video, is_pam4):
    data = data_extract(file)
    if is_mat:
        if is_keysight:
        # Rx_side
            data_XI = np.reshape(data["Y1"]["Values"][0][0][0],-1)
            # -1 refers to flatten it to 1-D array
            data_XQ = np.reshape(data["Y1"]["Values"][0][1][0],-1)
            data_YI = np.reshape(data["Y2"]["Values"][0][2][0],-1)
            data_YQ = np.reshape(data["Y2"]["Values"][0][3][0],-1)

        else:
            # (Pseudo_Tx-->refer)
            data_XI = np.reshape(data["Vblock"]["Values"][0][0][0], -1)
            # -1 refers to flatten it to 1-D array
            data_XQ = np.reshape(data["Vblock"]["Values"][0][1][0], -1)
            data_YI = np.reshape(data["Vblock"]["Values"][0][2][0], -1)
            data_YQ = np.reshape(data["Vblock"]["Values"][0][3][0], -1)
            refer_XI = np.reshape(data["zXSym"]["SeqRe"][0][0][:],-1)
            refer_XQ = np.reshape(data["zXSym"]["SeqIm"][0][0][:],-1)
            refer_YI = np.reshape(data["zYSym"]["SeqRe"][0][0][:],-1)
            refer_YQ = np.reshape(data["zYSym"]["SeqIm"][0][0][:],-1)
        #Go to the resample part
        sig = sample_more(data_XI, sample_per_symbol)
    else:
        sig = sample_more(data, sample_per_symbol)

    if is_pam4:
        # order F stands for Transfering to Gray encoding
        refer_XI = np.reshape(refer_XI, (-1, 2), order='F')
        refer_XQ = np.reshape(refer_XQ, (-1, 2), order='F')
        refer_YI = np.reshape(refer_YI, (-1, 2), order='F')
        refer_YQ = np.reshape(refer_YQ, (-1, 2), order='F')


    """ optimize the SNR"""
    index, snr = scan_eye(sig, sample_per_symbol)
    print("snr={} dB".format(snr[index]))
    print("index={}".format(index))
    """ path to work directory"""
    route = os.path.abspath(os.getcwd())
    path_chart = route + "/images/"
    Path(path_chart).mkdir(parents=True, exist_ok=True)
    """ plot all the scanning diagram"""
    if is_fig:
        optimize_plot(sig, sample_per_symbol, index, path_chart)
        print("===============Done Plotting===============")
    """ path to work directory"""
    if is_video:
        scan(sig, sample_per_symbol, index, path_chart)
        video_maker(path_chart)
        print("======Done Video=========")
    return

def time_analyzer(eye_sig):
    level_0 = 0.008797
    level_1 = 0.01211
    threshold_20 = level_0+0.2*(level_1-level_0)
    threshold_80 = level_0+0.8*(level_1-level_0)
    time_20 = [time for time, signal in enumerate(eye_sig[:,1]) if abs(signal-threshold_20) < 0.0001]
    time_80 = [time for time, signal in enumerate(eye_sig[:,1]) if abs(signal-threshold_80) < 0.0001]
    rise_time = eye_sig[time_80[0]]-eye_sig[time_20[0]]
    fall_time = eye_sig[time_20[2]]-eye_sig[time_80[1]]
    return rise_time, fall_time


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
    # start to progressing
    # progressing(file, sample_per_symbol, is_mat, is_fig, is_video, is_pam4)

    data = data_extract(file)
    if is_mat:
        if is_keysight:
            # -1 refers to flatten it to 1-D array
            data_XI = np.reshape(data["Vblock"]["Values"][0][0][0], -1)
            data_XQ = np.reshape(data["Vblock"]["Values"][0][1][0], -1)
            data_YI = np.reshape(data["Vblock"]["Values"][0][2][0], -1)
            data_YQ = np.reshape(data["Vblock"]["Values"][0][3][0], -1)
        # Rx_side
        else:
            data_XI = np.reshape(data["Vblock"]["Values"][0][0][0],-1)
            # -1 refers to flatten it to 1-D array
            data_XQ = np.reshape(data["Vblock"]["Values"][0][1][0],-1)
            data_YI = np.reshape(data["Vblock"]["Values"][0][2][0],-1)
            data_YQ = np.reshape(data["Vblock"]["Values"][0][3][0],-1)
            # (Pseudo_Tx-->refer)
            refer_XI = np.reshape(data["zXSym"]["SeqRe"][0][0][:],-1)
            refer_XQ = np.reshape(data["zXSym"]["SeqIm"][0][0][:],-1)
            refer_YI = np.reshape(data["zYSym"]["SeqRe"][0][0][:],-1)
            refer_YQ = np.reshape(data["zYSym"]["SeqIm"][0][0][:],-1)
            #Go to the resample part
        sig = sample_more(data_XI, sample_per_symbol)
    else:
        sig = sample_more(data, sample_per_symbol)
    """
    if is_pam4:
        # order F stands for Transfering to Gray encoding
        refer_XI = np.reshape(refer_XI, (-1, 2), order='F')
        refer_XQ = np.reshape(refer_XQ, (-1, 2), order='F')
        refer_YI = np.reshape(refer_YI, (-1, 2), order='F')
        refer_YQ = np.reshape(refer_YQ, (-1, 2), order='F')

    """
    index, snr = scan_eye(sig, sample_per_symbol)
    #print("snr_list={} dB".format(snr))
    print("snr={} dB".format(snr[index]))
    print("index={}".format(index))
    # Chart
    route = os.path.abspath(os.getcwd())
    path_chart = route + "/images/"
    Path(path_chart).mkdir(parents=True, exist_ok=True)
    """ plot all the scanning diagram"""
    if is_fig:
        optimize_plot(sig, sample_per_symbol, index, path_chart)
        print("===============Done Plotting===============")
    #referred_bitstream = pam4_bitcombiner(refer_XI)
    #score = correlation_extract(referred_bitstream, data_XI[index::sample_per_symbol])