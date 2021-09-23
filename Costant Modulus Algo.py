import numpy as np
from tqdm import trange
import numba as nb
from pathlib import Path
import os
import yaml
from optparse import OptionParser
import pandas as pd
import scipy.io as sio
import plotly.graph_objects as go
import scipy.signal as signal


def Histogram2D_Hot(filename, data, SNR=0, EVM=0, bercount=(0, 0, 0), path=None):
    x = np.real(data)
    y = np.imag(data)
    miny = y.min()
    fig = go.Figure()
    filename = str(filename)
    fig.add_trace(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color='#F58518'
        )
    ))
    fig.add_trace(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color='#F58518'
        )
    ))
    fig.add_trace(go.Histogram2d(
        x=x,
        y=y,
        colorscale='Hot',
        nbinsx=256,
        nbinsy=256,
        zauto=True
    ))
    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.9],
            showgrid=False,
            fixedrange=True,
            title="In-Phase",
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.9],
            showgrid=False,
            fixedrange=True,
            title="Quadrature-Phase",
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.905, 1],
            showgrid=False,
            fixedrange=True
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.905, 1],
            showgrid=False,
            fixedrange=True
        ),
        height=800,
        width=800,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        title=go.layout.Title(text="Color Histogram---" + filename),
        font=dict(
            family="Arial",
            size=20,
            color="Black"),
        # yaxis_range=[-4, 4],
        # xaxis_range=[-4,4]
    )
    if SNR != 0:
        print("SNR = {:.2f}(dB)".format(SNR))
        print("EVM = {:.2f}(%)".format(EVM))
        print("bercount = {:.2E}".format(bercount[0]))
        fig.update_layout(
            xaxis=dict(
                zeroline=False,
                domain=[0, 0.9],
                showgrid=False,
                fixedrange=True,
                title='In-Phase<br>SNR:{:.2f}(dB) || EVM:{:.2f}(%) || Bercount:{:.2E} [{}/{}]'.format(SNR, EVM * 100,
                                                                                                      bercount[0],
                                                                                                      bercount[1],
                                                                                                      bercount[2])
            ),
            font=dict(
                family="Arial",
                size=20,
                color="Black"),
        )
    if path:
        print(path + r"\{}.png".format(filename))
        fig.write_image(path + r"\{}.png".format(filename))
    else:
        fig.write_image(
            "/Users/suanyu/Documents/DSP/coherent/16QAM/image/{}.png".format(filename))
    return


@nb.jit(cache=True, nopython=True, parallel=True, nogil=True)
def initer(rx_x, rx_y, taps, iteration):
    rx_x_single = np.array(rx_x)
    rx_y_single = np.array(rx_y)

    rx_x = np.array(rx_x)
    rx_y = np.array(rx_y)

    datalength = len(rx_x)
    stepsize_list = [1e-3, 1e-4, 1e-5, 6e-6, 1e-6, 1e-10]

    batchsize = datalength
    overhead = 0.95
    step_index = 2
    trainlength = round(batchsize * overhead)
    cmataps = taps
    center = int(np.max(cmataps) // 2)
    batchnum = int(datalength / batchsize)
    iterator = iteration
    satisfaction = 0
    # stepping = 1
    rx_x.resize((batchnum, batchsize), refcheck=False)
    rx_y.resize((batchnum, batchsize), refcheck=False)
    stepsize = [stepsize_list[step_index], stepsize_list[step_index]]
    # record = np.zeros((100, 30), dtype="complex")
    # prec = np.zeros((30, 1), dtype="float32")
    step_x, step_y = stepsize[0], stepsize[1]
    cost_x = np.zeros((1, iterator), dtype="complex_")
    cost_y = np.zeros((1, iterator), dtype="complex_")

    inputrx = rx_x_single
    inputry = rx_y_single
    # initialize H
    hxx = np.zeros(cmataps, dtype="complex_")
    hxy = np.zeros(cmataps, dtype="complex_")
    hyx = np.zeros(cmataps, dtype="complex_")
    hyy = np.zeros(cmataps, dtype="complex_")
    hxx[center] = 1
    hyy[center] = 1
    exout = np.zeros(datalength, dtype="complex_")
    eyout = np.zeros(datalength, dtype="complex_")
    error_x = np.zeros(datalength, dtype="complex_")
    error_y = np.zeros(datalength, dtype="complex_")
    # hyy = np.conj(hxx)
    # hxy = np.conj(hyx)
    # rx_x_single = exout[center:-center]
    # rx_y_single = eyout[center:-center]
    return approximater(iterator, center, datalength, inputrx, inputry, exout, eyout, hxx, hxy, hyx, hyy, error_x, error_y,
                        step_x, step_y, cost_x, cost_y, stepsize_list, satisfaction)


@nb.jit(cache=True, nopython=True, parallel=True, nogil=True)
def approximater(iterator, center, datalength, inputrx, inputry, exout, eyout, hxx, hxy, hyx, hyy, error_x, error_y,
                 step_x, step_y, cost_x, cost_y, stepsize_list, step_index, satisfaction):
    R = [2, 10, 18]
    for it in trange(iterator):
        for indx in nb.prange(center, datalength - center):
            exout[indx] = np.matmul(hxx, inputrx[indx - center:indx + center + 1]) + np.matmul(
                hxy, inputry[indx - center:indx + center + 1])
            eyout[indx] = np.matmul(hyx, inputrx[indx - center:indx + center + 1]) + np.matmul(
                hyy, inputry[indx - center:indx + center + 1])
            if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                raise Exception(
                    "CMA Equaliser didn't converge at iterator {}".format(it))
            xdistance = [np.abs(np.abs(exout[indx]) - R[0] ** 0.5), np.abs(np.abs(exout[indx]) - R[1] ** 0.5),
                         np.abs(np.abs(exout[indx]) - R[2] ** 0.5)]
            ydistance = [np.abs(np.abs(eyout[indx]) - R[0] ** 0.5), np.abs(np.abs(eyout[indx]) - R[1] ** 0.5),
                         np.abs(np.abs(eyout[indx]) - R[2] ** 0.5)]
            error_x[indx] = np.abs(exout[indx]) ** 2 - R[np.argmin(xdistance)]
            error_y[indx] = np.abs(eyout[indx]) ** 2 - R[np.argmin(ydistance)]
            # Update weight function
            hxx = hxx - step_x * error_x[indx] * exout[indx] * np.conj(inputrx[
                indx - center:indx + center + 1])
            hxy = hxy - step_x * error_x[indx] * exout[indx] * np.conj(inputry[
                indx - center:indx + center + 1])
            hyx = hyx - step_y * error_y[indx] * eyout[indx] * np.conj(inputrx[
                indx - center:indx + center + 1])
            hyy = hyy - step_y * error_y[indx] * eyout[indx] * np.conj(inputry[
                indx - center:indx + center + 1])
        cost_x[0][it] = np.mean((error_x[center:]) ** 2)
        cost_y[0][it] = np.mean((error_y[center:]) ** 2)
        print('X_Pol Cost:{:.2f}'.format(
            cost_x[0][it]), 'Y_pol Cost:{:.2f}'.format(cost_y[0][it]))
        if it > 1:
            costmeanx = np.mean(cost_x[0][it - 1:it])
            costmeany = np.mean(cost_y[0][it - 1:it])
            if np.abs(cost_x[0][it] - costmeanx) < satisfaction * \
                    cost_x[0][it] and np.abs(
                cost_y[0][it] - costmeany) < satisfaction * \
                    cost_y[0][it]:
                print("Fulfilled satisfaction at iterator {}".format(it))
                break
            if np.abs(cost_x[0][it] - costmeanx) < 0.1:
                step_index += 1
                step_x = stepsize_list[step_index]
                # if step_x <= 1E-8:
                #     step_x = 1E-8
                print('X Stepsize adjust to {:.2E}'.format(step_x))

            if np.abs(cost_y[0][it] - costmeany) < 0.1:
                step_index += 1
                step_y = stepsize_list[step_index]
                # if step_y <= 1E-8:
                #     step_y = 1E-8
                print('Y Stepsize adjust to {:.2E}'.format(step_y))
    rx_x_single = exout[center:-center]
    rx_y_single = eyout[center:-center]
    return rx_x_single, rx_y_single


@nb.jit(nopython=True, cache=True)
def source(simulation, datafile, symbol_rate, pamorder, datafolder, samplepersymbol=None, timespan=None):
    if simulation:
        datalength = int(symbol_rate * samplepersymbol * timespan)
        RxXI = pd.read_table(
            datafolder + 'RxXI.txt', names=['RxXI'], header=9)['RxXI'].tolist()[-datalength:]
        RxXQ = pd.read_table(
            datafolder + 'RxXQ.txt', names=['RxXQ'], header=9)['RxXQ'].tolist()[-datalength:]
        RxYI = pd.read_table(
            datafolder + 'RxYI.txt', names=['RxYI'], header=9)['RxYI'].tolist()[-datalength:]
        RxYQ = pd.read_table(
            datafolder + 'RxYQ.txt', names=['RxYQ'], header=9)['RxYQ'].tolist()[-datalength:]
        TxXI = np.reshape(
            np.append(pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxXI1.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                      pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxXI2.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            order='F')
        TxXQ = np.reshape(
            np.append(pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxXQ1.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                      pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxXQ2.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            order='F')
        TxYI = np.reshape(
            np.append(pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxYI1.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                      pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxYI2.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            order='F')
        TxYQ = np.reshape(
            np.append(pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxYQ1.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy(),
                      pd.read_table(datafolder + r'\experiment_data\QAM16_53GBaud_simulation' + r'\PRBS13/LogTxYQ2.txt',
                                    header=5, names=['PRBS'], skiprows=0)['PRBS'].to_numpy()), (-1, 2),
            order='F')
    else:
        data = sio.loadmat(datafile)
        RxXI = data["Vblock"]["Values"][0][0][0]
        RxXQ = data["Vblock"]["Values"][0][1][0]
        RxYI = data["Vblock"]["Values"][0][2][0]
        RxYQ = data["Vblock"]["Values"][0][3][0]
        if pamorder == 4:
            TxXI = data["zXSym"]["SeqRe"][0][0][:]
            TxXI = np.reshape(np.reshape(TxXI, -1), (-1, 2), order='F')
            TxXQ = data["zXSym"]["SeqIm"][0][0][:]
            TxXQ = np.reshape(np.reshape(TxXQ, -1), (-1, 2), order='F')
            TxYI = data["zYSym"]["SeqRe"][0][0][:]
            TxYI = np.reshape(np.reshape(TxYI, -1), (-1, 2), order='F')
            TxYQ = data["zYSym"]["SeqIm"][0][0][:]
            TxYQ = np.reshape(np.reshape(TxYQ, -1), (-1, 2), order='F')
        else:
            TxXI = data["zXSym"]["SeqRe"][0][0][0].tolist()
            TxXQ = data["zXSym"]["SeqIm"][0][0][0].tolist()
            TxYI = data["zYSym"]["SeqRe"][0][0][0].tolist()
            TxYQ = data["zYSym"]["SeqIm"][0][0][0].tolist()
    return RxXI, RxXQ, RxYI, RxYQ, TxXI, TxXQ, TxYI, TxYQ


def Tx2Bit(Isig, Qsig, PAM_order):
    Tx_symbol = []
    seqlength = len(Isig)
    if PAM_order == 2:  # level[-1, 1]
        Ithreshold = np.mean(Isig)
        Qthreshold = np.mean(Qsig)
        for indx in range(seqlength):
            if Isig[indx] > Ithreshold and Qsig[indx] > Qthreshold:
                Tx_symbol.append(1 + 1j)
            if Isig[indx] > Ithreshold and Qsig[indx] < Qthreshold:
                Tx_symbol.append(1 - 1j)
            if Isig[indx] < Ithreshold and Qsig[indx] > Qthreshold:
                Tx_symbol.append(-1 + 1j)
            if Isig[indx] < Ithreshold and Qsig[indx] < Qthreshold:
                Tx_symbol.append(-1 - 1j)
    if PAM_order == 4:
        Tx = np.concatenate((Isig, Qsig), axis=1)
        symboldic = {'[0, 0, 0, 0]': -3 + 3j, '[0, 0, 0, 1]': -1 + 3j, '[0, 0, 1, 1]': 3 + 3j, '[0, 0, 1, 0]': 1 + 3j,
                     '[0, 1, 0, 0]': -3 + 1j, '[0, 1, 0, 1]': -1 + 1j, '[0, 1, 1, 1]': 3 + 1j, '[0, 1, 1, 0]': 1 + 1j,
                     '[1, 1, 0, 0]': -3 - 3j, '[1, 1, 0, 1]': -1 - 3j, '[1, 1, 1, 1]': 3 - 3j, '[1, 1, 1, 0]': 1 - 3j,
                     '[1, 0, 0, 0]': -3 - 1j, '[1, 0, 0, 1]': -1 - 1j, '[1, 0, 1, 1]': 3 - 1j, '[1, 0, 1, 0]': 1 - 1j}
        for indx in range(seqlength):
            Tx_symbol.append(symboldic[str(Tx[indx][:].tolist())])
    return Tx_symbol


def DataNormalize(seqr, seqi, PAM_order):

    if len(seqi) == 0:
        seqr = np.array(seqr)
        rmean = np.mean(seqr)
        rlen = len(seqr)
        normalseqr = []
        r_shift = seqr - rmean
        factor_amplitude_r = np.mean(abs(r_shift))*2/PAM_order
        normalseq = r_shift / factor_amplitude_r
        return normalseq
    else:
        seqr = np.array(seqr)
        seqi = np.array(seqi)
        rmean = np.mean(seqr)
        imean = np.mean(seqi)
        rlen = len(seqr)
        ilen = len(seqi)
        assert rlen == ilen, 'Length of I & Q should equal'
        normalseqr, normalseqi = [], []
        r_shift = seqr - rmean
        i_shift = seqi - imean
        factor_amplitude_r = np.mean(abs(r_shift)) * 2 / PAM_order
        factor_amplitude_i = np.mean(abs(i_shift)) * 2 / PAM_order
        normalseqr = r_shift / factor_amplitude_r
        normalseqi = i_shift / factor_amplitude_i
        return normalseqr, normalseqi


def CMA_iterate(eye, Rx_XI, Rx_XQ, Rx_YI, Rx_YQ):
    #
    snrx, snry = np.zeros((parameter['downsample'], 1)), np.zeros(
        (parameter['downsample'], 1))
    evmx, evmy = np.zeros((parameter['downsample'], 1)), np.zeros(
        (parameter['downsample'], 1))
    # Eye Position
    for eyepos in nb.prange(eye['start'], eye['end']):
        Rx_XI_eye = DataNormalize(signal.resample_poly(Rx_XI[eyepos:], up=1, down=parameter['downsample']), [],
                                  parameter['pamorder'])
        Rx_XQ_eye = DataNormalize(signal.resample_poly(Rx_XQ[eyepos:], up=1, down=parameter['downsample']), [],
                                  parameter['pamorder'])
        Rx_YI_eye = DataNormalize(signal.resample_poly(Rx_YI[eyepos:], up=1, down=parameter['downsample']), [],
                                  parameter['pamorder'])
        Rx_YQ_eye = DataNormalize(signal.resample_poly(Rx_YQ[eyepos:], up=1, down=parameter['downsample']), [],
                                  parameter['pamorder'])
        Rx_Signal_X = Rx_XI_eye + 1j * Rx_XQ_eye
        Rx_Signal_Y = Rx_YI_eye + 1j * Rx_YQ_eye
        Histogram2D_Hot('Rx_X_{}'.format(eyepos),
                        Rx_Signal_X[0:100000], path=cur_dir)
        Histogram2D_Hot('Rx_Y_{}'.format(eyepos),
                        Rx_Signal_Y[0:100000], path=cur_dir)

        for tap in nb.prange(33, 35, 2):
            cma = initer(Rx_Signal_X[0:100000],
                         Rx_Signal_Y[0:100000], taps=tap, iter=30)
            cma.RDE_butterfly()
            Rx_X_CMA, Rx_Y_CMA = cma.rx_x_single, cma.rx_y_single
            print('CMA Batch Size={}'.format(cma.batchsize), 'CMA Stepsize={}'.format(cma.stepsize),
                  'CMA OverHead={}%'.format(cma.overhead * 100))
            Histogram2D_Hot('CMAX_{}(taps{})'.format(
                eyepos, tap), Rx_X_CMA, path=cur_dir)
            Histogram2D_Hot('CMAY_{}(taps{})'.format(
                eyepos, tap), Rx_Y_CMA, path=cur_dir)


if __name__ == '__main__':
    # initial Parameter
    parser = OptionParser()
    parser.add_option('--input', dest="input",
                      help="input file", default='./behave_config.yml')
    (options, args) = parser.parse_args()
    # load config data from yaml file
    yml_cfg = {}
    if options.input != '':
        yml_cfg = yaml.safe_load(open(options.input))['Behave_Sim']
    else:
        raise Exception('Error: Input file is required')

    route = os.path.abspath(os.getcwd())
    cur_dir = ''.join([route, '../', yml_cfg['output']])
    Path(cur_dir).mkdir(parents=True, exist_ok=True)

    is_sim = yml_cfg['is_simulation']
    datafolder = yml_cfg['folder']
    eye = yml_cfg['eye']
    if is_sim:
        parameter = yml_cfg['parameter']
    else:
        parameter = yml_cfg['parameter_exp']

    # Current Parameter
    print('SymbolRate= {} GBaud \n'.format(parameter['symbol_rate']),
          'Pamorder = {} \n'.format(parameter['pamorder']),
          'resample_rate = {} \n'.format(parameter['sample_times']))
    RxXI, RxXQ, RxYI, RxYQ, TxXI, TxXQ, TxYI, TxYQ = source(is_sim,
                                                            parameter['data_file'],
                                                            parameter['symbol_rate'],
                                                            parameter['sample_rate'],
                                                            parameter['pamorder'], datafolder)

    Tx_Signal_X = np.array(Tx2Bit(TxXI, TxXQ, parameter['pamorder']))
    Tx_Signal_Y = np.array(
        Tx2Bit(parameter.TxYI_prbs, parameter.TxYQ_prbs, parameter.pamorder))
    Histogram2D_Hot("TX_X", Tx_Signal_X, path=cur_dir)
    Histogram2D_Hot("TX_Y", Tx_Signal_Y, path=cur_dir)
    # Data Normalize
    Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(RxXI, up=parameter['upsample'], down=1),
                                 signal.resample_poly(
                                     RxXQ, up=parameter['upsample'], down=1),
                                 parameter['pamorder'])
    Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(parameter.RxYI, up=parameter.upsamplenum, down=1),
                                 signal.resample_poly(
                                     parameter.RxYQ, up=parameter.upsamplenum, down=1),
                                 parameter.pamorder)
    Histogram2D_Hot("Rx_X(upsample)",
                    Rx_XI[0:100000] + 1j * Rx_XQ[0:100000], path=cur_dir)
    #Histogram2D_Hot("Rx_Y(upsample)", Rx_YI[0:100000] + 1j * Rx_YQ[0:100000], path=cur_dir)
    CMA_iterate(eye, Rx_XI, Rx_XQ, Rx_YI, Rx_YQ)
