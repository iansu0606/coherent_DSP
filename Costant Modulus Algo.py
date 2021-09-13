import numpy as np
from tqdm import trange
import numba as nb
from pathlib import Path
import os
import yaml
from optparse import OptionParser

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
    stepping = 1
    rx_x.resize((batchnum, batchsize), refcheck=False)
    rx_y.resize((batchnum, batchsize), refcheck=False)
    stepsize = [stepsize_list[step_index], stepsize_list[step_index]]
    record = np.zeros((100, 30), dtype="complex")
    prec = np.zeros((30, 1), dtype="float32")
    step_x, step_y = stepsize[0], stepsize[1]
    cost_x = np.zeros((1, iterator), dtype="complex_")
    cost_y = np.zeros((1, iterator), dtype="complex_")
    # initialize H
    inputrx = rx_x_single
    inputry = rx_y_single
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

    # approximater(iterator, center, datalength, inputrx, inputry,exout, eyout, error_x, error_y,
    #           step_x, step_y, cost_x, cost_y, record, satisfaction, stepsize_list)

    return iterator, center, datalength, inputrx, inputry,exout, eyout, error_x, error_y, step_x, step_y, cost_x, cost_y, record, satisfaction

@nb.jit(cache=True, nopython=True, parallel=True, nogil=True)
def approximater(iterator, center, datalength, inputrx, inputry, exout, eyout, hxx, hxy, hyx, hyy, error_x, error_y,
              step_x, step_y, cost_x, cost_y, stepsize_list, step_index, satisfaction, ):
    R = [2, 10, 18]
    for it in trange(iterator):
        for indx in nb.prange(center, datalength - center):
            exout[indx] = np.matmul(hxx, inputrx[indx - center:indx + center + 1]) + np.matmul(
                hxy, inputry[indx - center:indx + center + 1])
            eyout[indx] = np.matmul(hyx, inputrx[indx - center:indx + center + 1]) + np.matmul(
                hyy, inputry[indx - center:indx + center + 1])
            if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))
            xdistance = [np.abs(np.abs(exout[indx]) - R[0] ** 0.5), np.abs(np.abs(exout[indx]) - R[1] ** 0.5),
                         np.abs(np.abs(exout[indx]) - R[2] ** 0.5)]
            ydistance = [np.abs(np.abs(eyout[indx]) - R[0] ** 0.5), np.abs(np.abs(eyout[indx]) - R[1] ** 0.5),
                         np.abs(np.abs(eyout[indx]) - R[2] ** 0.5)]
            error_x[indx] = np.abs(exout[indx]) ** 2 - R[np.argmin(xdistance)]
            error_y[indx] = np.abs(eyout[indx]) ** 2 - R[np.argmin(ydistance)]
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
        print('X_Pol Cost:{:.2f}'.format(cost_x[0][it]), 'Y_pol Cost:{:.2f}'.format(cost_y[0][it]))
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
                step_y =  stepsize_list[step_index]
                # if step_y <= 1E-8:
                #     step_y = 1E-8
                print('Y Stepsize adjust to {:.2E}'.format(step_y))
    return

def source(folder):
    for file_name in os.listdir(folder):
        source = folder + '/' + file_name
        case = os.path.basename(source)[-6:-4]
        if case == 'XI' or 'xi':
            destination = datafolder + r'\RxXI.txt'
        elif case == 'XQ' or 'xq':
            destination = datafolder + r'\RxXQ.txt'
        elif case == 'YI' or 'yi':
            destination = datafolder + r'\RxYI.txt'
        elif case == 'YQ' or 'yq':
            destination = datafolder + r'\RxYQ.txt'



if __name__ == '__main__':
    # initial Parameter
    parser = OptionParser()
    parser.add_option('--input', dest="input", help="input file", default='./behave_config.yml')
    (options, args) = parser.parse_args()
    # load config data from yaml file
    yml_cfg = {}
    if options.input != '':
        yml_cfg = yaml.safe_load(open(options.input))['Behave_Sim']
    else:
        raise Exception('Error: Input file is required')

    route = os.path.abspath(os.getcwd())
    cur_dir = ''.join([route, '/', 'image'])
    Path(cur_dir).mkdir(parents=True, exist_ok=True)

    parameter = Parameter(simulation=False)
    print('SymbolRate={}GBaud'.format(parameter.symbolRate / 1e9), 'Pamorder={}'.format(parameter.pamorder),
          'resamplenumber={}'.format(parameter.resamplenumber))