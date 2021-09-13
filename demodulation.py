import yaml
from optparse import OptionParser
import scipy.signal as signal
from subfunction.DataNormalize import *
from subfunction.Histogram2D import *
from CMA import *
from Sourcing_txt import *
import os
import openpyxl
from openpyxl import Workbook

# open_excel(address)
def log(inputs):
# 這邊缺了輸入的資料
    lg_cfg = {}
    lg_cfg = yaml.safe_load(open(inputs))['Log']
    output = lg_cfg['logfile']
    wb = openpyxl.load_workbook(output)



def decoding(parameter, inputs):
    yml_cfg = {}
    yml_cfg = yaml.safe_load(open(inputs))['Param']
    output = yml_cfg['output_folder']
    isplot = yml_cfg['isplot']
    iswrite = yml_cfg['iswrite']
    xpart = yml_cfg['xpart']
    ypart = yml_cfg['ypart']
    eyestart = yml_cfg['eyestart']
    eyeend = yml_cfg['eyeend']
    eyescan = yml_cfg['eyescan']
    tap1_start = yml_cfg['tap1_start']
    tap1_end = yml_cfg['tap1_end']
    tap1_scan = yml_cfg['tap1_scan']
    tap2_start = yml_cfg['tap2_start']
    tap2_end = yml_cfg['tap2_end']
    tap2_scan = yml_cfg['tap2_scan']
    cma_stage = yml_cfg['cma_stage']
    cma_iter = yml_cfg['cma_iter']
    radii = yml_cfg['radii']
    isrealvolterra = yml_cfg['isrealvolterra']
    iscomplexvolterra = yml_cfg['iscomplexvolterra']
    window_length = yml_cfg['window_length']
    correlation_length = yml_cfg['correlation_length']
    final_length = correlation_length - 9000
    CMAstage1_tap = yml_cfg['CMAstage1_tap']
    CMAstage1_stepsize_x = yml_cfg['CMAstage1_stepsize_x']
    CMAstage1_iteration = yml_cfg['CMAstage1_iteration']
    CMAstage2_tap = yml_cfg['CMAstage2_tap']
    CMAstage2_stepsize_x = yml_cfg['CMAstage2_stepsize_x']
    CMAstage2_iteration = yml_cfg['CMAstage2_iteration']
    CMA_cost_X1 = yml_cfg['CMA_cost_X1']
    CMA_cost_X2 = yml_cfg['CMA_cost_X2']
    CMA_cost_Y1 = yml_cfg['CMA_cost_Y1']
    CMA_cost_Y2 = yml_cfg['CMA_cost_Y2']
    XIshift = yml_cfg['XIshift']
    XQshift = yml_cfg['XQshift']
    XI_corr = yml_cfg['XI_corr']
    XQ_corr = yml_cfg['XQ_corr']
    SNR_X = yml_cfg['SNR_X']
    EVM_X = yml_cfg['EVM_X']
    bercount_X = yml_cfg['bercount_X']
    YIshift = yml_cfg['YIshift']
    YQshift = yml_cfg['YQshift']
    YI_corr = yml_cfg['YI_corr']
    YQ_corr = yml_cfg['YQ_corr']
    SNR_Y = yml_cfg['SNR_Y']
    EVM_Y = yml_cfg['EVM_Y']
    bercount_Y = yml_cfg['bercount_Y']
    r1 = yml_cfg['r1']
    r2 = yml_cfg['r2']
    Imageaddress = output + 'image'
    try:
        os.mkdir(Imageaddress)
    except:
        pass
    ##############################################################################################################################
    print("symbolrate = {}Gbit/s\n"
          "pamorder = {}\n"
          "resamplenumber = {}".format(float(parameter.symbolRate) / 1e9, parameter.pamorder, parameter.resamplenumber))
    # Tx2Bit = KENG_Tx2Bit(PAM_order=parameter.pamorder)
    # downsample_Tx = KENG_downsample(down_coeff=parameter.resamplenumber)
    # downsample_Rx = KENG_downsample(down_coeff=parameter.resamplenumber)

    # Tx_XI, Tx_XQ = DataNormalize(parameter.TxXI, parameter.TxXQ, parameter.pamorder)
    # Tx_YI, Tx_YQ = DataNormalize(parameter.TxYI, parameter.TxYQ, parameter.pamorder)
    # TxXI = Tx2Bit.return_Tx(Tx_XI)
    # TxXQ = Tx2Bit.return_Tx(Tx_XQ)
    # TxYI = Tx2Bit.return_Tx(Tx_YI)
    # TxYQ = Tx2Bit.return_Tx(Tx_YQ)
    #
    # Tx_Signal_X = TxXI[:, 0] + 1j * TxXQ[:, 0]
    # Tx_Signal_Y = TxYI[:, 0] + 1j * TxYQ[:, 0]

    # RxX_signal = np.array(parameter.RxXI) + 1j * np.array(parameter.RxXQ)
    # RxY_signal = np.array(parameter.RxYI) + 1j * np.array(parameter.RxYQ)
    #
    # cd_compensator = CD_compensator(RxX_signal, RxY_signal, Gbaud=28.125e9 * parameter.upsamplenum, KM=100)  # 39.62
    # CD_X, CD_Y = cd_compensator.overlap_save(Nfft=len(parameter.RxXQ), NOverlap=128)  # 4096
    # Rx_XI = np.real(CD_X)
    # Rx_XQ = np.imag(CD_X)
    # Rx_YI = np.real(CD_Y)
    # Rx_YQ = np.imag(CD_Y)
    #
    # Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(Rx_XI, up=parameter.upsamplenum, down=1),
    #                            signal.resample_poly(Rx_XQ, up=parameter.upsamplenum, down=1),
    #                            parameter.pamorder)
    # Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(Rx_YI, up=parameter.upsamplenum, down=1),
    #                            signal.resample_poly(Rx_YQ, up=parameter.upsamplenum, down=1),
    #                            parameter.pamorder)
    #
    # Histogram2D('Y',Rx_YI[0: 100000] + 1j * Rx_YQ[0: 100000], Imageaddress)

    # Rx Upsample
    Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(parameter.RxXI, up=parameter.upsamplenum, down=1),
                                 signal.resample_poly(parameter.RxXQ, up=parameter.upsamplenum, down=1),
                                 parameter.pamorder)
    Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(parameter.RxYI, up=parameter.upsamplenum, down=1),
                                 signal.resample_poly(parameter.RxYQ, up=parameter.upsamplenum, down=1),
                                 parameter.pamorder)

    # print('Tx_Resample Length={}'.format(len(Tx_Signal_X)), 'Rx_Resample Length={}'.format(len(Rx_XI)))
    # prbs = np.ceil(DataNormalize(parameter.PRBS, [], parameter.pamorder))
    # XSNR, XEVM, YSNR, YEVM = np.zeros(parameter.resamplenumber), np.zeros(parameter.resamplenumber), np.zeros(
    #     parameter.resamplenumber), np.zeros(parameter.resamplenumber)

    # cd_compensator = CD_compensator(Rx_XI + 1j * Rx_XQ, Rx_YI + 1j * Rx_YQ, Gbaud=28.125e9 * parameter.upsamplenum, KM=100)  # 39.62
    # CD_X, CD_Y = cd_compensator.overlap_save(Nfft=4096, NOverlap=512)  # 4096
    # Rx_XI = np.real(CD_X); Rx_XQ = np.imag(CD_X)
    # Rx_YI = np.real(CD_Y); Rx_YQ = np.imag(CD_Y)

    # Eye position scan
    for eyepos in range(eyestart, eyeend, eyescan):
        down_num = eyepos
        print('eye position = {}'.format(down_num))
        n = 1
        RxXI = signal.resample_poly(Rx_XI[down_num:], up=1, down=parameter.resamplenumber / n)
        RxXQ = signal.resample_poly(Rx_XQ[down_num:], up=1, down=parameter.resamplenumber / n)
        RxYI = signal.resample_poly(Rx_YI[down_num:], up=1, down=parameter.resamplenumber / n)
        RxYQ = signal.resample_poly(Rx_YQ[down_num:], up=1, down=parameter.resamplenumber / n)
        Rx_Signal_X = RxXI + 1j * RxXQ
        Rx_Signal_Y = RxYI + 1j * RxYQ
        Histogram2D('X', Rx_Signal_X[0:100000], Imageaddress + "/")

        for tap_1 in range(tap1_start, tap1_end, tap1_scan):
            print("eye : {} ,tap : {}".format(eyepos, tap_1))
            cma = CMA_single(Rx_Signal_X[0:100000], Rx_Signal_Y[0:100000], taps=tap_1, iter=cma_iter[0], mean=0)
            cma.stepsize_x = cma.stepsizelist[4]
            cma.stepsize_y = cma.stepsizelist[4]
            #CMAstage1_stepsize_y = cma.stepsize_y
            cma.qam_4_butter_RD(cma_stage, radii)
            # cma.qam_4_side_RD(stage=cma_stage[0])
            Rx_X_CMA_stage1 = cma.rx_x_cma[cma.rx_x_cma != 0]
            Rx_Y_CMA_stage1 = cma.rx_y_cma[cma.rx_y_cma != 0]
            """
            for tap_2 in range(tap2_start, tap2_end, tap2_scan):
                print("eye : {} ,tap : {}".format(eyepos, tap_2))
                cma = CMA_single(Rx_X_CMA_stage1, Rx_Y_CMA_stage1, taps=tap_2, iter=cma_iter[1], mean=0)
                cma.stepsize_x = cma.stepsizelist[4]
                cma.stepsize_y = cma.stepsizelist[4]
                # CMAstage1_stepsize_y = cma.stepsize_y
                cma.qam_4_butter_RD(cma_stage, radii)
                # cma.qam_4_side_RD(stage=cma_stage[0])
                Rx_X_CMA_stage1 = cma.rx_x_cma[cma.rx_x_cma != 0]
                Rx_Y_CMA_stage1 = cma.rx_y_cma[cma.rx_y_cma != 0]
            
            if isplot:
                Histogram2D('CMA_X_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_X_CMA_stage1,
                            Imageaddress + "/")
                Histogram2D('CMA_Y_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type), Rx_Y_CMA_stage1,
                            Imageaddress + "/")
            print("======================One done!!========================")
            """
    return Rx_X_CMA_stage1, Rx_Y_CMA_stage1


def phase_recover():

    return
"""
            Rx_X_CMA = Rx_X_CMA_stage1
            Rx_Y_CMA = Rx_Y_CMA_stage1
        # --------------------------- X PART------------------------
        print('--------------------------------')
        print('X part')
        ph = KENG_phaserecovery()
        FOcompen_X = ph.FreqOffsetComp(Rx_X_CMA)
        Histogram2D('KENG_FOcompensate_X', FOcompen_X, Imageaddress)

        DDPLL_RxX = ph.PLL(FOcompen_X)
        Histogram2D('KENG_FreqOffset_X', DDPLL_RxX[0, :], Imageaddress)

        phasenoise_RxX = DDPLL_RxX
        PN_RxX = ph.QAM_6(phasenoise_RxX, c1_radius=1.55, c2_radius=3.2)
        PN_RxX = PN_RxX[PN_RxX != 0]
        Histogram2D('KENG_PhaseNoise_X', PN_RxX, Imageaddress)

        Normal_ph_RxX_real, Normal_ph_RxX_imag = DataNormalize(np.real(PN_RxX), np.imag(PN_RxX), parameter.pamorder)
        Normal_ph_RxX = Normal_ph_RxX_real + 1j * Normal_ph_RxX_imag
        # Histogram2D('KENG_PLL_Normalized_Y', Normal_ph_RxX, Imageaddress)

        Correlation = KENG_corr(window_length=7000)
        TxX_real, RxX_real, p = Correlation.corr(TxXI, np.real(Normal_ph_RxX[0:50000]), 13)
        TxX_imag, RxX_imag, p = Correlation.corr(TxXQ, np.imag(Normal_ph_RxX[0:50000]), 13)
        RxX_corr = RxX_real[0:40000] + 1j * RxX_imag[0:40000]
        TxX_corr = TxX_real[0:40000] + 1j * TxX_imag[0:40000]
        # Histogram2D('KENG_Corr_X', RxX_corr, Imageaddress)

        SNR_X, EVM_X = SNR(RxX_corr, TxX_corr)
        bercount_X = BERcount(np.array(TxX_corr), np.array(RxX_corr), parameter.pamorder)
        print('BER_X = {} \nSNR_X = {} \nEVM_X = {}'.format(bercount_X, SNR_X, EVM_X))
        XSNR[eyepos], XEVM[eyepos] = SNR_X, EVM_X
        # --------------------------- Y PART------------------------
        print('--------------------------------')
        print('Y part')
        ph = KENG_phaserecovery()
        FOcompen_Y = ph.FreqOffsetComp(Rx_Y_CMA)
        # Histogram2D('KENG_FOcompensate_Y', FOcompen_Y, Imageaddress)

        DDPLL_RxY = ph.PLL(FOcompen_Y)
        # Histogram2D('KENG_FreqOffset_Y', DDPLL_RxY[0, :], Imageaddress)

        phasenoise_RxY = DDPLL_RxY
        PN_RxY = ph.QAM_6(phasenoise_RxY, c1_radius=1.55, c2_radius=3.2)
        PN_RxY = PN_RxY[PN_RxY != 0]
        Histogram2D('KENG_PhaseNoise_Y', PN_RxY, Imageaddress)

        Normal_ph_RxY_real, Normal_ph_RxY_imag = DataNormalize(np.real(PN_RxY), np.imag(PN_RxY), parameter.pamorder)
        Normal_ph_RxY = Normal_ph_RxY_real + 1j * Normal_ph_RxY_imag
        # Histogram2D('KENG_PLL_Normalized_Y', Normal_ph_RxY, Imageaddress)

        Correlation = KENG_corr(window_length=7000)
        TxY_real, RxY_real, p = Correlation.corr(TxYI, np.real(Normal_ph_RxY[0:50000]), 13)
        TxY_imag, RxY_imag, p = Correlation.corr(TxYQ, np.imag(Normal_ph_RxY[0:50000]), 13)
        RxY_corr = RxY_real[0:40000] + 1j * RxY_imag[0:40000]
        TxY_corr = TxY_real[0:40000] + 1j * TxY_imag[0:40000]
        # Histogram2D('KENG_Corr_Y', RxY_corr, Imageaddress)

        SNR_Y, EVM_Y = SNR(RxY_corr, TxY_corr)
        bercount_Y = BERcount(np.array(TxY_corr), np.array(RxY_corr), parameter.pamorder)
        print('BER_Y = {} \nSNR_Y = {} \nEVM_Y = {}'.format(bercount_Y, SNR_Y, EVM_Y))
        YSNR[eyepos], YEVM[eyepos] = SNR_Y, EVM_Y
        print('--------------------------------')
    # ---
    
    equalizer_real = Equalizer(np.real(np.array(Tx_corr.T)[0, :]), np.real(np.array(Rx_corr.T)[0, :]), 3, [11, 31, 31],
                               0.5)
    equalizer_imag = Equalizer(np.imag(np.array(Tx_corr.T)[0, :]), np.imag(np.array(Rx_corr.T)[0, :]), 3, [11, 31, 31],
                               0.5)
    Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
    # ---
    # equalizer_complex = Equalizer(Tx_corr, Rx_corr, 3, [11, 3, 3], 0.5)
    # equalizer_complex = Equalizer( np.array(Tx_corr.T)[0,:], np.array(Rx_corr.T)[0,:], 3, [21, 3, 1], 0.1)

    # equalizer_real = Equalizer(np.real(Tx_corr), np.real(Rx_corr), 3, [11, 11, 11])
    # equalizer_imag = Equalizer(np.imag(Tx_corr), np.imag(Rx_corr), 3, [11, 11, 11])
    # Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    # Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    # Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    # Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
    # Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()
    snr_volterra, evm_volterra = SNR(Tx_real_volterra, Rx_real_volterra)
    # snr_volterra, evm_volterra = SNR(Rx_complex_volterra, Tx_complex_volterra)
    # bercount = BERcount(Rx_complex_volterra, Tx_complex_volterra, parameter.pamorder)
    # bercount = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
    # print(bercount)
    print(snr_volterra, evm_volterra)
    Histogram2D("ComplexVolterra", Rx_real_volterra, snr_volterra, evm_volterra)
"""


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input', dest="input", help="input file", default='./behave_config.yml')
    (options, args) = parser.parse_args()
    # load config data from yaml file
    yml_cfg = {}
    if options.input != '':
        yml_cfg = yaml.safe_load(open(options.input))['Behave_Sim']
    else:
        raise Exception('Error: Input file is required')
    #          #################### Initialize  #################### #
    folder = yml_cfg['folder']
    symbolRate = yml_cfg['symbolRate']
    pamorder = yml_cfg['pamorder']
    is_simulation = yml_cfg['is_simulation']
    # ///////////////////////////////////////////////////////////////////////////////////////// #
    # In[]
    # Start Implement
    parameter = Parameter(folder, symbolRate, pamorder, is_simulation)
    decoding(parameter, options.input)

