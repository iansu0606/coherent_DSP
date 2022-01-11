from _init_ import *
# initial Parameter
route = os.path.abspath(os.getcwd())
cur_dir = '..\image_200Gbd_QPSK'
# cur_dir = ''.join([route, r'\image'])
Path(cur_dir).mkdir(parents=True, exist_ok=True)

parameter = Parameter(simulation=False)
print('SymbolRate={}GBaud'.format(parameter.symbolRate / 1e9), 'Pamorder={}'.format(parameter.pamorder),
      'resamplenumber={}'.format(parameter.resamplenumber))
# print('Tx Length={}'.format(len(parameter.TxXI)), 'Rx Length={}'.format(len(parameter.RxXI)))
# ----------------------------------------------------------------------------------------------------------------------
#                               Tx to Bit
# ----------------------------------------------------------------------------------------------------------------------
Tx_Signal_X = np.array(Tx2Bit(parameter.TxXI, parameter.TxXQ, parameter.pamorder))
Tx_Signal_Y = np.array(Tx2Bit(parameter.TxYI, parameter.TxYQ, parameter.pamorder))
# Tx_Signal_X = np.array(Tx2Bit(parameter.TxXI_prbs, parameter.TxXQ_prbs, parameter.pamorder))
# Tx_Signal_Y = np.array(Tx2Bit(parameter.TxYI_prbs, parameter.TxYQ_prbs, parameter.pamorder))
Histogram2D_Hot("TX_X", Tx_Signal_X, path=cur_dir)
Histogram2D_Hot("TX_Y", Tx_Signal_Y, path=cur_dir)
# Histogram2D_Hot("Rx1", parameter.RxXI)
# Histogram2D_Hot("TX", parameter.RxXQ)
# Histogram2D_Hot("TX", )
# Histogram2D_Hot("TX", Tx_Signal_Y)
# Histogram2D_Hot("RX_", np.array(parameter.RxXI)+1j*np.array(parameter.RxXQ))
#-----------------------------------------------------------------------------------------------------------------------
#                               Rx Upsample
#-----------------------------------------------------------------------------------------------------------------------
# Rx_XI, Rx_XQ = DataNormalize(parameter.RxXI, parameter.RxXQ,
#                              parameter.pamorder)
# Rx_YI, Rx_YQ = DataNormalize(parameter.RxYI, parameter.RxYQ,
#                              parameter.pamorder)
# CD_comp = CD_compensator(Rx_XI+1j*Rx_XQ, Rx_YI+1j*Rx_YQ, parameter.symbolRate*16/9, 100)
# Rx_X_CD, Rx_Y_CD = CD_comp.overlap_save(300000, 16)
# window = signal.windows.kaiser(parameter.resamplenumber, beta=14)
# Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(np.real(Rx_X_CD), up=parameter.upsamplenum, down=1),
#                              signal.resample_poly(np.imag(Rx_X_CD), up=parameter.upsamplenum, down=1),
#                              parameter.pamorder)
# Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(np.real(Rx_Y_CD), up=parameter.upsamplenum, down=1),
#                              signal.resample_poly(np.imag(Rx_Y_CD), up=parameter.upsamplenum, down=1),
#                              parameter.pamorder)



# window = signal.windows.kaiser(parameter.resamplenumber, beta=14)
Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(parameter.RxXI, up=parameter.upsamplenum, down=1),
                             signal.resample_poly(parameter.RxXQ, up=parameter.upsamplenum, down=1),
                             parameter.pamorder)
Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(parameter.RxYI, up=parameter.upsamplenum, down=1),
                             signal.resample_poly(parameter.RxYQ, up=parameter.upsamplenum, down=1),
                             parameter.pamorder)
Histogram2D_Hot("Rx_X(upsample)", Rx_XI[0:100000]+1j*Rx_XQ[0:100000], path=cur_dir)
Histogram2D_Hot("Rx_Y(upsample)", Rx_YI[0:100000]+1j*Rx_YQ[0:100000], path=cur_dir)
# symbol_sync = Symbol_synchronizer(Rx_XI+1J*Rx_XQ, Rx_YI+1J*Rx_YQ, sps=16)
# Rx_X_SYNC, Rx_Y_SYNC = symbol_sync.Gardner()
# CD_comp = CD_compensator(Rx_X_SYNC[0:562500], Rx_Y_SYNC[0:562500], parameter.symbolRate, 80)
# Rx_X_CD, Rx_Y_CD = CD_comp.overlap_save(562500, 14)
# CD_comp = CD_compensator(Rx_XI+1j*Rx_XQ, Rx_YI+1j*Rx_YQ, 106e9, 80)
# Rx_X_CD, Rx_Y_CD = CD_comp.overlap_save(len(Rx_XI), 2656)

# Rx_XI_eye = DataNormalize(signal.resample_poly(Rx_XI, up=1, down=parameter.resamplenumber, window=window), [], parameter.pamorder)
# Rx_XQ_eye = DataNormalize(signal.resample_poly(Rx_XQ, up=1, down=parameter.resamplenumber, window=window), [], parameter.pamorder)
# Rx_YI_eye = DataNormalize(signal.resample_poly(Rx_YI, up=1, down=parameter.resamplenumber, window=window), [], parameter.pamorder)
# Rx_YQ_eye = DataNormalize(signal.resample_poly(Rx_YQ, up=1, down=parameter.resamplenumber, window=window), [], parameter.pamorder)
# print('Tx_Resample Length={}'.format(len(Tx_Signal_X)), 'Rx_Resample Length={}'.format(len(Rx_XI)))
# prbs = np.ceil(DataNormalize(parameter.PRBS, [], parameter.pamorder))
# ----------------------------------------------------------------------------------------------------------------------
#                               keysight Ref&Meas Data
# ----------------------------------------------------------------------------------------------------------------------
# # X
# window_ref = signal.windows.kaiser(parameter.keysightdownnum, beta=14)
# Rx_keysightXI_eye_ref = signal.resample_poly(parameter.keysightXI_ref, up=1, down=parameter.keysightdownnum, window=window_ref)
# Rx_keysightXQ_eye_ref = signal.resample_poly(parameter.keysightXQ_ref, up=1, down=parameter.keysightdownnum, window=window_ref)
# # Rx_keysightXI_eye_ref = Downsample(parameter.keysightXI_ref, downsample=parameter.keysightdownnum, downnum=0)
# # Rx_keysightXQ_eye_ref = Downsample(parameter.keysightXQ_ref, downsample=parameter.keysightdownnum, downnum=0)
# Rx_keysight_Xref = Rx_keysightXI_eye_ref + 1j * Rx_keysightXQ_eye_ref
# Tx_X = np.array(Tx2Bit(Rx_keysightXI_eye_ref, Rx_keysightXQ_eye_ref, parameter.pamorder))
# Histogram2D_Hot('Rx_Keysight_X_ref', Rx_keysight_Xref)
# Rx_keysightXI_eye_meas = signal.resample_poly(parameter.keysightXI_meas, up=1, down=parameter.keysightdownnum, window=window_ref)
# Rx_keysightXQ_eye_meas = signal.resample_poly(parameter.keysightXQ_meas, up=1, down=parameter.keysightdownnum, window=window_ref)
# # Rx_keysightXI_eye_meas = Downsample(parameter.keysightXI_meas, downsample=parameter.keysightdownnum, downnum=0)
# # Rx_keysightXQ_eye_meas = Downsample(parameter.keysightXQ_meas, downsample=parameter.keysightdownnum, downnum=0)
# Rx_keysight_Xmeas = Rx_keysightXI_eye_meas + 1j * Rx_keysightXQ_eye_meas
# Histogram2D_Hot('Rx_Keysight_X_meas', Rx_keysight_Xmeas)
# # Y
# Rx_keysightYI_eye_ref = signal.resample_poly(parameter.keysightYI_ref, up=1, down=parameter.keysightdownnum, window=window_ref)
# Rx_keysightYQ_eye_ref = signal.resample_poly(parameter.keysightYQ_ref, up=1, down=parameter.keysightdownnum, window=window_ref)
# Rx_keysight_Yref = Rx_keysightYI_eye_ref + 1j * Rx_keysightYQ_eye_ref
# Tx_Y = np.array(Tx2Bit(Rx_keysightYI_eye_ref, Rx_keysightYQ_eye_ref, parameter.pamorder))
# Histogram2D_Hot('Rx_Keysight_Y_ref', Rx_keysight_Yref)
# Rx_keysightYI_eye_meas = signal.resample_poly(parameter.keysightYI_meas, up=1, down=parameter.keysightdownnum, window=window_ref)
# Rx_keysightYQ_eye_meas = signal.resample_poly(parameter.keysightYQ_meas, up=1, down=parameter.keysightdownnum, window=window_ref)
# Rx_keysight_Ymeas = Rx_keysightYI_eye_meas + 1j * Rx_keysightYQ_eye_meas
# Histogram2D_Hot('Rx_Keysight_Y_meas', Rx_keysight_Ymeas)
# ----------------------------------------------------------------------------------------------------------------------
snrx, snry = np.zeros((parameter.resamplenumber, 1)), np.zeros((parameter.resamplenumber, 1))
evmx, evmy = np.zeros((parameter.resamplenumber, 1)), np.zeros((parameter.resamplenumber, 1))
#Eye position scan
start, end = 0, 1
for eyepos in range(start, end):
    # Rx_XI_eye = Downsample(Rx_XI, downsample=parameter.resamplenumber, downnum=eyepos)
    # Rx_XQ_eye = Downsample(Rx_XQ, downsample=parameter.resamplenumber, downnum=eyepos)
    # Rx_YI_eye = Downsample(Rx_YI, downsample=parameter.resamplenumber, downnum=eyepos)
    # Rx_YQ_eye = Downsample(Rx_YQ, downsample=parameter.resamplenumber, downnum=eyepos)
    # Rx_XI_eye = DataNormalize(signal.resample_poly(Rx_XI[eyepos:], up=1, down=parameter.resamplenumber, window=window), [],
    #                           parameter.pamorder)
    # Rx_XQ_eye = DataNormalize(signal.resample_poly(Rx_XQ[eyepos:], up=1, down=parameter.resamplenumber, window=window), [],
    #                           parameter.pamorder)
    # Rx_YI_eye = DataNormalize(signal.resample_poly(Rx_YI[eyepos:], up=1, down=parameter.resamplenumber, window=window), [],
    #                           parameter.pamorder)
    # Rx_YQ_eye = DataNormalize(signal.resample_poly(Rx_YQ[eyepos:], up=1, down=parameter.resamplenumber, window=window), [],
    #                           parameter.pamorder)
    Rx_XI_eye = DataNormalize(signal.resample_poly(Rx_XI[eyepos:], up=1, down=parameter.resamplenumber), [],
                              parameter.pamorder)
    Rx_XQ_eye = DataNormalize(signal.resample_poly(Rx_XQ[eyepos:], up=1, down=parameter.resamplenumber), [],
                              parameter.pamorder)
    Rx_YI_eye = DataNormalize(signal.resample_poly(Rx_YI[eyepos:], up=1, down=parameter.resamplenumber), [],
                              parameter.pamorder)
    Rx_YQ_eye = DataNormalize(signal.resample_poly(Rx_YQ[eyepos:], up=1, down=parameter.resamplenumber), [],
                              parameter.pamorder)
    Rx_Signal_X = Rx_XI_eye + 1j * Rx_XQ_eye
    Rx_Signal_Y = Rx_YI_eye + 1j * Rx_YQ_eye
    # Rx_Signal_X = Rx_X_SYNC
    # Rx_Signal_Y = Rx_Y_SYNC

    Histogram2D_Hot('Rx_X_{}'.format(eyepos), Rx_Signal_X[0:100000], path=cur_dir)
    Histogram2D_Hot('Rx_Y_{}'.format(eyepos), Rx_Signal_Y[0:100000], path=cur_dir)
    # CD_comp = CD_compensator(Rx_Signal_X, Rx_Signal_Y, parameter.symbolRate, 80*2)
    # Rx_X_CD, Rx_Y_CD = CD_comp.overlap_save(2048, 64)
    # Histogram2D_Hot('Rx_X_CD{}'.format(eyepos), Rx_X_CD, path=cur_dir)
    # Histogram2D_Hot('Rx_Y_CD{}'.format(eyepos), Rx_Y_CD, path=cur_dir)
# ----------------------------------------------------------------------------------------------------------------------
#                               QPSK CMA
# ----------------------------------------------------------------------------------------------------------------------
    for i in range(9, 11, 2):
        cma = CMA(Rx_Signal_X[0:70000], Rx_Signal_Y[0:70000], taps=i, iter=10)
        # cma = CMA(Rx_X_CD[0:200000], Rx_Y_CD[0:200000], taps=i, iter=30)
        # cma = CMA(Rx_keysight_X, Rx_keysight_Ymeas, taps=5, iter=1000)
        cma.CMA_butterfly()
        Rx_X_CMA, Rx_Y_CMA = cma.rx_x_single, cma.rx_y_single
        Histogram2D_Hot('CMA1X_{}(taps{})'.format(eyepos, i), Rx_X_CMA,path=cur_dir)
        Histogram2D_Hot('CMA1Y_{}(taps{})'.format(eyepos, i), Rx_Y_CMA,path=cur_dir)
    # symbol_sync = Symbol_synchronizer(Rx_X_CMA, Rx_Y_CMA, sps=2)
    # Rx_X_CMA, Rx_Y_CMA = symbol_sync.Gardner()
    # Histogram2D_Hot("CMA_D_X", Rx_X_CMA,path=cur_dir)
    # Histogram2D_Hot("CMA_D_Y", Rx_Y_CMA,path=cur_dir)
# ----------------------------------------------------------------------------------------------------------------------
#                               16QAM CMA
# ----------------------------------------------------------------------------------------------------------------------
#     for i in range(17, 33, 2):
#         # cma = CMA(Rx_Signal_X[0:100000], Rx_Signal_Y[0:100000], taps=[i, i], iter=30)
#         cma = CMA(Rx_Signal_X[0:100000], Rx_Signal_Y[0:100000], taps=i, iter=30)
#         cma.RDE_butterfly()
#         Rx_X_CMA, Rx_Y_CMA = cma.rx_x_single, cma.rx_y_single
#         print('CMA Batch Size={}'.format(cma.batchsize), 'CMA Stepsize={}'.format(cma.stepsize),
#               'CMA OverHead={}%'.format(cma.overhead * 100))
#         Histogram2D_Hot('CMAX_{}(taps{})'.format(eyepos, i), Rx_X_CMA, path=cur_dir)
#         Histogram2D_Hot('CMAY_{}(taps{})'.format(eyepos, i), Rx_Y_CMA, path=cur_dir)
    # symbol_sync = Symbol_synchronizer(Rx_X_CMA, Rx_Y_CMA, sps=2)
    # Rx_X_CMA, Rx_Y_CMA = symbol_sync.Gardner()
    # Histogram2D_Hot("CMA_D_X", Rx_X_CMA, path=cur_dir)
    # Histogram2D_Hot("CMA_D_Y", Rx_Y_CMA, path=cur_dir)

    # figure, axes = plt.subplots()
    # C1 = plt.Circle((0, 0), 2**0.5, fill=False)
    # C2 = plt.Circle((0, 0), 10**0.5, fill=False)
    # C3 = plt.Circle((0, 0), 18**0.5, fill=False)
    # plt.gcf().gca().add_artist(C1)
    # plt.gcf().gca().add_artist(C2)
    # plt.gcf().gca().add_artist(C3)
    # for pp in range(0, 10):
    #     a = np.real(cma.record[pp][:])
    #     b = np.imag(cma.record[pp][:])
    #     plt.scatter(a, b)
    # axes.set_aspect(1)
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.show()


# ----------------------------------------------------------------------------------------------------------------------
#                               FFT-FOE
# ----------------------------------------------------------------------------------------------------------------------
    freqoffsetX = Phaserecovery(Rx_X_CMA)
    Rx_X_FO = freqoffsetX.FFT_FOE()
    freqoffsetY = Phaserecovery(Rx_Y_CMA)
    Rx_Y_FO = freqoffsetY.FFT_FOE()
    a = np.array(Rx_X_CMA)**4
    Histogram2D_Hot("FreqOffset 4th", a, path=cur_dir)
    Histogram2D_Hot(str(eyepos)+"_FreqOffsetCompX", Rx_X_FO, path=cur_dir)
    Histogram2D_Hot(str(eyepos)+"_FreqOffsetCompY", Rx_Y_FO, path=cur_dir)
    # symbol_sync = Symbol_synchronizer(Rx_X_FO, Rx_Y_FO, sps=2)
    # Rx_X_FO, Rx_Y_FO = symbol_sync.Gardner()
    # pll = Phaserecovery(Rx_X_FO)
    # pll_out = pll.PLL()
    # Histogram2D_Hot("PLL", pll_out[0])
    Rx_X_FO = DataNormalize(np.real(Rx_X_FO), [], parameter.pamorder) + 1j * DataNormalize(
        np.imag(Rx_X_FO), [], parameter.pamorder)
    Rx_Y_FO = DataNormalize(np.real(Rx_Y_FO), [], parameter.pamorder) + 1j * DataNormalize(
        np.imag(Rx_Y_FO), [], parameter.pamorder)

# ----------------------------------------------------------------------------------------------------------------------
#                               QPSK VV
# ----------------------------------------------------------------------------------------------------------------------
#     phase_viterbi = Phaserecovery(Rx_X_FO, taps=101)
#     vv_x = phase_viterbi.V_Valg().reshape(-1, )
#     # pll_X = phase_viterbi.PLL().reshape(-1, )
#     Histogram2D_Hot("VV_X", vv_x,path=cur_dir)
#     phase_viterbi = Phaserecovery(Rx_Y_FO, taps=101)
#     vv_y = phase_viterbi.V_Valg().reshape(-1, )
#     # pll_X = phase_viterbi.PLL().reshape(-1, )
#     Histogram2D_Hot("VV_y", vv_y,path=cur_dir)
#     cor_data_x = vv_x
#     cor_data_y = vv_y
#     qq = np.linspace(0, 2000, 2000)
#     plt.figure()
#     plt.plot(qq, phase_viterbi.y[20000:22000])
#     # plt.scatter(qq, phase_viterbi.y[20000:22000])
#     plt.xlabel("Kth Point Phase")
#     plt.ylabel("Radius")
#     plt.show()
# ----------------------------------------------------------------------------------------------------------------------
#                               QPSK Partition VVPE + ML
# ----------------------------------------------------------------------------------------------------------------------
    phase_viterbi = Phaserecovery(Rx_X_FO, taps=101)
    vv_out = phase_viterbi.QPSK_partition_VVPE(cur_dir, inner=2.3, outter=3.8)
    ct_out = phase_viterbi.CT(vv_out, cur_dir, taps=11)
    # mlx_out = phase_viterbi.ML(ct_out, taps=9)
    eml_x = phase_viterbi.Enhanced_ML(ct_out, cur_dir, taps=51)
    pll_x = phase_viterbi.PLL().reshape(-1)
    Histogram2D_Hot(str(eyepos) + "E_ML_X", eml_x, path=cur_dir)
    Histogram2D_Hot(str(eyepos) + "PLL_X", pll_x, path=cur_dir)
    phase_viterbi = Phaserecovery(Rx_Y_FO, taps=51)
    vv_out = phase_viterbi.QPSK_partition_VVPE(cur_dir, inner=2.3, outter=3.8)
    ct_out = phase_viterbi.CT(vv_out, cur_dir, taps=11)
    # mly_out = phase_viterbi.ML(ct_out, taps=9)
    eml_y = phase_viterbi.Enhanced_ML(ct_out, cur_dir, taps=51)
    Histogram2D_Hot(str(eyepos) + "E_ML_Y", eml_y, path=cur_dir)

    cor_data_x = eml_x
    cor_data_y = eml_y
# ----------------------------------------------------------------------------------------------------------------------
#                               Data Correlation
# ----------------------------------------------------------------------------------------------------------------------
    corrindx = []
    #x
    cor_data_x= DataNormalize(np.real(cor_data_x), [], parameter.pamorder) + 1j * DataNormalize(
        np.imag(cor_data_x), [], parameter.pamorder)
    Tx_real, Rx_real, c_real = corr(np.real(Tx_Signal_X), np.real(cor_data_x), parameter.Prbsnum)
    Tx_imag, Rx_imag, c_imag = corr(np.imag(Tx_Signal_X), np.imag(cor_data_x), parameter.Prbsnum)
    length = min([len(Rx_real), len(Rx_imag)])
    Tx_corr_X = Tx_real[:length] + 1j * Tx_imag[:length]
    Rx_corr_X = Rx_real[:length] + 1j * Rx_imag[:length]
    corrindx.append([c_real, c_imag])
    snr, evm = SNR(Tx_corr_X, Rx_corr_X)
    ber = BER(evm, parameter.pamorder)
    bercount = BERcount(Tx_corr_X, Rx_corr_X, parameter.pamorder)
    Histogram2D_Hot(str(eyepos)+'_Rx_corr_X', Rx_corr_X, snr, evm, bercount,path=cur_dir)
    print("Eyepos:", eyepos)
    print("BERcount:", bercount, "BER:", ber)
    print("SNR:", snr, "EVM:", evm)
    snrx[eyepos] = snr
    evmx[eyepos] = evm

    #y
    cor_data_y= DataNormalize(np.real(cor_data_y), [], parameter.pamorder) + 1j * DataNormalize(
        np.imag(cor_data_y), [], parameter.pamorder)
    Tx_real, Rx_real, c_real = corr(np.real(Tx_Signal_Y), np.real(cor_data_y), parameter.Prbsnum)
    Tx_imag, Rx_imag, c_imag = corr(np.imag(Tx_Signal_Y), np.imag(cor_data_y), parameter.Prbsnum)
    length = min([len(Rx_real), len(Rx_imag)])
    Tx_corr_Y = Tx_real[:length] + 1j * Tx_imag[:length]
    Rx_corr_Y = Rx_real[:length] + 1j * Rx_imag[:length]
    corrindx.append([c_real, c_imag])
    snr, evm = SNR(Tx_corr_Y, Rx_corr_Y)
    ber = BER(evm, parameter.pamorder)
    bercount = BERcount(Tx_corr_Y, Rx_corr_Y, parameter.pamorder)
    Histogram2D_Hot(str(eyepos)+'_Rx_corr_Y',Rx_corr_Y, snr, evm, bercount,path=cur_dir)
    print("Eyepos:", eyepos)
    print("BERcount:", bercount, "BER:", ber)
    print("SNR:", snr, "EVM:", evm)
    snry[eyepos] = snr
    evmy[eyepos] = evm

    print("X:", np.max(snrx), np.argmax(snrx), "Y: bn ", np.max(snry), np.argmax(snry))
    ParaRecord(eyepos_X=np.argmax(snrx), eyepos_Y=np.argmax(snry), cmataps=cma.cmataps, cmastepsize=cma.stepsize, snry=np.max(snry),
               snrx=np.max(snrx), corr_indx=corrindx)
# ----------------------------------------------------------------------------------------------------------------------
#                          NN equalizer
# ----------------------------------------------------------------------------------------------------------------------

# taps = 31
# batch_size = 500
# LR = 1e-3
# EPOCH = 300
# overhead = 0.2
# trainnum = int(len(Rx_corr) * overhead)
# device = torch.device("cuda:0")
# train_inputr = rolling_window(torch.Tensor(Rx_corr.real), taps)[:trainnum]
# train_inputi = rolling_window(torch.Tensor(Rx_corr.imag), taps)[:trainnum]
# train_targetr = torch.Tensor(Tx_corr.real[taps // 2:-taps // 2 + 1])[:trainnum]
# train_targeti = torch.Tensor(Tx_corr.imag[taps // 2:-taps // 2 + 1])[:trainnum]
# train_tensor = Data.TensorDataset(train_inputr, train_inputi, train_targetr, train_targeti)
# train_loader = Data.DataLoader(train_tensor, batch_size=batch_size, shuffle=False)
#
# val_inputr = rolling_window(torch.Tensor(Rx_corr.real), taps)
# val_inputi = rolling_window(torch.Tensor(Rx_corr.imag), taps)
# val_targetr = torch.Tensor(Tx_corr.real[taps // 2:-taps // 2 + 1])
# val_targeti = torch.Tensor(Tx_corr.imag[taps // 2:-taps // 2 + 1])
# val_tensor = Data.TensorDataset(val_inputr, val_inputi, val_targetr, val_targeti)
# val_loader = Data.DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
#
# layer_widths = [16, 16, 1]
# layer_centres = [16, 16]
# basis_func = gaussian
# final_modelr, final_modeli = Network(layer_widths, layer_centres, basis_func, taps).to(device), Network(
#     layer_widths, layer_centres, basis_func, taps).to(device)
# final_optr = torch.optim.Adam(final_modelr.parameters(), lr=LR)
# final_opti = torch.optim.Adam(final_modeli.parameters(), lr=LR)
# # modelx = conv1dResNet(Residual_Block, [2, 2, 2, 2]).to(device)
# # lossxr = nn.MSELoss()
# # lossxi = nn.MSELoss()
# # lossxc = nn.CrossEntropyLoss()
# # opty = torch.optim.Adam(modely.parameters(), weight_decay=1e-2, lr=LR)
# L = []
# val_L = []
# for epoch in tqdm(range(EPOCH)):
#     for i, (dr, di, txr, txi) in enumerate(train_loader):
#         final_modelr.train()
#         final_modeli.train()
#         outr, outi = final_modelr(dr.to(device)), final_modeli(di.to(device))
#         # outr,outi= final_modelr(dr.unsqueeze(1).to(device)),final_modeli(di.unsqueeze(1).to(device))
#         # trr,tri = harddecision(txr,txi)
#         Lossr = nn.MSELoss()(outr.squeeze().cpu(), torch.Tensor(txr))
#         Lossi = nn.MSELoss()(outi.squeeze().cpu(), torch.Tensor(txi))
#         Loss = Lossr + Lossi
#         final_optr.zero_grad()
#         Lossr.backward(retain_graph=True)
#         final_optr.step()
#         final_opti.zero_grad()
#         Lossi.backward()
#         final_opti.step()
#         L.append(Loss.detach().cpu().numpy())
#         # modely.eval()
#     print("Train Loss:{:.3f}".format(Loss),
#           '||' "Train Bercount:{:.3E}".format(BERcount(txr + 1j * txi, outr.cpu() + 1j * outi.cpu(), 4)[0]))
#     # print('\n training Accx: %f\n training Accy: %f\n' % (np.mean(Accx), np.mean(Accy)))
#     # Accx = []
#     # # Accy = []
#     predictr, predicti = [], []
# final_modelr.eval()
# final_modeli.eval()
# for i, (dr, di, txr, txi) in enumerate(val_loader):
#     outr, outi = final_modelr(dr.to(device)), final_modeli(di.to(device))
#     predictr.extend(outr.cpu().detach().numpy())
#     predicti.extend(outi.cpu().detach().numpy())
#     # Lossxr = lossxr(outr.cpu(), txr)
#     # Lossxi = lossxi(outi.cpu(), txi)
#     # Lossyr = lossyr(outy[:, 0].cpu(), tyr)
#     # Lossyi = lossyi(outy[:, 1].cpu(), tyi)
#     # Lossxc = lossxc(outcx.cpu(), tgx)
#     # Lossyc = lossyc(outcy.cpu(), tgy)
#     # xacc = (tgx.eq(torch.max(outcx.cpu(), 1)[1])).sum() / outcx.shape[0]
#     # yacc = (tgy.eq(torch.max(outcy.cpu(), 1)[1])).sum() / outcy.shape[0]
#     print("Val BERcount:{:.3E}".format(BERcount(txr + 1j * txi, outr.cpu() + 1j * outi.cpu(), 4)[0]))
# predictr = np.array(predictr).squeeze()
# predicti = np.array(predicti).squeeze()
# snr, evm = SNR(Tx_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze())
# bercount = BERcount(Tx_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze(), 4)
# print(snr, evm)
# Histogram2D_Hot("RBF-Net", (np.array(predictr) + 1j * np.array(predicti)).squeeze(), snr, evm, bercount)

# ----------------------------------------------------------------------------------------------------------------------
#                          Volterra equalizer
# ----------------------------------------------------------------------------------------------------------------------
## complex volterra
    equalizer_complex = Equalizer(Tx_corr_Y, Rx_corr_Y, 3, [11, 11, 15], 0.32)
    Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()
    snr_complex, evm_complex = SNR(Tx_complex_volterra, Rx_complex_volterra)
    bercomplex = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
    Histogram2D_Hot(str(eyepos)+"_ComplexVolterra_X", Rx_complex_volterra, snr_complex, evm_complex, bercomplex,path=cur_dir)
    ## real volterra
    equalizer_real = Equalizer(np.real(Tx_corr_Y), np.real(Rx_corr_Y), 3, [21, 21, 21], 0.32)
    equalizer_imag = Equalizer(np.imag(Tx_corr_Y), np.imag(Rx_corr_Y), 3, [21, 21, 21], 0.32)
    Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
    snr_volterra, evm_volterra = SNR(Tx_real_volterra, Rx_real_volterra)
    bercount = BERcount(Tx_real_volterra, Rx_real_volterra, parameter.pamorder)
    Histogram2D_Hot(str(eyepos)+"_RealVolterra_Y_0.11", Rx_real_volterra, snr_volterra, evm_volterra, bercount,path=cur_dir)

    ParaRecord(eyepos_X=np.argmax(snrx), eyepos_Y=np.argmax(snry), cmataps=cma.cmataps, cmastepsize=cma.stepsize, snry=np.max(snry),
               snrx=np.max(snrx), corr_indx=corrindx, volterrataps=equalizer_real.taps, volterraoverhead=equalizer_real.overhead, snr_volterra=snr_volterra)
# # ----------------------------------------------------------------------------------------------------------------------
# #                           Tektronix DSP QP20
# # ----------------------------------------------------------------------------------------------------------------------
# Tx = np.round(DataNormalize(parameter.TxXI[:50000], [], parameter.pamorder)) + 1j * np.round(
#     DataNormalize(parameter.TxXQ[:50000], [], parameter.pamorder))
# Rx = DataNormalize(np.real(parameter.Xsoft[:50000]), [], parameter.pamorder) + 1j * DataNormalize(
#     np.imag(parameter.Xsoft[:50000]), [], parameter.pamorder)
# snr_tek, evm_tek = SNR(Tx, Rx)
# ber = BER(evm_tek, parameter.pamorder)
# bercount = BERcount(Tx, Rx, parameter.pamorder)
# print(bercount, ber)
# Histogram2D_Hot("Tektronix_X", Rx, snr_tek, evm_tek, bercount,path=cur_dir)
# Tx = np.round(DataNormalize(parameter.TxYI[:50000], [], parameter.pamorder)) + 1j * np.round(
#     DataNormalize(parameter.TxYQ[:50000], [], parameter.pamorder))
# Rx = DataNormalize(np.real(parameter.Ysoft[:50000]), [], parameter.pamorder) + 1j * DataNormalize(
#     np.imag(parameter.Ysoft[:50000]), [], parameter.pamorder)
# snr_tek, evm_tek = SNR(Tx, Rx)
# ber = BER(evm_tek, parameter.pamorder)
# bercount = BERcount(Tx, Rx, parameter.pamorder)
# print(bercount, ber)
# Histogram2D_Hot("Tektronix_Y", Rx, snr_tek, evm_tek, bercount,path=cur_dir)
# # ----------------------------------------------------------------------------------------------------------------------
# #                           Tektronix DSP 16QAM
# # ----------------------------------------------------------------------------------------------------------------------
# TxI = []
# TxQ = []
# for indx in range(50000):
#     TxI.append(int(np.array2string(parameter.TxXI[indx], separator='')[1:-1], 2))
#     TxQ.append(int(np.array2string(parameter.TxXQ[indx], separator='')[1:-1], 2))
# Tx = np.round(DataNormalize(TxI, [], parameter.pamorder)) + 1j * np.round(DataNormalize(TxQ, [], parameter.pamorder))
# Rx = DataNormalize(np.real(parameter.Xsoft[:50000]), [], parameter.pamorder) + 1j * DataNormalize(
#     np.imag(parameter.Xsoft[:50000]), [], parameter.pamorder)
# snr_tek, evm_tek = SNR(Tx, Rx)
# ber = BER(evm_tek, parameter.pamorder)
# bercount = BERcount(Tx, Rx, parameter.pamorder)
# print(bercount, ber)
# Histogram2D_Hot("Tektronix_DSP_X", Rx, snr_tek, evm_tek, bercount, path=cur_dir)
# TxI = []
# TxQ = []
# for indx in range(50000):
#     TxI.append(int(np.array2string(parameter.TxYI[indx], separator='')[1:-1], 2))
#     TxQ.append(int(np.array2string(parameter.TxYQ[indx], separator='')[1:-1], 2))
# Tx = np.round(DataNormalize(TxI, [], parameter.pamorder)) + 1j * np.round(DataNormalize(TxQ, [], parameter.pamorder))
# Rx = DataNormalize(np.real(parameter.Ysoft[:50000]), [], parameter.pamorder) + 1j * DataNormalize(
#     np.imag(parameter.Ysoft[:50000]), [], parameter.pamorder)
# snr_tek, evm_tek = SNR(Tx, Rx)
# ber = BER(evm_tek, parameter.pamorder)
# bercount = BERcount(Tx, Rx, parameter.pamorder)
# print(bercount, ber)
# Histogram2D_Hot("Tektronix_DSP_Y", Rx, snr_tek, evm_tek, bercount, path=cur_dir)