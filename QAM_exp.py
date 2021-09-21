from _init_ import *
# initial Parameter

def main():
    parameter = Parameter()
    print('SymbolRate={}GBaud'.format(parameter.symbolRate / 1e9), 'Pamorder={}'.format(parameter.pamorder),
          'resamplenumber={}'.format(parameter.resamplenumber))
    print('Tx Length={}'.format(len(parameter.TxXI)), 'Rx Length={}'.format(len(parameter.RxXI)))
    # Tx Normalize
    Tx_Signal_X = np.array(Tx2Bit(parameter.TxXI, parameter.TxXQ, parameter.pamorder))
    Tx_Signal_Y = np.array(Tx2Bit(parameter.TxYI, parameter.TxYQ, parameter.pamorder))
    Histogram2D_Hot("TX", Tx_Signal_Y)
    Histogram2D_Hot("RX_", np.array(parameter.RxXI)+1j*np.array(parameter.RxXQ))
    # Rx Upsample
    Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(parameter.RxXI, up=parameter.upsamplenum, down=1),
                                 signal.resample_poly(parameter.RxXQ, up=parameter.upsamplenum, down=1),
                                 parameter.pamorder)
    Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(parameter.RxYI, up=parameter.upsamplenum, down=1),
                                 signal.resample_poly(parameter.RxYQ, up=parameter.upsamplenum, down=1),
                                 parameter.pamorder)

    print('Tx_Resample Length={}'.format(len(Tx_Signal_X)), 'Rx_Resample Length={}'.format(len(Rx_XI)))
    # prbs = np.ceil(DataNormalize(parameter.PRBS, [], parameter.pamorder))
    snrscan = np.zeros((parameter.resamplenumber, 1))
    evmscan = np.zeros((parameter.resamplenumber, 1))
    #Eye position scan
    start, end = 3, 4
    for eyepos in range(start, end):
        Rx_XI_eye = signal.resample_poly(Rx_XI[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_XQ_eye = signal.resample_poly(Rx_XQ[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_YI_eye = signal.resample_poly(Rx_YI[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_YQ_eye = signal.resample_poly(Rx_YQ[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_Signal_X = Rx_XI_eye + 1j * Rx_XQ_eye
        Rx_Signal_Y = Rx_YI_eye + 1j * Rx_YQ_eye
        Histogram2D_Hot('Rx', Rx_Signal_X[0:100000])

        # for eyeskew in range(0, 16, 1):
        #     Rx_YI_eye = signal.resample_poly(Rx_YI[eyepos+eyeskew:], up=1, down=parameter.resamplenumber)
        #     Rx_YQ_eye = signal.resample_poly(Rx_YQ[eyepos+eyeskew:], up=1, down=parameter.resamplenumber)
        #     Rx_Signal_Y = Rx_YI_eye + 1j * Rx_YQ_eye
        #     cma = CMA(Rx_Signal_X[0:100000], Rx_Signal_Y[0:100000], taps=[21, 21], iter=30)
        #     cma.RDE_CMA_butterfly()
        #     Rx_X_CMA, Rx_Y_CMA = cma.rx_x_single, cma.rx_y_single
        #     Histogram2D('tapscan/' + str(eyepos) + 'CMA1Y' + str(eyeskew), Rx_Y_CMA[0:100000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA2Y' + str(tap), Rx_Y_CMA[100000:200000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA3Y' + str(tap), Rx_Y_CMA[200000:300000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA4Y' + str(tap), Rx_Y_CMA[300000:400000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA5Y' + str(tap), Rx_Y_CMA[400000:500000])
        #     Histogram2D('tapscan/' + str(eyepos) + 'CMA1X' + str(eyeskew), Rx_X_CMA[0:100000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA2X' + str(tap), Rx_X_CMA[100000:200000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA3X' + str(tap), Rx_X_CMA[200000:300000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA4X' + str(tap), Rx_X_CMA[300000:400000])
        #     # Histogram2D('tapscan/' + str(eyepos) + 'CMA5X' + str(tap), Rx_X_CMA[400000:500000])



        cma = CMA(Rx_Signal_X, Rx_Signal_Y, taps=[21, 21], iter=50)
        cma.RDE_CMA_butterfly()
        Rx_X_CMA, Rx_Y_CMA = cma.rx_x_single, cma.rx_y_single
        print('CMA Batch Size={}'.format(cma.batchsize), 'CMA Stepsize={}'.format(cma.stepsize),
              'CMA OverHead={}%'.format(cma.overhead * 100))
        Histogram2D_Hot(str(eyepos) + 'CMA1X', Rx_X_CMA)
        Histogram2D_Hot(str(eyepos) + 'CMA1Y', Rx_Y_CMA)



        # Histogram2D(str(eyepos) + '+CMA2Y', Rx_Y_CMA[100000:200000])
        # Histogram2D(str(eyepos) + '+CMA2X', Rx_X_CMA[100000:200000])
        # Histogram2D(str(eyepos) + '+CMA3', Rx_Y_CMA[200000:300000])
        # Histogram2D('exp/'+str(eyepos) + '+CMA4', Rx_Y_CMA[300000:400000])
        # Histogram2D('exp/'+str(eyepos) + '+CMA5', Rx_Y_CMA[400000:500000])
        # Rx_X_CMA = IQimbaCompensator(Rx_X_CMA[0:100000], 1e-4)
        # Rx_Y_CMA = DataNormalize(np.real(Rx_Y_CMA), [], parameter.pamorder) + 1j * DataNormalize(
        #     np.imag(Rx_Y_CMA), [], parameter.pamorder)


        ### freq. compensation
        # freqoffsetX = Phaserecovery(Rx_X_CMA)
        # Rx_X_FO = freqoffsetX.FFT_FOE()
        # Rx_X_FO = DataNormalize(np.real(Rx_X_FO), [], parameter.pamorder) + 1j * DataNormalize(
        #     np.imag(Rx_X_FO), [], parameter.pamorder)

        freqoffsetX = Phaserecovery(Rx_X_CMA[0:100000])
        Rx_X_FO = freqoffsetX.FFT_FOE()
        freqoffsetY = Phaserecovery(Rx_Y_CMA[0:100000])
        Rx_Y_FO = freqoffsetY.FFT_FOE()
        Histogram2D_Hot("FreqOffsetCompX", Rx_X_FO)
        Histogram2D_Hot("FreqOffsetCompY", Rx_Y_FO)

        # pll = Phaserecovery(Rx_X_FO)
        # pll_out = pll.PLL()
        # Histogram2D_Hot("PLL", pll_out[0])

        Rx_X_FO = DataNormalize(np.real(Rx_X_FO), [], parameter.pamorder) + 1j * DataNormalize(
            np.imag(Rx_X_FO), [], parameter.pamorder)
        Rx_Y_FO = DataNormalize(np.real(Rx_Y_FO), [], parameter.pamorder) + 1j * DataNormalize(
            np.imag(Rx_Y_FO), [], parameter.pamorder)


        phase_viterbi = Phaserecovery(Rx_X_FO, taps=51)
        vv_out = phase_viterbi.QPSK_partition_VVPE(inner=2.3, outter=3.8)
        ct_out = phase_viterbi.CT(vv_out, taps=11)
        # mlx_out = phase_viterbi.ML(ct_out, taps=9)
        emlx_out = phase_viterbi.Enhanced_ML(ct_out, taps=51)
        Histogram2D_Hot(str(eyepos) + "E_ML_X", emlx_out)

        phase_viterbi = Phaserecovery(Rx_Y_FO, taps=51)
        vv_out = phase_viterbi.QPSK_partition_VVPE(inner=2.3, outter=3.8)
        ct_out = phase_viterbi.CT(vv_out, taps=11)
        # mly_out = phase_viterbi.ML(ct_out, taps=9)
        emly_out = phase_viterbi.Enhanced_ML(ct_out, taps=51)
        Histogram2D_Hot(str(eyepos) + "E_ML_Y", emly_out)

        ### data correlation
        cor_data = emlx_out
        cor_data= DataNormalize(np.real(cor_data), [], parameter.pamorder) + 1j * DataNormalize(
            np.imag(cor_data), [], parameter.pamorder)

        # Tx_corr, Rx_corr = corr(Tx_Signal_X, Rx_X_recovery[0], parameter.Prbsnum)
        Tx_real, Rx_real, c = corr(np.real(Tx_Signal_X), np.real(cor_data), parameter.Prbsnum)
        Tx_imag, Rx_imag, c = corr(np.imag(Tx_Signal_X), np.imag(cor_data), parameter.Prbsnum)
        length = min([len(Rx_real), len(Rx_imag)])
        Tx_corr = Tx_real[:length] + 1j * Tx_imag[:length]
        Rx_corr = Rx_real[:length] + 1j * Rx_imag[:length]
        Histogram2D_Hot('Rx_corr_X', Rx_corr)
        snr, evm = SNR(Tx_corr, Rx_corr)
        ber = BER(evm, parameter.pamorder)
        bercount = BERcount(Tx_corr, Rx_corr, parameter.pamorder)
        print("Eyepos:", eyepos)
        print("BERcount:", bercount, "BER:", ber)
        print("SNR:", snr, "EVM:", evm)
        snrscan[eyepos] = snr
        evmscan[eyepos] = evm
    print(np.max(snrscan), np.argmax(snrscan))

    # NN equalizer
    # parameter
    taps = 31
    batch_size = 500
    LR = 1e-3
    EPOCH = 300
    overhead = 0.2
    trainnum = int(len(Rx_corr) * overhead)
    device = torch.device("cuda:0")
    train_inputr = rolling_window(torch.Tensor(Rx_corr.real), taps)[:trainnum]
    train_inputi = rolling_window(torch.Tensor(Rx_corr.imag), taps)[:trainnum]
    train_targetr = torch.Tensor(Tx_corr.real[taps // 2:-taps // 2 + 1])[:trainnum]
    train_targeti = torch.Tensor(Tx_corr.imag[taps // 2:-taps // 2 + 1])[:trainnum]
    train_tensor = Data.TensorDataset(train_inputr, train_inputi, train_targetr, train_targeti)
    train_loader = Data.DataLoader(train_tensor, batch_size=batch_size, shuffle=False)

    val_inputr = rolling_window(torch.Tensor(Rx_corr.real), taps)
    val_inputi = rolling_window(torch.Tensor(Rx_corr.imag), taps)
    val_targetr = torch.Tensor(Tx_corr.real[taps // 2:-taps // 2 + 1])
    val_targeti = torch.Tensor(Tx_corr.imag[taps // 2:-taps // 2 + 1])
    val_tensor = Data.TensorDataset(val_inputr, val_inputi, val_targetr, val_targeti)
    val_loader = Data.DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    layer_widths = [16, 16, 1]
    layer_centres = [16, 16]
    basis_func = gaussian
    final_modelr, final_modeli = Network(layer_widths, layer_centres, basis_func, taps).to(device), Network(
        layer_widths, layer_centres, basis_func, taps).to(device)
    final_optr = torch.optim.Adam(final_modelr.parameters(), lr=LR)
    final_opti = torch.optim.Adam(final_modeli.parameters(), lr=LR)
    # modelx = conv1dResNet(Residual_Block, [2, 2, 2, 2]).to(device)
    # lossxr = nn.MSELoss()
    # lossxi = nn.MSELoss()
    # lossxc = nn.CrossEntropyLoss()
    # opty = torch.optim.Adam(modely.parameters(), weight_decay=1e-2, lr=LR)
    L = []
    val_L = []
    for epoch in tqdm(range(EPOCH)):
        for i, (dr, di, txr, txi) in enumerate(train_loader):
            final_modelr.train()
            final_modeli.train()
            outr, outi = final_modelr(dr.to(device)), final_modeli(di.to(device))
            # outr,outi= final_modelr(dr.unsqueeze(1).to(device)),final_modeli(di.unsqueeze(1).to(device))
            # trr,tri = harddecision(txr,txi)
            Lossr = nn.MSELoss()(outr.squeeze().cpu(), torch.Tensor(txr))
            Lossi = nn.MSELoss()(outi.squeeze().cpu(), torch.Tensor(txi))
            Loss = Lossr + Lossi
            final_optr.zero_grad()
            Lossr.backward(retain_graph=True)
            final_optr.step()
            final_opti.zero_grad()
            Lossi.backward()
            final_opti.step()
            L.append(Loss.detach().cpu().numpy())
            # modely.eval()
        print("Train Loss:{:.3f}".format(Loss),
              '||' "Train Bercount:{:.3E}".format(BERcount(txr + 1j * txi, outr.cpu() + 1j * outi.cpu(), 4)[0]))
        # print('\n training Accx: %f\n training Accy: %f\n' % (np.mean(Accx), np.mean(Accy)))
        # Accx = []
        # # Accy = []
        predictr, predicti = [], []
    final_modelr.eval()
    final_modeli.eval()
    for i, (dr, di, txr, txi) in enumerate(val_loader):
        outr, outi = final_modelr(dr.to(device)), final_modeli(di.to(device))
        predictr.extend(outr.cpu().detach().numpy())
        predicti.extend(outi.cpu().detach().numpy())
        # Lossxr = lossxr(outr.cpu(), txr)
        # Lossxi = lossxi(outi.cpu(), txi)
        # Lossyr = lossyr(outy[:, 0].cpu(), tyr)
        # Lossyi = lossyi(outy[:, 1].cpu(), tyi)
        # Lossxc = lossxc(outcx.cpu(), tgx)
        # Lossyc = lossyc(outcy.cpu(), tgy)
        # xacc = (tgx.eq(torch.max(outcx.cpu(), 1)[1])).sum() / outcx.shape[0]
        # yacc = (tgy.eq(torch.max(outcy.cpu(), 1)[1])).sum() / outcy.shape[0]
        print("Val BERcount:{:.3E}".format(BERcount(txr + 1j * txi, outr.cpu() + 1j * outi.cpu(), 4)[0]))
    predictr = np.array(predictr).squeeze()
    predicti = np.array(predicti).squeeze()
    snr, evm = SNR(Tx_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze())
    bercount = BERcount(Tx_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze(), 4)
    print(snr, evm)
    Histogram2D_Hot("RBF-Net", (np.array(predictr) + 1j * np.array(predicti)).squeeze(), snr, evm, bercount)


    ## Equalizer
    ## complex volterra
    equalizer_complex = Equalizer(Tx_corr, Rx_corr, 3, [31, 31, 31], 0.3)
    Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()
    snr_complex, evm_complex = SNR(Tx_complex_volterra, Rx_complex_volterra)
    bercomplex = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
    Histogram2D("ComplexVolterra_X", Rx_complex_volterra, snr_complex, evm_complex, bercomplex)
    print(snr_complex, evm_complex)

    ## real volterra
    equalizer_real = Equalizer(np.real(Tx_corr), np.real(Rx_corr), 3, [31, 31, 31], 0.15)
    equalizer_imag = Equalizer(np.imag(Tx_corr), np.imag(Rx_corr), 3, [31, 31, 31], 0.15)
    Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
    snr_volterra, evm_volterra = SNR(Tx_real_volterra, Rx_real_volterra)
    bercount = BERcount(Tx_real_volterra, Rx_real_volterra, parameter.pamorder)
    Histogram2D_Hot("RealVolterra_X", Rx_real_volterra, snr_volterra, evm_volterra, bercount)

    # bercount = BERcount(Tx_real_volterra, Rx_real_volterra, parameter.pamorder)
    # print(bercount, berco  mplex)

    ParaRecord(eyepos=np.argmax(snrscan), cmataps=cma.cmataps, cmastepsize=cma.stepsize, cmaoverhead=cma.overhead,
               snr=np.max(snrscan), datapath=parameter.datapath, volterrataps=equalizer_real.taps, volterraoverhead=equalizer_real.overhead, snr_volterra=snr_volterra)

    ## software DSP results
    TxI = []
    TxQ = []
    for indx in range(50000):
        TxI.append(int(np.array2string(parameter.TxXI[indx], separator='')[1:-1], 2))
        TxQ.append(int(np.array2string(parameter.TxXQ[indx], separator='')[1:-1], 2))
    Tx = np.round(DataNormalize(TxI, [], parameter.pamorder)) + 1j * np.round(DataNormalize(TxQ, [], parameter.pamorder))
    Rx = DataNormalize(np.real(parameter.Xsoft[:50000]), [], parameter.pamorder) + 1j * DataNormalize(
        np.imag(parameter.Xsoft[:50000]), [], parameter.pamorder)
    snr_tek, evm_tek = SNR(Tx, Rx)
    ber = BER(evm_tek, parameter.pamorder)
    bercount = BERcount(Tx, Rx, parameter.pamorder)
    print(bercount, ber)
    Histogram2D("Tektronix_DSP_Y", Rx, snr_tek, evm_tek, bercount)

if __name__ == '__main__':
    main()