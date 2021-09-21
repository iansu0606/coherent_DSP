from _init_ import *
# initial Parameter

def main():
    parameter = Parameter(simulation=True)
    print('SymbolRate={}'.format(parameter.symbolRate / 1e9), 'Pamorder={}'.format(parameter.pamorder),
          'resamplenumber={}'.format(parameter.resamplenumber))
    print('Tx Length={}'.format(len(parameter.TxXI_prbs)), 'Rx Length={}'.format(len(parameter.RxXI)/parameter.resamplenumber))

    # Tx Normalize
    start = 0
    Tx_X = PRBS2bit(parameter.TxXI_prbs, parameter.TxXQ_prbs)
    Tx_Y = PRBS2bit(parameter.TxYI_prbs, parameter.TxYQ_prbs)

    # Rx Normalize
    Rx_XI, Rx_XQ = DataNormalize(parameter.RxXI[start:], parameter.RxXQ[start:], parameter.pamorder)
    Rx_YI, Rx_YQ = DataNormalize(parameter.RxYI[start:], parameter.RxYQ[start:], parameter.pamorder)


    snrscan = np.zeros((parameter.resamplenumber, 1))
    evmscan = np.zeros((parameter.resamplenumber, 1))
    #Eye position scan
    start, end = 25, 26
    for eyepos in range(start, end):
        # Rx_XI_eye = Downsample(Rx_XI, parameter.resamplenumber, eyepos)
        # Rx_XQ_eye = Downsample(Rx_XQ, parameter.resamplenumber, eyepos)
        # Rx_YI_eye = Downsample(Rx_YI, parameter.resamplenumber, eyepos)
        # Rx_YQ_eye = Downsample(Rx_YQ, parameter.resamplenumber, eyepos)
        Rx_XI_eye = signal.resample_poly(Rx_XI[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_XQ_eye = signal.resample_poly(Rx_XQ[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_YI_eye = signal.resample_poly(Rx_YI[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_YQ_eye = signal.resample_poly(Rx_YQ[eyepos:], up=1, down=parameter.resamplenumber)
        Rx_Signal_X = Rx_XI_eye + 1j * Rx_XQ_eye
        Rx_Signal_Y = Rx_YI_eye + 1j * Rx_YQ_eye

        Histogram2D_Hot("TX", Tx_X)
        Histogram2D_Hot('Rx', Rx_Signal_X[0:32767])
        cma2 = CMA(Rx_Signal_X, Rx_Signal_Y, 61)
        # print('CMA Batch Size={}'.format(cma1.batchsize), 'CMA Stepsize={}'.format(cma1.stepsize),
        #       'CMA OverHead={}%'.format(cma1.overhead * 100))
        # cma1.CMA_single()
        # Histogram2D(str(eyepos) + '+CMA_single', cma1.rx_x_single)
        # cma2 = CMA(cma1.rx_x_single, cma1.rx_y_single)
        cma2.RDE_CMA_single()
        Rx_X_CMA, Rx_Y_CMA = cma2.rx_x_single, cma2.rx_y_single
        Histogram2D_Hot(str(eyepos) + '+CMA_RDE_X', Rx_X_CMA)
        Histogram2D_Hot(str(eyepos) + '+CMA_RDE_Y', Rx_Y_CMA)

        # Histogram2D(str(eyepos) + '+CMA_single', Rx_Y_CMA)
        # cma2 = CMA(cma_x_single, cma_y_single)
        # cma2.run_16qam_2()
        # cma2.run_16qam_3()
        # Rx_X_CMA = ConstModulusAlgorithm(Rx_Signal_X, 102, 10, 1e-6)
        # Rx_X_CMA, Rx_Y_CMA = Downsample(cma2.rx_x_cma, 1, cma2.center), Downsample(cma2.rx_y_cma, 1, cma2.center)
        # Rx_X_CMA, Rx_Y_CMA = Downsample(cma2.rx_x_cma, 1, cma2.center), Downsample(cma2.rx_y_cma, 1, cma2.center)
        # Rx_X_CMA = DataNormalize(np.real(Rx_X_CMA), [], parameter.pamorder) + 1j * DataNormalize(np.imag(Rx_X_CMA), [], parameter.pamorder)
        # Rx_Y_CMA = DataNormalize(np.real(Rx_Y_CMA), [], parameter.pamorder) + 1j * DataNormalize(np.imag(Rx_Y_CMA), [], parameter.pamorder)
        # print(cma2.costfunx[0][0:10])
        # Histogram2D(str(eyepos) + '+CMAX', Rx_X_CMA)
        # Histogram2D(str(eyepos) + '+CMAY', Rx_Y_CMA[0])
        # Rx_X_CMA = np.reshape(Rx_X_CMA,(-1, ))
        # Rx_Y_iqimba = IQimbaCompensator(Rx_Y_CMA[0], 1e-4)
        # Rx_X_iqimba = IQimbaCompensator(Rx_X_CMA, 1e-4)
        # Rx_X_iqimba = IQimbaCompensator(Rx_Y_CMA, 1e-4)
        # Histogram2D("IQimbaComp", Rx_X_iqimba[0])

        ### freq. compensation
        # freqoffsetX = Phaserecovery(Rx_X_CMA)
        # Rx_X_FO = freqoffsetX.FFT_FOE()
        # Rx_X_FO = DataNormalize(np.real(Rx_X_FO), [], parameter.pamorder) + 1j * DataNormalize(
        #     np.imag(Rx_X_FO), [], parameter.pamorder)
        # Histogram2D("FreqOffsetComp", Rx_X_FO)

        freqoffsetY = Phaserecovery(Rx_Y_CMA)
        Rx_Y_FO = freqoffsetY.FFT_FOE()
        # phaserec_Y = Phaserecovery(Rx_Y_FO)
        # Rx_Y_recovery = phaserec_Y.PLL()
        Rx_Y_FO = DataNormalize(np.real(Rx_Y_FO), [], parameter.pamorder) + 1j * DataNormalize(
            np.imag(Rx_Y_FO), [], parameter.pamorder)
        Histogram2D_Hot("FreqOffsetComp", Rx_Y_FO)
        # Histogram2D(str(eyepos) + '+PLL', Rx_Y_recovery[0])

        outr, outi = MstageBPS(Rx_Y_FO.real, Rx_Y_FO.imag, 64, 1, 71)
        Histogram2D_Hot("bps", outr+1j*outi)


        ###phase recovery
        phase_viterbi = Phaserecovery(Rx_Y_FO, taps=51)
        vv_out = phase_viterbi.QPSK_partition_VVPE(inner=2.3, outter=3.8)
        ct_out = phase_viterbi.CT(vv_out, taps=11)
        # ml_out = phase_viterbi.ML(ct_out, taps=39)
        eml_out = phase_viterbi.Enhanced_ML(ct_out, taps=51)
        Histogram2D_Hot(str(eyepos) + "E_ML", eml_out)

        # Histogram2D(str(eyepos) + '+CPE_Y', CPE_Y)
        # inputx = np.vstack((np.real(CPE_X), np.imag(CPE_X))).T
        # datar, datai = PCPE16QAM(inputx, 81)
        # datar, datai = MstageBPS(datar, datai, 11, 2, 64)
        # xx = datar + 1j * datai
        # Histogram2D(str(eyepos) + '+PCPE', xx)


        ### data correlation
        cor_data = eml_out
        cor_data= DataNormalize(np.real(cor_data), [], parameter.pamorder) + 1j * DataNormalize(
            np.imag(cor_data), [], parameter.pamorder)

        Tx_real, Rx_real, c = corr(np.imag(Tx_Y), np.real(cor_data), parameter.Prbsnum)
        Tx_imag, Rx_imag, c = corr(np.real(Tx_Y), np.imag(cor_data), parameter.Prbsnum)
        length = min([len(Rx_real), len(Rx_imag)])
        Tx_corr = Tx_real[:length] + 1j * Tx_imag[:length]
        Rx_corr = Rx_real[:length] + 1j * Rx_imag[:length]
        Histogram2D_Hot('Rx_corr', Rx_corr)
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
    #parameter
    taps = 41
    batch_size = 500
    LR = 1e-3
    EPOCH = 1000
    overhead = 0.1
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
    L =[]
    val_L= []
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
        print("Train Loss:{:.3f}".format(Loss),'||' "Train Bercount:{:.3E}".format(BERcount(txr + 1j*txi, outr.cpu() + 1j * outi.cpu(), 4)[0]))
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
        print("Val BERcount:{:.3E}".format(BERcount(txr + 1j*txi, outr.cpu() + 1j * outi.cpu(), 4)[0]))
    predictr = np.array(predictr).squeeze()
    predicti = np.array(predicti).squeeze()
    snr, evm = SNR(Tx_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze())
    bercount = BERcount(Tx_corr[taps // 2:-taps // 2 + 1], (np.array(predictr) + 1j * np.array(predicti)).squeeze(), 4)
    print(snr, evm)
    Histogram2D("RBF-Net", (np.array(predictr) + 1j * np.array(predicti)).squeeze(), snr ,evm, bercount)






    ParaRecord(eyepos=np.argmax(snrscan), cmataps=cma2.cmataps, cmastepsize=cma2.stepsize, cmaoverhead=cma2.overhead, snr=np.max(snrscan), datapath=parameter.datafolder)
    equalizer_complex = Equalizer(Tx_corr, Rx_corr, 3, [101, 3, 3], 0.2)
    # equalizer_real = Equalizer(np.real(Tx_corr), np.real(Rx_corr), 3, [51, 21, 41], 0.5)
    # equalizer_imag = Equalizer(np.imag(Tx_corr), np.imag(Rx_corr), 3, [51, 21, 41], 0.5)
    # Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    # Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    # Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    # Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
    Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()
    # snr_volterra, evm_volterra = SNR(Tx_real_volterra, Rx_real_volterra)
    snr_complex, evm_complex = SNR(Tx_complex_volterra, Rx_complex_volterra)
    bercomplex = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
    # bercount = BERcount(Rx_complex_volterra, Tx_complex_volterra, parameter.pamorder)
    # print(snr_complex, evm_complex)
    # Histogram2D_Hot("RealVolterra", Rx_real_volterra, snr_volterra, evm_volterra, bercount)
    Histogram2D_Hot("ComplexVolterra", Rx_complex_volterra, snr_complex, evm_complex, bercomplex)
    # bercount = BERcount(Tx_real_volterra, Rx_real_volterra, parameter.pamorder)
    # print(bercount, bercomplex)
if __name__ == '__main__':
    main()