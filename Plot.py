import matplotlib.pyplot as plt


a = [-18, -16, -14, -12, -10, -8, -6]
qpsk = [5.28, 6.18, 5.45, 4.25, 4.36, 5.18, 5.29]
qam = [9.92, 9.78, 8.77, 8.94, 7.92, 8.16, 8.01]
qam_cd = [9.98, 9.64, 8.14, 7.85, 7.21, 7.74, 7.59]


plt.figure()
plt.plot(a, qpsk, label="QPSK")
plt.plot(a, qam, label="16QAM w/o CD comp.")
plt.plot(a, qam_cd, label="16QAM w/ CD comp.")
plt.scatter(a, qpsk)
plt.scatter(a, qam)
plt.scatter(a, qam_cd)
plt.ylabel("EVM(%)")
plt.ylim(3, 12)
plt.xlabel("ROP(dBm)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
plt.savefig("cd.pdf")


over = [8, 10, 12, 14, 16, 18, 20]
e = [13.25, 11.97, 8.94, 7.45, 6.25, 6.15, 5.90]
plt.figure()
plt.plot(over, e, label="16QAM 80km ROP -12dBm")
plt.xlabel("EVM(%)")
plt.ylabel("overhead(%)")
plt.legend()
plt.grid(True)
plt.scatter(over, e, label="16QAM 80km ROP -12dBm")
plt.show()
plt.savefig("Overheadevm.jpg")



D = 16e-12/(1e-9 * 1e3)
lam = 1550e-9
Z = 80e3
C = 3e8
T = 1/56e9

N = 2* np.floor(D*(lam**2)*Z/(2*C*T**2))