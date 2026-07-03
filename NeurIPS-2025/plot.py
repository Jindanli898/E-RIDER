# import numpy as np
# import matplotlib.pyplot as plt


# Qk = np.array([
#     0.1233395031, 0.1290568127, 0.1345363671, 0.1336156992,
#     0.1390938219, 0.1553822337, 0.2501239396, 0.2474388853,
#     0.2438520328, 0.2426946133, 0.2475763809, 0.2487656343,
#     0.2423451875, 0.2464166407, 0.2508773935, 0.2503201008,
#     0.2440727348, 0.2477898596, 0.2530250717, 0.2413646508,
#     0.2446114049, 0.2506227322, 0.2500224489, 0.2458351325,
#     0.2383822974, 0.2461984981, 0.24888449, 0.2481250516,
#     0.2518420701, 0.2572144463, 0.251133747, 0.2541794868,
#     0.2415386322, 0.2505759133
# ])


# loss = np.array([
#     2.33, 2.303886, 2.302621, 2.222567,
#     0.748577, 0.385678, 0.287873, 0.235978,
#     0.197241, 0.169053, 0.145790, 0.128637,
#     0.115075, 0.105151, 0.097295
# ])


# epochs_Q = np.linspace(0, 14, len(Qk))
# epochs_loss = np.arange(15)


# gt_sp_mean = 0.25

# plt.rcParams.update({
#     "font.size": 16,
#     "axes.titlesize": 20,
#     "axes.labelsize": 18,
#     "legend.fontsize": 16,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16
# })

# plt.figure(figsize=(7.0, 4.6))


# plt.plot(epochs_Q, Qk, marker="o", linewidth=3.0, label="Qk")
# plt.plot(epochs_loss, loss, marker="s", linewidth=3.0, label="Loss")
# plt.axhline(y=gt_sp_mean, linestyle="--", linewidth=2.5,
#             label="Ground Truth SP")

# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.title("Qk approaches ground truth SP while loss decreases")

# plt.legend(frameon=True)
# plt.grid(True, linewidth=1.2)

# plt.xlim(0, 14)
# plt.margins(x=0.02, y=0.08)
# plt.tight_layout(pad=0.25)

# plt.savefig("Qk_Loss_SP_alignment_2to1.png", dpi=300, bbox_inches="tight")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

mu_r = np.array([
    -0.002365,
    -0.013586,
    -0.151310,
    -0.067504,
    -0.030447,
    -0.336492,
])

loss_A = np.array([
    0.018708,
    0.018708,
    0.034834,
    0.015539,
    0.019803,
    0.066609,
])

loss_B = np.array([
    0.010829,
    0.059916,
    2.29,
    2.29,
    1.97,
    2.29,
])


idx = np.argsort(mu_r)
mu_sorted = mu_r[idx]
loss_A_sorted = loss_A[idx]
loss_B_sorted = loss_B[idx]

x_pos = np.abs(mu_sorted)  

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

plt.figure(figsize=(6, 4.5))

plt.plot(x_pos, loss_A_sorted, marker="o", linewidth=3,
         label=r"Loss (on-the-fly)")
plt.plot(x_pos, loss_B_sorted, marker="s", linewidth=3,
         label=r"Loss (offline)")

ax = plt.gca()
ax.set_xscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{v:.3f}" for v in mu_sorted])

plt.xlabel(r"$\mu_r$ of offset")
plt.ylabel("Loss")
plt.title(r"Loss vs $\mu_r$ for two algorithms")


plt.grid(True, linewidth=1.0, which="major")

plt.legend(loc="upper left", frameon=True)

plt.tight_layout(pad=0.3)

plt.savefig("loss_vs_mu_r_two_algs_sorted_logx_clean.png",
            dpi=300, bbox_inches="tight")
plt.show()
