import matplotlib.pyplot as plt

x = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
ttv4 = [0.364097, 0.355826, 0.379337, 0.340162, 0.335352, 0.313056]
e_rider = [0.279246, 0.295744, 0.290798, 0.286799, 0.273785, 0.308202]  # <-- put E-RIDER data here
ttv2 = [0.334538, 0.372265, 0.404979, 0.469862, 0.488826, 0.430145]

fig = plt.figure(figsize=(5, 4))

plt.plot(x, ttv4, marker='o', linewidth=3, markersize=8, label='TTv4')
plt.plot(x, e_rider, marker='d', linewidth=3, markersize=8, label='E-RIDER')
plt.plot(x, ttv2, marker='s', linewidth=3, markersize=8, label='TTv2')

plt.ylim(0.1, 0.55)

plt.xlabel('Ref std', fontsize=18)
plt.ylabel('Train loss', fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15, loc='lower right')

plt.grid(True, linewidth=1)
plt.tight_layout()
plt.show()

fig.savefig('train_loss_vs_std_y0_55.png', dpi=300, bbox_inches='tight')
