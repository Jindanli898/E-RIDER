import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ------------------ data from your 3 rows ------------------
# each row: (states, {method: epochs_to_target_acc})
data = {
    2000: {"N=6000": 6, "N=8000": 6, "ours": 5},
    1000: {"N=4000": 3, "N=6000": 4, "N=8000": 4, "ours": 4},
    500:  {"N=4000": 3, "N=6000": 5, "N=8000": 5, "ours": 3},
}

# ------------------ assumptions / knobs ------------------
batch_size = 64
mnist_train_samples = 60000
steps_per_epoch = math.ceil(mnist_train_samples / batch_size)  # 938

ns = 1
desired_bl = 5

# Dimension-free per-device proxy (your choice A):
# pulses per device per minibatch update ~= desired_bl * ns
pulses_per_step = desired_bl * ns

def parse_Npulse(label: str) -> int:
    if label.startswith("N="):
        return int(label.split("=")[1])
    return 0

# ------------------ plotting setup ------------------
states_list = [2000, 1000, 500]  # preserve your row order
methods_order = ["N=4000", "N=6000", "N=8000", "ours"]

colors = {
    "N=4000": "tab:orange",
    "N=6000": "tab:green",
    "N=8000": "tab:red",
    "ours": "tab:blue",
}

fig, ax = plt.subplots(figsize=(6, 4))

group_x = np.arange(len(states_list))
bar_w = 0.18
gap = 0.02

offsets = {
    "N=4000": -1.5*(bar_w+gap),
    "N=6000": -0.5*(bar_w+gap),
    "N=8000":  0.5*(bar_w+gap),
    "ours":       1.5*(bar_w+gap),
}

method_handles = []
method_labels = []

for method in methods_order:
    xs = group_x + offsets[method]
    cal_vals = []
    train_vals = []

    for st in states_list:
        epochs = data[st].get(method, np.nan)
        if np.isnan(epochs):
            cal_vals.append(np.nan)
            train_vals.append(np.nan)
            continue

        cal = parse_Npulse(method)  # per-device calibration pulses (metric A)
        train = epochs * steps_per_epoch * pulses_per_step  # per-device training pulses
        cal_vals.append(cal)
        train_vals.append(train)

    cal_vals = np.array(cal_vals, float)
    train_vals = np.array(train_vals, float)

    ok = ~np.isnan(cal_vals)

    # solid (calibration)
    b1 = ax.bar(xs[ok], cal_vals[ok], width=bar_w,
                color=colors[method], edgecolor="black", linewidth=0.6)

    # hatched (training) stacked on top
    ax.bar(xs[ok], train_vals[ok], bottom=cal_vals[ok], width=bar_w,
           color=colors[method], edgecolor="black", linewidth=0.6,
           hatch="///", alpha=0.9)

    method_handles.append(b1[0])
    method_labels.append(method)

# legend for stack semantics
# ------------------ legend (2 columns, wide & short) ------------------
# stack semantics
stack_legend = [
    Patch(facecolor="white", edgecolor="black", label="calibration pulses", linewidth=0.6),
    Patch(facecolor="white", edgecolor="black", hatch="///", label="training pulses", linewidth=0.6),
]

# --- Legend layout you want: 3 rows × 2 cols ---
# Row1: (ours, N=4000)
# Row2: (N=6000, N=8000)
# Row3: (calibration pulses, training pulses)

# build a dict from label -> handle
handle_by_label = dict(zip(method_labels, method_handles))

key_to_handle = {
    "ours": handle_by_label["ours"],
    "N=4000": handle_by_label["N=4000"],
    "N=6000": handle_by_label["N=6000"],
    "N=8000": handle_by_label["N=8000"],
    "calibration pulses": stack_legend[0],
    "training pulses": stack_legend[1],
}

key_to_label = {
    "ours": "E-RIDER",
    "N=4000": "ZS(N=4000)",
    "N=6000": "ZS(N=6000)",
    "N=8000": "ZS(N=8000)",
    "calibration pulses": "calibration pulses",
    "training pulses": "training pulses",
}


rows = [
    ("ours", "N=4000"),
    ("N=6000", "N=8000"),
    ("calibration pulses", "training pulses"),
]

# NOTE: Matplotlib fills legend entries column-major when ncol>1,
# so we pack as: [col1 rows..., col2 rows...]
legend_handles = [key_to_handle[r[0]] for r in rows] + [key_to_handle[r[1]] for r in rows]
legend_labels  = [key_to_label[r[0]]  for r in rows] + [key_to_label[r[1]]  for r in rows]

ax.legend(
    legend_handles,
    legend_labels,
    ncol=2,
    loc="upper left",
    frameon=True,
    fontsize=12,
    columnspacing=1.6,
    handletextpad=0.6,
    borderpad=0.3,
    labelspacing=0.4,
)

# ------------------ y-axis label bigger ------------------
# ax.set_ylabel("Total pulses", fontsize=20)  # or 18/22 as you like
ax.tick_params(axis="y", labelsize=14)      # optional: make y tick labels bigger too

ax.set_xticks(group_x)
ax.set_xticklabels([f"{st} states" for st in states_list], fontsize=14)
ax.set_ylabel("Total pulses", fontsize=14)

# ax.set_title(
#     f"Calibration and training pulse cost "
#     f"(batch={batch_size}, steps/epoch={steps_per_epoch}, pulses/step={pulses_per_step})",
#     fontsize=12
# )

ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("pulse_cost_stacked_bars_per_device.png", dpi=250)
plt.show()
