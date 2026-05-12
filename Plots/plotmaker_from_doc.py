import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data from the table
# -----------------------------
data = [
    [0.05, 5,      4,   1.00010218,  1.00017535,   0.96901362],
    [0.05, 10,     24,  1.00025681,  1.00051370,   0.84239086],
    [0.05, 50,     640, 0.99791737,  1.00009293,   0.89401480],

    [0.07, 5.7,    2,   0.999998197, 1.00008244815, 0.8520611224],
    [0.07, 10.5,   8,   0.99970317,  0.999617601,  0.97924620105],
    [0.07, 51.5,   170, 0.999946046, 0.99754992681, 0.9132453747],
    [0.07, 98.4,   500, 1.0026175698, 0.9976554103, 1.0515211568],

    [0.10, 5,      0.3, 1.000017896, 0.9999913,    1.076974554],
    [0.10, 10.3,   1.5, 0.999938806, 1.00003018184, 1.13430488],
    [0.10, 52.7,   40,  0.998912321, 0.9997139,    1.0653898],
    [0.10, 101.06, 120, 1.0012683,   1.0000535,    0.877066],
]

df = pd.DataFrame(
    data,
    columns=[
        "epsilon",
        "|P|/|Q|",
        "k",
        "Cost(Q,C)/Cost(P,C)",
        "Cost(Q,C')/Cost(P,C')",
        "Cost(P,C')/Cost(P,C)",
    ],
)

# Optional: sort nicely
df = df.sort_values(["epsilon", "|P|/|Q|"])

ratio_cols = [
    "Cost(Q,C)/Cost(P,C)",
    "Cost(Q,C')/Cost(P,C')",
    "Cost(P,C')/Cost(P,C)",
]

# -----------------------------
# Plot settings
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
})

# -----------------------------
# 1. Cost ratios vs |P|/|Q|
# -----------------------------
plt.figure()

for eps, group in df.groupby("epsilon"):
    plt.plot(
        group["|P|/|Q|"],
        group["Cost(Q,C)/Cost(P,C)"],
        marker="o",
        linewidth=2,
        label=f"ε = {eps}",
    )

plt.axhline(1, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("|P| / |Q|")
plt.ylabel("Cost(Q,C) / Cost(P,C)")
plt.title("Cost(Q,C) / Cost(P,C) vs |P| / |Q|")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ratio_QC_PC_vs_size_ratio.png", dpi=300)
plt.show()


# -----------------------------
# 2. Cost ratios with C' vs |P|/|Q|
# -----------------------------
plt.figure()

for eps, group in df.groupby("epsilon"):
    plt.plot(
        group["|P|/|Q|"],
        group["Cost(Q,C')/Cost(P,C')"],
        marker="o",
        linewidth=2,
        label=f"ε = {eps}",
    )

plt.axhline(1, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("|P| / |Q|")
plt.ylabel("Cost(Q,C') / Cost(P,C')")
plt.title("Cost(Q,C') / Cost(P,C') vs |P| / |Q|")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ratio_QCp_PCp_vs_size_ratio.png", dpi=300)
plt.show()


# -----------------------------
# 3. Cost(P,C') / Cost(P,C) vs |P|/|Q|
# -----------------------------
plt.figure()

for eps, group in df.groupby("epsilon"):
    plt.plot(
        group["|P|/|Q|"],
        group["Cost(P,C')/Cost(P,C)"],
        marker="o",
        linewidth=2,
        label=f"ε = {eps}",
    )

plt.axhline(1, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("|P| / |Q|")
plt.ylabel("Cost(P,C') / Cost(P,C)")
plt.title("Cost(P,C') / Cost(P,C) vs |P| / |Q|")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ratio_PCp_PC_vs_size_ratio.png", dpi=300)
plt.show()


# -----------------------------
# 4. Deviation from 1 for all cost ratios
# -----------------------------
df_dev = df.copy()
for col in ratio_cols:
    df_dev[col] = df_dev[col] - 1

plt.figure(figsize=(9, 5))

markers = ["o", "s", "^"]

for col, marker in zip(ratio_cols, markers):
    plt.scatter(
        df["|P|/|Q|"],
        df_dev[col],
        s=70,
        marker=marker,
        label=col,
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("|P| / |Q|")
plt.ylabel("Ratio - 1")
plt.title("Deviation of Cost Ratios from 1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cost_ratio_deviations.png", dpi=300)
plt.show()


# -----------------------------
# 5. Relationship between k and |P|/|Q|
# -----------------------------
plt.figure()

for eps, group in df.groupby("epsilon"):
    plt.scatter(
        group["|P|/|Q|"],
        group["k"],
        s=80,
        label=f"ε = {eps}",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("|P| / |Q|")
plt.ylabel("k")
plt.title("k vs |P| / |Q|")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("k_vs_size_ratio.png", dpi=300)
plt.show()


# -----------------------------
# 6. Optional: print summary statistics
# -----------------------------
print("\nSummary statistics by epsilon:")
print(df.groupby("epsilon")[ratio_cols + ["k"]].mean())