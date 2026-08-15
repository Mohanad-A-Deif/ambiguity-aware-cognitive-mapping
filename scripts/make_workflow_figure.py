"""Generate the complete daily decision-loop diagram used as manuscript Figure 3."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility" / "figures" / "fig01_decision_support_workflow.png"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8.0,
        "mathtext.fontset": "dejavuserif",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 600,
    }
)


def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    lines,
    *,
    face="0.96",
    edge="black",
    linestyle="-",
    title_size=7.9,
    body_size=6.8,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.006",
        linewidth=0.85,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)
    title_lines = title.count("\n") + 1
    ax.text(
        x + width / 2,
        y + height - 0.020,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        linespacing=1.02,
        multialignment="center",
        color="black",
        zorder=3,
    )
    if lines:
        title_fraction = 0.29 if title_lines == 1 else 0.38
        body_center = y + (height * (1.0 - title_fraction)) / 2
        ax.text(
            x + width / 2,
            body_center,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=body_size,
            linespacing=1.34,
            multialignment="center",
            color="black",
            zorder=3,
        )
    return box


def add_arrow(
    ax,
    start,
    end,
    *,
    label=None,
    label_xy=None,
    linestyle="-",
    connectionstyle="arc3,rad=0",
    linewidth=0.9,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=linewidth,
        linestyle=linestyle,
        color="black",
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=6.4,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            zorder=5,
        )
    return arrow


fig, ax = plt.subplots(figsize=(7.25, 5.45))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Section labels keep the physical and cognitive parts of the loop explicit.
ax.text(
    0.5,
    0.974,
    "Decision epoch $t$: physical transition, ambiguity-aware inference, and allocation",
    ha="center",
    va="top",
    fontsize=8.5,
    fontweight="bold",
)
ax.plot([0.015, 0.985], [0.946, 0.946], color="0.35", linewidth=0.55)

left_x, middle_x, right_x = 0.040, 0.360, 0.680
node_w = 0.280
top_y, top_h = 0.695, 0.215
middle_y, middle_h = 0.365, 0.225
bottom_y, bottom_h = 0.070, 0.205

add_box(
    ax,
    left_x,
    top_y,
    node_w,
    top_h,
    "1  Post arrivals\nand serve demand",
    [
        r"post scheduled $A_h^t,A_d^t$",
        r"realize $D^t$; serve $D^t+B^t$",
        r"update residual $I$, $B$, and $Q$",
    ],
    face="0.95",
    title_size=7.4,
    body_size=6.7,
)

add_box(
    ax,
    middle_x,
    top_y,
    node_w,
    top_h,
    "2  Physical inputs\nat decision time",
    [
        r"demand $D$; backlog $B$",
        r"hospital stock $I$; supply stock $Q$",
        r"route/lead $L$; capacity $C$",
        r"confidence $c$; expected inbound",
    ],
    face="0.95",
    title_size=7.4,
    body_size=6.55,
)

add_box(
    ax,
    right_x,
    top_y,
    node_w,
    top_h,
    "3  Four-channel\nevidence encoding",
    [
        r"five risk signals $p\in[0,1]$ with $c$",
        r"$e^t=(e_{\mathrm{T}},e_{\mathrm{F}},e_{\mathrm{PT}},e_{\mathrm{PF}})$",
        r"aggregate available sources",
        r"project each row onto $\mathcal{A}_i$",
    ],
    face="0.90",
    title_size=7.4,
    body_size=6.45,
)

add_box(
    ax,
    right_x,
    middle_y,
    node_w,
    middle_h,
    "4  Coupled ACM\ninner iteration",
    [
        r"$x^{m+1,t}=\mathcal{P}_{\mathcal{A}}[\rho_e e^t$",
        r"$\quad +(1-\rho_e)\Psi(B_Cx^{m,t})]$",
        r"fixed $e^t$; no reset within $m$",
        r"stop at $10^{-4}$ or 60 steps",
    ],
    face="0.87",
    title_size=7.4,
    body_size=6.35,
)

memory_x, memory_y, memory_w, memory_h = right_x, bottom_y, node_w, bottom_h
add_box(
    ax,
    memory_x,
    memory_y,
    memory_w,
    memory_h,
    "ACM state memory\nand initialization",
    [
        r"$x^{0,0}=(0.5,0.5,0.5,0.5)$",
        r"$x^{0,t}=x^{*,t-1}$ for $t>0$",
        r"new concept: neutral state",
    ],
    face="white",
    linestyle="--",
    title_size=7.2,
    body_size=6.35,
)

add_box(
    ax,
    middle_x,
    middle_y,
    node_w,
    middle_h,
    "5  Priority\ngeneration",
    [
        r"$y=\mathrm{T}-\mathrm{F}+\gamma(\mathrm{PT}-\mathrm{PF})$",
        r"blend state and direct evidence",
        r"$p^t=\operatorname{softmax}_h(y^t/\tau)$",
    ],
    face="0.91",
    title_size=7.4,
    body_size=6.55,
)

add_box(
    ax,
    left_x,
    middle_y,
    node_w,
    middle_h,
    "6  Form requests\nand allocate",
    [
        r"$R^t=[B+\widehat D-I-\mathrm{inbound}]_+$",
        r"shared LP or donor matching",
        r"stock, request, capacity, route limits",
    ],
    face="0.94",
    title_size=7.4,
    body_size=6.4,
)

add_box(
    ax,
    left_x,
    bottom_y,
    node_w,
    bottom_h,
    "7  Commit dispatch\nand schedule",
    [
        r"debit $q^t$ from available $Q$",
        r"hospital arrival at $t+L$",
        r"supplier-to-DC replenishment",
    ],
    face="0.95",
    title_size=7.4,
    body_size=6.55,
)

add_box(
    ax,
    middle_x,
    bottom_y,
    node_w,
    bottom_h,
    "Evaluation\noutput",
    [
        "service, fairness, agreement",
        "transport and convergence",
        "paired statistical analysis",
    ],
    face="white",
    linestyle=":",
    title_size=7.4,
    body_size=6.55,
)

# Main within-epoch serpentine flow.
mid_top = top_y + top_h / 2
add_arrow(ax, (left_x + node_w, mid_top), (middle_x, mid_top))
add_arrow(ax, (middle_x + node_w, mid_top), (right_x, mid_top), label="risk signals", label_xy=(0.660, mid_top + 0.028))
add_arrow(ax, (right_x + node_w / 2, top_y), (right_x + node_w / 2, middle_y + middle_h), label=r"$e^t$", label_xy=(0.835, 0.642))

# Persistent ACM memory: load the previous converged state, then store the new one.
add_arrow(
    ax,
    (right_x + 0.085, memory_y + memory_h),
    (right_x + 0.085, middle_y),
    label=r"load $x^{0,t}$",
    label_xy=(right_x + 0.050, 0.320),
)
add_arrow(
    ax,
    (right_x + 0.205, middle_y),
    (right_x + 0.205, memory_y + memory_h),
    label=r"store $x^{*,t}$",
    label_xy=(right_x + 0.242, 0.320),
    linestyle="--",
)

# Converged state to priorities, allocation, and commitment.
mid_middle = middle_y + middle_h / 2
add_arrow(ax, (right_x, mid_middle), (middle_x + node_w, mid_middle), label=r"$x^{*,t}$", label_xy=(0.660, mid_middle + 0.030))
add_arrow(ax, (middle_x, mid_middle), (left_x + node_w, mid_middle), label=r"$p^t$", label_xy=(0.340, mid_middle + 0.030))
add_arrow(ax, (left_x + node_w / 2, middle_y), (left_x + node_w / 2, bottom_y + bottom_h), label=r"$q^t$", label_xy=(0.205, 0.320))

# Evaluation is an output branch rather than a state transition.
add_arrow(
    ax,
    (left_x + node_w, bottom_y + bottom_h / 2),
    (middle_x, bottom_y + bottom_h / 2),
    label="record outcomes",
    label_xy=(0.340, bottom_y + bottom_h / 2 + 0.030),
    linestyle=":",
)

# Outer physical loop: committed shipments and state feed the next epoch.
ax.plot([left_x, 0.014], [bottom_y + bottom_h / 2, bottom_y + bottom_h / 2], color="black", linewidth=0.9, zorder=3)
ax.plot([0.014, 0.014], [bottom_y + bottom_h / 2, top_y + top_h / 2], color="black", linewidth=0.9, zorder=3)
add_arrow(ax, (0.014, top_y + top_h / 2), (left_x, top_y + top_h / 2), linewidth=0.9)
ax.text(
    0.022,
    0.485,
    r"arrivals and physical state for $t+1$",
    ha="center",
    va="center",
    rotation=90,
    fontsize=6.25,
    color="black",
)

ax.text(
    0.5,
    0.010,
    "Solid arrows: operational flow   |   Dashed arrows: cognitive-state persistence   |   Dotted arrow: evaluation only",
    ha="center",
    va="bottom",
    fontsize=6.25,
    color="0.20",
)

fig.tight_layout(pad=0.20)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
print(OUTPUT)
