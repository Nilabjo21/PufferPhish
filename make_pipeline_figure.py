"""Draw the G65 detection pipeline figure for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

fig = plt.figure(figsize=(10.5, 6.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

# Palette: professional blues/grays
COLORS = {
    "input": "#E3F2FD",
    "process": "#FFFFFF",
    "model": "#FFF3E0",
    "outcome": "#E8F5E8",
    "note": "#FAFAFA",
    "text": "#212121",
    "border": "#1565C0",
    "subborder": "#90A4AE",
}

def box(x, y, w, h, label, subtitle=None, color="process", fs=10.5, bold=False, wght=550):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.4, edgecolor=COLORS["border"] if color != "process" else COLORS["subborder"],
                       facecolor=color, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2 + (0 if subtitle else 0), label, ha="center", va="center",
            fontsize=fs, fontweight=bold, color=COLORS["text"], zorder=3, family="Arial")
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.38, subtitle, ha="center", va="center",
                fontsize=8.6, color="#424242", zorder=3, family="Arial")

def arrow(x1, y1, x2, y2, color="#1565C0", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=lw, color=color, zorder=4,
                                 shrinkA=3, shrinkB=3))

# Figure layout (y grows upward)
y0, dy_main = 2.6, 1.02
# Row 1: input + collection
box(0.12, y0+dy_main, 1.22, 1.02, "Input URL\n/Webpage", "one example at a time", color=COLORS["input"])
box(1.50, y0+dy_main, 1.34, 1.02, "Data Collection\n& Preprocessing", "84 illustrative URLs\n(42 phishing / 42 legit)", color=COLORS["process"])
box(3.00, y0+dy_main, 1.42, 1.02, "Feature Extraction\n(URL branch)", "url_length only\n(implemented)", color=COLORS["process"])
box(4.58, y0+dy_main, 1.42, 1.02, "Individual\nClassifiers", "LR / RF / SVM /\nXGBoost", color=COLORS["model"])
box(6.16, y0+dy_main, 1.22, 1.02, "Score Fusion\n(Fixed weights)", "LR 0.25 + RF 0.35\n+ XGBoost 0.40", color=COLORS["outcome"])
box(7.52, y0+dy_main, 1.30, 1.02, "Prediction\n& Label", "Phishing / Legitimate\nwith confidence", color=COLORS["outcome"])
for i, x in enumerate([1.40, 2.90, 4.48, 6.06, 7.42]):
    arrow(x, y0+dy_main+0.51, x+0.04, y0+dy_main+0.51)

# Row 2: text/context branch (proposed)
box(1.50, y0+0.34, 1.34, 0.62, "Text/NLP Branch\n(proposed)", "DistilBERT\nembeddings", color=COLORS["note"])
box(3.00, y0+0.34, 1.42, 0.62, "Context Branch\n(proposed)", "page/DOM signals,\nBitB indicators", color=COLORS["note"])
arrow(2.85, y0+0.93, 4.52, y0+1.06)
arrow(1.38, y0+0.93, 2.96, y0+1.06)

# Row 3: explanation + prototype (proposed/implemented)
box(0.12, y0, 1.22, 0.70, "Explanation\n(SHAP)", "proposed feature\nattribution", color=COLORS["note"])
box(7.52, y0, 1.30, 0.70, "Prototype\nOutput", "Streamlit demo\n(implemented)", color=COLORS["outcome"])
arrow(1.28, y0+0.35, 1.46, y0+0.74)
arrow(7.38, y0+0.74, 7.22, y0+0.35)

ax.text(0.12, 0.06, "Figure 1. Proposed AI-based phishing detection pipeline. Only the URL-length branch was implemented and evaluated; text/NLP and context branches are proposed (see Sections 3 and 5).",
        fontsize=9.2, color="#37474F", family="Arial", style="italic")

plt.savefig(r"C:\Users\Nilabjo\G65_Phishing_Detection_Project\report_figures\pipeline_diagram.png",
            bbox_inches="tight", facecolor="white", dpi=200)
plt.close()
print("Saved pipeline_diagram.png")
