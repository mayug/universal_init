"""Generate methodology figure for the platonic init summary report."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={'wspace': 0.35})
fig.patch.set_facecolor('white')

# Colors
C_TEACHER = '#3B7DD8'
C_STUDENT = '#D4652A'
C_LOSS    = '#5B9A2E'
C_DATA    = '#BDBDBD'
C_EMBED   = '#8E44AD'
C_PROJ    = '#C0784A'
C_WHITEN  = '#F0D86E'
C_MEL     = '#E8A87C'
C_DOWN    = '#EEEEEE'

BOX_LW = 1.5

def draw_box(ax, cx, cy, w, h, label, color, fontsize=10, text_color='white',
             bold=False, alpha=0.9, edge_color='#444444'):
    x = cx - w / 2
    y = cy - h / 2
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=BOX_LW, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=weight, zorder=3)

def arrow(ax, x1, y1, x2, y2, color='#444444', lw=1.8, dashed=False):
    style = 'dashed' if dashed else 'solid'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=style, shrinkA=2, shrinkB=2),
                zorder=1)

# ============================================================
# Panel A: Vision Pipeline
# ============================================================
ax = axes[0]
ax.set_xlim(-1, 11)
ax.set_ylim(-2.5, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a)  Vision Distillation', fontsize=14, fontweight='bold', pad=16)

# --- Data ---
draw_box(ax, 5, 10, 4.5, 1.0, 'ImageNet Images  (1.28M)', C_DATA,
         fontsize=9, text_color='#333', alpha=0.45)

# Fork arrows
arrow(ax, 3.5, 9.45, 2.5, 8.7)
arrow(ax, 6.5, 9.45, 7.5, 8.7)

# --- Teacher (left column, cx=2.5) ---
draw_box(ax, 2.5, 7.9, 4.0, 1.3, 'CLIP ViT-B/16\n(frozen  \u2744)', C_TEACHER,
         fontsize=10, bold=True)
arrow(ax, 2.5, 7.2, 2.5, 6.3)
draw_box(ax, 2.5, 5.6, 3.6, 1.1, 'Teacher Embedding\n768-dim', C_EMBED, fontsize=9)

# --- Student (right column, cx=7.5) ---
draw_box(ax, 7.5, 7.9, 4.0, 1.3, 'RegNetY-400MF\n4.3M params', C_STUDENT,
         fontsize=10, bold=True)
arrow(ax, 7.5, 7.2, 7.5, 6.3)
draw_box(ax, 7.5, 5.6, 3.6, 1.1, 'Projector\n440 \u2192 768', C_PROJ,
         fontsize=9, alpha=0.75)
arrow(ax, 7.5, 5.0, 7.5, 4.1)
draw_box(ax, 7.5, 3.4, 3.6, 1.1, 'Student Embedding\n768-dim', C_EMBED,
         fontsize=9, alpha=0.75)

# --- Loss (center bottom) ---
arrow(ax, 2.5, 5.0, 5.0, 2.1, dashed=True, color='#666666')
arrow(ax, 7.5, 2.8, 5.0, 2.1, dashed=True, color='#666666')
draw_box(ax, 5.0, 1.4, 3.8, 1.2, 'Cosine  +  \u03bb\u00b7CKA  Loss', C_LOSS,
         fontsize=10, bold=True)

# --- Downstream ---
arrow(ax, 7.5, 2.8, 7.5, -0.2, color=C_STUDENT, lw=2.2)
draw_box(ax, 7.5, -1.0, 5.0, 1.0,
         'Downstream:  LP / FT\n@ 1%,  10%,  100% labels', C_DOWN,
         fontsize=8.5, text_color='#333', alpha=0.55)

# ============================================================
# Panel B: Audio Pipeline (cross-modal)
# ============================================================
ax = axes[1]
ax.set_xlim(-1, 11)
ax.set_ylim(-2.5, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(b)  Cross-Modal Audio Distillation', fontsize=14, fontweight='bold', pad=16)

# --- Separate data sources ---
draw_box(ax, 2.2, 10, 3.8, 1.0, 'Text Captions  (46K)', C_DATA,
         fontsize=9, text_color='#333', alpha=0.45)
draw_box(ax, 7.8, 10, 3.8, 1.0, 'Audio Clips  (46K)', C_DATA,
         fontsize=9, text_color='#333', alpha=0.45)

arrow(ax, 2.2, 9.45, 2.2, 8.7)
arrow(ax, 7.8, 9.45, 7.8, 8.7)

# --- Teacher (left) ---
draw_box(ax, 2.2, 7.9, 4.0, 1.3, 'CLIP Text  /  SBERT\n(frozen  \u2744)', C_TEACHER,
         fontsize=10, bold=True)
arrow(ax, 2.2, 7.2, 2.2, 6.5)

# Whitening step
draw_box(ax, 2.2, 5.85, 3.8, 0.9, 'Whiten  (if cos > 0.5)', C_WHITEN,
         fontsize=8.5, text_color='#555', alpha=0.9, edge_color='#AAA')
arrow(ax, 2.2, 5.35, 2.2, 4.7)

draw_box(ax, 2.2, 4.0, 3.6, 1.1, 'Teacher Embedding\n768-dim', C_EMBED, fontsize=9)

# --- Student (right) ---
# Mel spectrogram preprocessing
draw_box(ax, 7.8, 7.9, 4.0, 0.9, 'Mel Spectrogram', C_MEL,
         fontsize=9, text_color='#444', alpha=0.85, edge_color='#AAA')
arrow(ax, 7.8, 7.4, 7.8, 6.7)
draw_box(ax, 7.8, 5.85, 4.0, 1.3, 'MobileNetV3-Large\n3M params', C_STUDENT,
         fontsize=10, bold=True)
arrow(ax, 7.8, 5.15, 7.8, 4.5)
draw_box(ax, 7.8, 3.8, 3.6, 1.1, 'Student Embedding\n768-dim', C_EMBED,
         fontsize=9, alpha=0.75)

# --- Cross-modal label ---
ax.text(5.0, 7.0, 'text  \u2192  audio\ncross-modal', fontsize=9, ha='center', va='center',
        color='#888888', fontstyle='italic')

# --- Loss ---
arrow(ax, 2.2, 3.4, 5.0, 1.7, dashed=True, color='#666666')
arrow(ax, 7.8, 3.2, 5.0, 1.7, dashed=True, color='#666666')
draw_box(ax, 5.0, 1.0, 3.8, 1.2, 'Cosine  +  \u03bb\u00b7CKA  Loss', C_LOSS,
         fontsize=10, bold=True)

# --- Downstream ---
arrow(ax, 7.8, 3.2, 7.8, -0.2, color=C_STUDENT, lw=2.2)
draw_box(ax, 7.8, -1.0, 5.0, 1.0,
         'Downstream:  ESC-50\nLP / FT  @ 1%,  10%,  100%', C_DOWN,
         fontsize=8.5, text_color='#333', alpha=0.55)

plt.savefig('reports/figures/method_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/method_overview.pdf', bbox_inches='tight', facecolor='white')
print("Saved to reports/figures/method_overview.{png,pdf}")
