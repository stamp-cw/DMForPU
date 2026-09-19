from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/WWFCA_DIRECTIONAL_RESIDUAL_ARCHITECTURE.png"
fig, ax = plt.subplots(figsize=(19, 10), dpi=180)
ax.set_xlim(0, 19); ax.set_ylim(0, 10); ax.axis('off')

def box(x, y, w, h, text, color, fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                       facecolor=color, edgecolor='#263238', linewidth=1.2)
    ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, wrap=True)
    return (x+w, y+h/2), (x, y+h/2)

def arrow(a, b, label=None, color='#37474f', rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, linewidth=1.3,
                                 color=color, connectionstyle=f'arc3,rad={rad}'))
    if label: ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+0.16, label, ha='center', fontsize=8, color=color)

ax.text(9.5, 9.65, 'WWFCA directional high-frequency residual diffusion network', ha='center', fontsize=18, weight='bold')
ax.text(9.5, 9.25, 'Current experiments: wwfca_v41_noattn · x0/epsilon prediction · bounded residual injection', ha='center', fontsize=10, color='#455a64')

inp = box(.3, 7.55, 1.45, .75, 'x_t\nnoisy phase', '#bbdefb')
unet = box(2.25, 7.35, 2.4, 1.15, 'HF UNet backbone\n128/128/128/128\n2 ResBlocks/level', '#90caf9')
down = box(5.25, 7.55, 1.55, .75, '32→16\nlocation', '#64b5f6')
tail = box(15.95, 7.35, 2.25, 1.15, 'remaining UNet\nencoder + decoder', '#90caf9')
out = box(17.0, 5.7, 1.5, .75, 'epsilon or x0\nprediction', '#c5e1a5')
arrow(inp[0], unet[1]); arrow(unet[0], down[1]); arrow(down[0], tail[1]); arrow(tail[0], out[1])

split = box(5.15, 4.9, 1.75, .75, 'Haar DWT\nLL,LH,HL,HH', '#ffe082')
base = box(7.6, 7.55, 1.55, .75, 'base(x)\noriginal downsample', '#b3e5fc')
arrow(down[0], base[1], color='#0277bd')
arrow(down[0], split[1], color='#ef6c00', rad=.12)

bands = []
for i, (txt, y) in enumerate([('LH\nDW3×3+PW1×1', 3.75), ('HL\nDW3×3+PW1×1', 2.75), ('HH\nDW3×3+PW1×1', 1.75)]):
    bands.append(box(7.55, y, 1.75, .72, txt, '#ffcc80'))
    arrow((6.9, 5.22), bands[-1][1], color='#ef6c00')
fuse = box(10.15, 2.7, 2.1, 1.25, 'concat + high_fuse\n1×1 conv + GN', '#ffb74d')
for b in bands: arrow(b[0], fuse[1], color='#ef6c00')
cond = box(7.35, .55, 2.25, .78, 't + sigma\nFiLM condition', '#ce93d8')
film = box(10.15, 1.25, 2.1, .9, 'FiLM scale/shift\nthen output 1×1', '#ba68c8')
arrow(cond[0], film[1], color='#7b1fa2'); arrow(fuse[0], film[1], color='#7b1fa2', rad=-.12)
res = box(13.0, 2.05, 1.8, .85, 'residual r', '#f48fb1')
inj = box(13.0, 6.0, 1.8, .85, 'gamma·r\n0≤gamma≤0.1', '#ef9a9a')
add = box(15.0, 6.0, 1.4, .85, 'base +\ngamma·r', '#a5d6a7')
arrow(film[0], res[1], color='#7b1fa2'); arrow(res[0], inj[1], color='#c62828'); arrow(base[0], add[1], color='#2e7d32'); arrow(inj[0], add[1], color='#c62828')
arrow(add[0], tail[1], color='#2e7d32', rad=.12)

ax.text(1.0, 5.0, 'Residual branch', fontsize=11, weight='bold', color='#e65100')
ax.text(7.4, 0.15, 'Current no-attention ablation: LL is decomposed but does not form Q/K/V; only directional high-frequency features are processed.', fontsize=9, color='#37474f')
fig.tight_layout(); fig.savefig(OUT, bbox_inches='tight'); print(OUT)
