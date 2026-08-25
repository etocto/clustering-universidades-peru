"""
15_figura_overall.py
====================
Genera la Figura 1 del manuscrito: marco metodologico del pipeline v7.
Los valores mostrados deben coincidir con los que reportan Methods y Results.

Uso:
    python src/15_figura_overall.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import FIG_DIR

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans"})

PURPLE="#8F86D6"; GREEN="#2E9E7E"; AMBER="#C07C10"
HEAD={"Data sources":PURPLE,"Linkage":PURPLE,"Indicator matrix":GREEN,
      "Reduction":GREEN,"Partition":AMBER,"Profiles":AMBER,"Validation":AMBER}
TINT={PURPLE:"#CFC9EC",GREEN:"#A9DCC9",AMBER:"#C07C10"}
DARK="#1A1A1A"; ARROW="#9A9A9A"
LH=0.30; HH=0.52; PAD=0.26

def height(lines):
    n=sum(0.55 if l=="" else 1 for l in lines)
    return HH + PAD*2 + n*LH

fig,ax=plt.subplots(figsize=(15.6,5.4))
ax.set_xlim(0,15.6); ax.set_ylim(0,5.4); ax.axis("off")

def box(x,ytop,w,title,lines,fs=10.2):
    h=height(lines); y=ytop-h
    c=HEAD[title]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="white",ec=c,lw=2.0,zorder=3))
    ax.add_patch(FancyBboxPatch((x,ytop-HH),w,HH,boxstyle="round,pad=0.02,rounding_size=0.10",
        fc=TINT[c],ec=c,lw=2.0,zorder=4))
    ax.text(x+w/2,ytop-HH/2,title,ha="center",va="center",
            fontsize=11.6,fontweight="bold",color="black" if c!=AMBER else "white",zorder=6)
    yy=ytop-HH-PAD-LH/2
    for l in lines:
        if l=="": yy-=LH*0.55; continue
        ax.text(x+w/2,yy,l,ha="center",va="center",fontsize=fs,color=DARK,zorder=5)
        yy-=LH
    return y,h

def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
        mutation_scale=16,color=ARROW,lw=1.9,zorder=2))

TOP=5.15
b1=box(0.10,TOP,2.50,"Data sources",
   ["SUNEDU registers","faculty . enrolment","graduation, 2025-I","",
    "CONCYTEC RENACYT","Apr 2026 extraction"])
b2=box(2.92,TOP,2.32,"Linkage",
   ["Token-sort fuzzy","match (>=82)","83/99 automatic","16 manual review","",
    "Equivalence dictionary"],fs=9.9)
b3=box(5.56,TOP,2.42,"Indicator matrix",
   ["96 x 20","","8 employment","10 enrolment","2 formal coupling","",
    "3 postgraduate schools","held out as","a priori stratum","",
    "Density winsorised","at 100%"],fs=9.9)
b4=box(8.30,TOP,2.30,"Reduction",
   ["z-standardisation","","PCA, 90% var.","-> 12 components","",
    "PC1 19.8%   PC2 14.1%","","Retention criterion","robust (ARI = 1.000)"],fs=9.9)
b5=box(10.92,TOP,2.28,"Partition",
   ["k-means++","50 restarts, seed 42","k = 3","","ASW = 0.221","",
    "Ward . GMM . DBSCAN","as cross-checks"],fs=9.9)
b6=box(13.42,TOP,1.95,"Profiles",
   ["P1  n = 9","P2  n = 49","P3  n = 38","","+ postgraduate","stratum (n = 3)"],fs=9.7)
b7=box(13.42,b6[0]-0.34,1.95,"Validation",
   ["Kruskal-Wallis + Dunn","Specification","sensitivity (S0-S4)","Attrition test","",
    "Employment axis","replicated across","3 waves (9 indicators)"],fs=8.9)

y1=b1[0]+b1[1]/2; y2=b2[0]+b2[1]/2; y3=b3[0]+b3[1]/2
y4=b4[0]+b4[1]/2; y5=b5[0]+b5[1]/2
arrow(2.60,y1,2.92,y2)
arrow(5.24,y2,5.56,y3)
arrow(7.98,y3,8.30,y4)
arrow(10.60,y4,10.92,y5)
arrow(13.10,y5+0.28,13.42,b6[0]+b6[1]*0.62)
arrow(13.10,y5-0.28,13.42,b7[0]+b7[1]*0.72)

bot=min(b3[0],b5[0],b7[0])
ax.text(7.7,bot-0.30,"All steps reproduce deterministically from the archived repository (fixed seed)",
        ha="center",fontsize=10.2,color="#6C6C6C",style="italic")
ax.set_ylim(bot-0.55,5.35)
fig.savefig(FIG_DIR / "overall.png",dpi=300,bbox_inches="tight",facecolor="white")
print("overall.png generado")
