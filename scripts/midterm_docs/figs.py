import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"]="Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"]=False
GRAY="#C8C8C8"; INK="#000000"; ORG="#6E6E6E"; GRID="#E6E2DC"
def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#9A938A")
    ax.tick_params(colors="#4A453F", labelsize=9)
    ax.yaxis.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
# Fig 1: own eval set v1 test (first use) R@5 by type
types=["단일 수치\n(29)","연도 비교\n(36)","기업 비교\n(40)","서술 요약\n(29)"]
pub=[0.207,0.250,0.025,0.828]; ft=[0.517,0.500,0.075,0.862]; fin=[0.862,0.778,0.200,0.931]
fig,ax=plt.subplots(figsize=(6.4,3.0),dpi=200); style(ax)
import numpy as np
x=np.arange(4); w=0.26
for i,(v,c,l) in enumerate([(pub,GRAY,"공개 KURE + RRF·필터"),(ft,INK,"fine-tuning KURE + RRF·필터"),(fin,ORG,"+ 재정렬기 학습 (최종 v1)")]):
    b=ax.bar(x+(i-1)*w,v,w,color=c,label=l)
    for xx,vv in zip(x+(i-1)*w,v): ax.text(xx,vv+0.015,f"{vv:.2f}",ha="center",fontsize=7,color="#4A453F")
ax.set_xticks(x); ax.set_xticklabels(types,fontsize=9); ax.set_ylim(0,1.05); ax.set_ylabel("R@5",fontsize=9)
ax.legend(frameon=False,fontsize=8,loc="upper left",ncol=3,bbox_to_anchor=(0,1.13))
fig.tight_layout(); fig.savefig("fig1_type_r5.png"); plt.close(fig)
# Fig 2: FDB test dense R@5 by year present/absent
fig,ax=plt.subplots(figsize=(6.4,2.8),dpi=200); style(ax)
groups=["연도 있는 질문 (608)","연도 없는 질문 (1,339)"]
pubv=[0.627,0.584]; ftv=[0.712,0.571]; diff=["+8.6%p [+2.4, +15.1]","−1.3%p [−5.8, +3.1]"]
x=np.arange(2); w=0.32
ax.bar(x-w/2,pubv,w,color=GRAY,label="공개 KURE-v1"); ax.bar(x+w/2,ftv,w,color=INK,label="fine-tuning KURE-v1")
for xx,v in zip(x-w/2,pubv): ax.text(xx,v+0.012,f"{v:.3f}",ha="center",fontsize=8,color="#4A453F")
for xx,v in zip(x+w/2,ftv): ax.text(xx,v+0.012,f"{v:.3f}",ha="center",fontsize=8,color="#4A453F")
for xx,d,c in zip(x,diff,["#000000","#000000"]): ax.text(xx,0.80,"차이 "+d,ha="center",fontsize=9,color=c,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=9); ax.set_ylim(0,0.9); ax.set_ylabel("dense R@5",fontsize=9)
ax.legend(frameon=False,fontsize=8,loc="upper left",ncol=2,bbox_to_anchor=(0,1.14))
fig.tight_layout(); fig.savefig("fig2_year_r5.png"); plt.close(fig)
