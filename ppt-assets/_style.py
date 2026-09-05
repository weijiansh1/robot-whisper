"""PPT 配图样式:白底,蓝主调+暖色对比,纯黑粗体大字,图幅紧凑"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FONT="/home/jovyan/.fonts/NotoSansSC.ttf"
fm.fontManager.addfont(FONT)
FAM=fm.FontProperties(fname=FONT).get_name()
plt.rcParams.update({
    "font.family":FAM, "font.weight":"bold",
    "axes.labelweight":"bold", "axes.titleweight":"bold",
    "axes.unicode_minus":False,
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "savefig.dpi":240,
    "font.size":15,            # 基准字号显著提高
})
C = dict(
    blue="#3573B9", navy="#000000", sky="#9EC5E8", mist="#E4EEF8",
    orange="#E07B39", rust="#B4472E", sand="#EFD3B4",
    grey="#4A5560", ink="#000000", line="#C8D0D8",
)
def clean(ax, grid=None, spines=("left","bottom")):
    for k,sp in ax.spines.items():
        sp.set_visible(k in spines); sp.set_color(C["line"]); sp.set_linewidth(1.1)
    ax.tick_params(colors=C["ink"], labelsize=14, length=0, pad=5)
    for lb in ax.get_xticklabels()+ax.get_yticklabels(): lb.set_fontweight("bold")
    if grid:
        ax.grid(axis=grid, color=C["line"], lw=.8); ax.set_axisbelow(True)
    return ax
def title(fig, text=None, sub=None):
    """图内不写结论性标题;标题一律在 PPT 里排。保留函数以兼容调用。"""
    return None
