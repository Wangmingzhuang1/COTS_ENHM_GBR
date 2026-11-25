"""论文结果图的 A4 统一画布、分级字体和矢量导出接口。"""
from pathlib import Path
import re
import numpy as np
import matplotlib as mpl
from matplotlib.text import Text

A4_PORTRAIT = (210 / 25.4, 297 / 25.4)   # 8.27 x 11.69 in
A4_LANDSCAPE = (297 / 25.4, 210 / 25.4)  # 11.69 x 8.27 in

# 统一分级字号体系（按 A4 1:1 排版设计）
FONT_PANEL = 13.0
FONT_TITLE = 11.0
FONT_LABEL = 10.5
FONT_TICK = 9.5
FONT_LEGEND = 9.0
FONT_ANNOT = 9.0


def apply_base_style():
    """配置全局 Times New Roman 与矢量导出参数。"""
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': FONT_TICK,
        'axes.titlesize': FONT_TITLE,
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_LABEL,
        'axes.labelweight': 'bold',
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': FONT_LEGEND,
        'legend.title_fontsize': FONT_LEGEND,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
    })


def _layout(fig, name):
    axes = fig.axes
    if name.startswith('Figure1'):
        # 紧致上下构图：地图 12°×15° (aspect 1.25)，w=0.80 -> h=0.707
        positions = [
            [0.10, 0.250, 0.80, 0.707],
            [0.10, 0.065, 0.80, 0.140],
        ]
    elif name.startswith('Figure2'):
        # 恢复旧稿 1×4 谱图 + 1 通栏直方图布局，居中舒展于 A4
        gap = 0.018
        total_w = 0.86
        w_top = (total_w - 3 * gap) / 4.0
        x0 = 0.095
        positions = [
            [x0 + i * (w_top + gap), 0.560, w_top, 0.350] for i in range(4)
        ] + [
            [x0, 0.100, total_w, 0.355]
        ]
        for i, ax in enumerate(axes[:4]):
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlim(3, 30)
            ax.set_ylim(0, 1.08)
            if i == 0:
                ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
                ax.set_ylabel('Normalized Power')
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')
    elif name.startswith('Figure3'):
        # 严格对齐旧稿 2×2 紧凑布局：A/C 在左 + 右侧竖向 colorbar，B/D 在右（Y 轴均在左）
        # C 为地图 (aspect 1.25)：w=0.350 -> h=0.310
        positions = [
            [0.082, 0.555, 0.350, 0.345],  # A: Hovmöller
            [0.618, 0.555, 0.345, 0.345],  # B: Circular-linear phase
            [0.082, 0.115, 0.350, 0.310],  # C: Meridional velocity diff map
            [0.618, 0.115, 0.345, 0.310],  # D: Regional period boxplot
            [0.444, 0.580, 0.014, 0.295],  # cbar A
            [0.444, 0.140, 0.014, 0.260],  # cbar C
        ]
        for i in (4, 5):
            axes[i].yaxis.set_label_position('right')
            axes[i].yaxis.set_ticks_position('right')
    elif name.startswith('Figure4'):
        # 4 面板紧凑 2×2：上排地图配旧稿左右外侧竖向 colorbar，下排径流箱线图与叶绿素时序图
        # 地图 (aspect 1.25): w=0.320 -> h=0.283
        positions = [
            [0.155, 0.565, 0.320, 0.283],  # A: Rainfall map
            [0.560, 0.565, 0.320, 0.283],  # B: Chl-a map
            [0.085, 0.595, 0.014, 0.222],  # cbar A (left)
            [0.892, 0.595, 0.014, 0.222],  # cbar B (right)
            [0.115, 0.140, 0.365, 0.315],  # C: River discharge boxplot
            [0.560, 0.140, 0.365, 0.315],  # D: Chl-a vs -IPO time series
        ]
    elif name.startswith('FigureS1'):
        # 横向 A4 单图，保留脚本自身布局
        return
    elif name.startswith('FigureS2'):
        positions = [
            [0.12, 0.690, 0.82, 0.210],
            [0.12, 0.390, 0.82, 0.210],
            [0.12, 0.090, 0.82, 0.210],
        ]
    elif name.startswith('FigureS3'):
        # 两幅 GBR 地图并排，w=0.40 -> h=0.354
        positions = [
            [0.09, 0.330, 0.40, 0.354],  # A: R^2 map
            [0.11, 0.245, 0.36, 0.016],  # cbar A
            [0.54, 0.330, 0.40, 0.354],  # B: Period map
            [0.56, 0.245, 0.36, 0.016],  # cbar B
        ]
        axes[0].set_anchor('C')
        axes[2].set_anchor('C')
    elif name.startswith('FigureS5'):
        # 3×2 谱图对比（对应 SI Fig. S4）
        positions = [
            [x, y, 0.38, 0.215]
            for y in (0.690, 0.390, 0.090)
            for x in (0.11, 0.56)
        ]
    elif name.startswith('FigureS6'):
        # SI Fig. S5: 上排全宽 R(k) 混叠诊断谱 (A) + 下排左右两幅相位-纬度图 (B: 全 58 礁, C: 2 年滞后 17 礁)
        positions = [
            [0.11, 0.565, 0.83, 0.355],  # A: R(k) curves
            [0.11, 0.110, 0.37, 0.345],  # B: All 58 reefs unconditional
            [0.57, 0.110, 0.37, 0.345],  # C: Lag=2 yr (17 reefs, k>0 branch)
        ]
    else:
        raise ValueError(f'No A4 layout defined for {name}')

    assert len(positions) == len(axes), (name, len(positions), len(axes))
    for ax, pos in zip(axes, positions):
        ax.set_position(pos)
    if fig._suptitle is not None:
        if name.startswith('FigureS3'):
            fig._suptitle.set_position((0.5, 0.775))
        elif not name.startswith('FigureS1'):
            fig._suptitle.set_position((0.5, 0.965))


def save_publication(fig, stem, orientation='portrait'):
    """统一 A4 页面、Times New Roman 字体体系，并将标量图像矢量化后导出 PNG/PDF/SVG。"""
    stem = Path(stem).with_suffix('')
    apply_base_style()

    page = A4_LANDSCAPE if orientation == 'landscape' else A4_PORTRAIT
    fig.set_size_inches(*page, forward=True)
    fig.set_facecolor('white')
    fig.set_layout_engine(None)
    _layout(fig, stem.name)

    for ax in fig.axes:
        if not stem.name.startswith('FigureS1'):
            ax.set_facecolor('white')
        limits = ax.get_xlim(), ax.get_ylim()
        for image in list(ax.images):
            array = np.ma.asarray(image.get_array())
            assert array.ndim == 2, 'Color imagery must not be misrepresented as vector data'
            left, right, bottom, top = image.get_extent()
            if image.origin == 'upper':
                array = np.flipud(array)
            mesh = ax.pcolormesh(
                np.linspace(left, right, array.shape[1] + 1),
                np.linspace(bottom, top, array.shape[0] + 1),
                array,
                shading='flat',
                cmap=image.get_cmap(),
                norm=image.norm,
                zorder=image.get_zorder(),
                rasterized=False,
            )
            if image.colorbar is not None:
                image.colorbar.update_normal(mesh)
            image.remove()
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    fig.canvas.draw()

    # 统一字体家族，但尊重各层级字号而非一刀切压成 8 pt
    for artist in fig.findobj():
        if hasattr(artist, 'set_rasterized'):
            artist.set_rasterized(False)
        if isinstance(artist, Text):
            artist.set_fontfamily('Times New Roman')
            artist.set_math_fontfamily('custom')

    if not stem.name.startswith('FigureS1'):
        for ax in fig.axes:
            ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
            for title in (ax.title, ax._left_title, ax._right_title):
                if title.get_text():
                    title.set_fontsize(FONT_TITLE)
                    title.set_fontweight('bold')
            for label in (ax.xaxis.label, ax.yaxis.label):
                if label.get_text():
                    label.set_fontsize(FONT_LABEL)
                    label.set_fontweight('bold')
            for text in ax.texts:
                if re.fullmatch(r'[A-F]', text.get_text()):
                    text.set_fontsize(FONT_PANEL)
                    text.set_fontweight('bold')
            legend = ax.get_legend()
            if legend:
                for t in legend.get_texts():
                    t.set_fontsize(FONT_LEGEND)
                if legend.get_title().get_text():
                    legend.get_title().set_fontsize(FONT_LEGEND)
                    legend.get_title().set_fontweight('bold')

    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(13.0)
        fig._suptitle.set_fontweight('bold')

    fig.canvas.draw()
    # 禁用自动裁边：三个格式均保持精确的 A4 页面尺寸
    for suffix in ('.svg', '.pdf', '.png'):
        fig.savefig(
            stem.with_suffix(suffix),
            dpi=300,
            bbox_inches=None,
            facecolor='white',
            transparent=False,
        )
    print(f'A4 ({orientation}) vector export: {stem.name}')
