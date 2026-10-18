"""
Shared plotting style for the ANR photocatalysis ML scripts.

Defines the colour palette, stage / weather / model colour maps, axis labels
for each feature code, and apply_style(), which sets the matplotlib rcParams
used by every figure. With STRIP_TITLES=1, descriptive figure titles are
removed at save time (as in the paper figures).
"""
import matplotlib.pyplot as plt
from cycler import cycler

PAL = {
    'teal':'#0E7C7B','coral':'#E2542C','indigo':'#3D348B','amber':'#F49D0C',
    'rose':'#C9184A','sky':'#3A86FF','green':'#2A9D8F','plum':'#7B2CBF',
    'slate':'#3F4756','black':'#1A1A1A',
}
OI = {
    'black':PAL['black'],'orange':PAL['amber'],'skyblue':PAL['sky'],
    'green':PAL['green'],'yellow':'#F4D35E','blue':PAL['teal'],
    'vermillion':PAL['coral'],'purple':PAL['plum'],'grey':PAL['slate'],
}

STAGE1=PAL['teal']; STAGE2=PAL['coral']
CV_PRIMARY=PAL['teal']; CV_STRESS=PAL['amber']
TARGET_LINE=PAL['coral']
GOOD=PAL['green']; BAD=PAL['coral']; NEUTRAL=PAL['slate']

WEATHER_COLOUR={'sunny_summer':PAL['coral'],'sunny_winter':PAL['sky'],'cloudy_winter':PAL['plum']}
SEASON_COLOUR={'Summer':PAL['coral'],'Winter':PAL['sky']}
WEATHERTYPE_COLOUR={'Sunny':PAL['coral'],'Cloudy':PAL['plum']}
DIURNAL_COLOUR={'Morning':PAL['sky'],'Midday':PAL['amber'],'Evening':PAL['plum']}
CAT_COLORS={'kinetic':PAL['teal'],'material':PAL['green'],'operational':PAL['coral'],'environmental':PAL['amber']}
MODEL_COLORS={
    'XGBoost':PAL['teal'],'LightGBM':PAL['amber'],'ExtraTrees':PAL['green'],
    'RandomForest':PAL['sky'],'GradientBoost':PAL['coral'],'SVR_RBF':PAL['plum'],
    'GaussianProcess':PAL['indigo'],'KNN':PAL['slate'],'Stacked':PAL['black'],
}
STAGE1_CMAP='GnBu'; STAGE2_CMAP='OrRd'
LUX_BIN_COLOUR={'low':PAL['plum'],'mid':PAL['teal'],'high':PAL['coral']}
SEQ=[PAL['teal'],PAL['coral'],PAL['indigo'],PAL['amber'],PAL['rose'],PAL['sky'],PAL['green'],PAL['plum'],PAL['slate']]

TIME_LABEL='Time Step (min)'
DEG_LABEL='Degradation efficiency (%)'
FEATURE_LABELS={
    'F1_CalcTemp':'Calcination temp (°C)','F2_CalcRamp':'Calcination ramp (°C/min)',
    'F3_CalcTime':'Calcination time (h)','F8_TimeStep':TIME_LABEL,
    'F9_RhBConc':'RhB concentration (×10⁻⁵ M)','F10_pH':'Solution pH',
    'F11_Ions':'Co-existing ions (ppm)','F12_Scavengers':'Radical scavengers',
    'SunlightLux':'Average sunlight intensity (lux)','Temperature_C':'Temperature (°C)',
    'F13_Degradation':DEG_LABEL,
}
def flabel(code): return FEATURE_LABELS.get(code, code)

# Model abbreviations for compact axis labels
MODEL_ABBR = {
    'XGBoost':'XGB', 'LightGBM':'LGBM', 'ExtraTrees':'ET', 'RandomForest':'RF',
    'GradientBoost':'GB', 'SVR_RBF':'SVR-RBF', 'GaussianProcess':'GR',
    'KNN':'KNN', 'Stacked':'SE',
}
def mabbr(name): return MODEL_ABBR.get(name, name)

def _maybe_strip_titles():
    """If STRIP_TITLES=1, remove descriptive titles and suptitles before saving.

    Short per-panel data labels that start with 't = ' are kept.
    """
    import os
    if os.environ.get('STRIP_TITLES') != '1':
        return
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import re as _re
    _orig = Axes.set_title
    _panel = _re.compile(r'^(\([a-z]\))')
    def _set_title(self, label='', *a, **k):
        if isinstance(label, str):
            st = label.strip()
            if st.startswith('t = '):
                return _orig(self, label, *a, **k)
            m = _panel.match(st)
            if m:
                return _orig(self, m.group(1), *a, **k)
        return _orig(self, '', *a, **k)
    Axes.set_title = _set_title
    Figure.suptitle = lambda self, *a, **k: None

def apply_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':12.5,'font.weight':'bold',
        'axes.titlesize':14.5,'axes.titlelocation':'left','axes.titleweight':'bold',
        'axes.labelsize':13,'axes.labelweight':'bold',
        'axes.edgecolor':'#2A2A2A','axes.linewidth':1.3,
        'axes.spines.top':True,'axes.spines.right':True,
        'axes.grid':True,'grid.alpha':0.30,'grid.linestyle':':','grid.linewidth':0.8,
        'axes.prop_cycle':cycler(color=SEQ),
        'legend.fontsize':11,'legend.framealpha':0.95,'legend.edgecolor':'#888888',
        'xtick.labelsize':11.5,'ytick.labelsize':11.5,
        'figure.dpi':110,'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white',
    })
    _maybe_strip_titles()
