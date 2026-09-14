#!/usr/bin/env python3
"""Create a report and actual-value figures from saved full-section captures."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'results/cuda-context-study/diagnostics/q2-q4-full-sections'
sys.path.insert(0, str(ROOT / '.run/paper-deps'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import markdown
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak, KeepTogether

FIGURES = HERE / 'figures'
Q4, Q2 = 'Q4_K_M', 'Q2_K'
PALETTE = {Q4: '#2a9d8f', Q2: '#8b5fbf'}
INK, MUTED, GRID = '#142c43', '#526274', '#e0e6ec'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


COUNTERS, ROOFS, OPERATOR_POINTS, OPCODES = [], [], [], []
OPERATOR_VALIDATION, LOOKUP = {}, {}


def load_historical_data():
    global COUNTERS, ROOFS, OPERATOR_POINTS, OPERATOR_VALIDATION, OPCODES, LOOKUP
    FIGURES.mkdir(exist_ok=True)
    COUNTERS = read_csv(SOURCE / 'all-hardware-metrics.csv')
    ROOFS = read_csv(SOURCE / 'roofline-values.csv')
    OPERATOR_POINTS = read_csv(SOURCE / 'operator-roofline-points.csv')
    OPERATOR_VALIDATION = json.loads((SOURCE / 'operator-roofline-validation.json').read_text())
    OPCODES = [r for r in read_csv(SOURCE / 'instruction-instances.csv')
               if r['metric'] == 'sass__inst_executed_per_opcode']
    LOOKUP = {(r['phase'], int(r['gpu']), r['format'], r['metric']): r for r in COUNTERS}
METRICS = {
    'time': ('gpu__time_duration.sum', 1e6, 'Kernel duration (ms)', 3),
    'instructions': ('sm__inst_executed.sum', 1e6, 'Executed instructions (M)', 3),
    'reads': ('dram__bytes_read.sum', 1e6, 'DRAM reads (MB)', 3),
    'tensor': ('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed', 1, 'Tensor pipeline (% peak, elapsed)', 2),
    'alu': ('sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed', 1, 'ALU pipeline (% peak, elapsed)', 2),
    'ipc': ('sm__inst_executed.avg.per_cycle_active', 1, 'IPC (active cycles)', 2),
    'occupancy': ('sm__warps_active.avg.pct_of_peak_sustained_active', 1, 'Achieved occupancy (%)', 2),
    'registers': ('launch__registers_per_thread', 1, 'Registers per thread', 0),
    'blocks': ('launch__occupancy_limit_registers', 1, 'Register-limited blocks per SM', 0),
}


def value(phase, key, fmt, gpu=0):
    metric, divisor, _, _ = METRICS[key]
    row = LOOKUP[phase, gpu, fmt, metric]
    assert row['available'] == 'True' and int(row['samples']) == 5
    return float(row['median']) / divisor


def pct(a, b):
    return 100 * (b / a - 1) if a else None


def signed(n):
    return f'{n:+.1f}%'.replace('-', '−') if n is not None else 'Not defined (0 → 0)'


def delta(phase, key, gpu=0):
    return signed(pct(value(phase, key, Q4, gpu), value(phase, key, Q2, gpu)))


def roof(fmt, gpu=0):
    rows = [r for r in ROOFS if r['phase'] == 'prefill' and int(r['gpu']) == gpu
            and r['format'] == fmt and r['arithmetic_path'] == 'INT8 Tensor Core dense'
            and r['memory_level'] == 'DRAM']
    assert len(rows) == 1 and int(rows[0]['samples']) == 5
    return rows[0]


def opcode_counts(phase, fmt, gpu=0):
    rows = [r for r in OPCODES if r['phase'] == phase and int(r['gpu']) == gpu and r['format'] == fmt]
    assert rows and all(r['available'] == 'True' and int(r['samples']) == 5 for r in rows)
    counts = {r['instance']:float(r['median']) for r in rows}
    assert len(counts) == len(rows)
    assert sum(counts.values()) == value(phase, 'instructions', fmt, gpu) * 1e6
    return counts


def instruction_categories(phase, fmt, gpu=0):
    counts = opcode_counts(phase, fmt, gpu)
    return {'FFMA':counts['FFMA'], 'I2FP':counts['I2FP'],
            'Other opcodes':sum(n for op,n in counts.items() if op not in ('FFMA','I2FP')),
            'All instructions':sum(counts.values())}


def instruction_share(phase, gpu=0):
    a,b = (instruction_categories(phase, fmt, gpu) for fmt in (Q4,Q2))
    return 100 * sum(b[op]-a[op] for op in ('FFMA','I2FP')) / (b['All instructions']-a['All instructions'])


def save(fig, stem):
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(FIGURES / f'{stem}.{suffix}', dpi=240, facecolor='white')
    plt.close(fig)


def bar_figure(phase):
    if phase == 'prefill':
        keys = ['ipc', 'alu', 'tensor', 'instructions', 'reads', 'time']
        titles = ['IPC (active cycles)', 'ALU pipeline', 'Tensor pipeline',
                  'Executed instructions', 'DRAM reads', 'Kernel duration']
        units = ['Instructions / active cycle', '% of peak, elapsed', '% of peak, elapsed',
                 'Million instructions', 'MB (decimal)', 'Milliseconds']
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.0))
        axes = axes.flat
        fig.text(.045, .95, 'Higher GPU activity ≠ useful work', fontsize=18, fontweight='bold', color=INK)
        fig.text(.045, .90, '70B gate/up · prefill · GPU 0 · median of five matched launches', fontsize=11, color=MUTED)
        fig.subplots_adjust(left=.065, right=.985, bottom=.13, top=.80, wspace=.39, hspace=.65)
    else:
        keys = ['time', 'instructions', 'reads', 'occupancy']
        titles = ['Kernel duration', 'Executed instructions', 'DRAM reads', 'Achieved occupancy']
        units = ['Microseconds', 'Million instructions', 'MB (decimal)', '% of active-cycle peak']
        fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.1))
        fig.text(.045, .94, 'Decode: memory savings with a longer kernel', fontsize=18, fontweight='bold', color=INK)
        fig.text(.045, .875, '70B gate/up · decode · GPU 0 · median of five matched launches', fontsize=11, color=MUTED)
        fig.subplots_adjust(left=.055, right=.985, bottom=.20, top=.73, wspace=.54)
    for ax, key, title, unit in zip(axes, keys, titles, units):
        vals = [value(phase, key, fmt) for fmt in (Q4, Q2)]
        if phase == 'decode' and key == 'time':
            vals = [v * 1000 for v in vals]
        ax.bar([0, 1], vals, width=.57, color=[PALETTE[Q4], PALETTE[Q2]], zorder=3)
        ax.set_title(title, loc='left', pad=10, fontsize=11, weight='bold', color=INK)
        ax.set_ylabel(unit, color=MUTED, fontsize=10)
        ax.set_xticks([0, 1], ['Q4_K_M', 'Q2_K'], fontsize=10)
        ax.set_xlim(-.62, 1.64)
        ax.set_ylim(0, max(vals) * 1.48)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, scale=max(vals): f'{v:,.0f}' if scale > 10 else f'{v:g}'))
        ax.grid(axis='y', color=GRID, zorder=0)
        ax.tick_params(axis='both', length=0, labelcolor=MUTED)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        for x, v in enumerate(vals):
            label = f'{v:,.3f}' if key == 'time' and phase == 'prefill' else f'{v:,.2f}' if key in ('time', 'tensor', 'occupancy', 'ipc', 'alu') else f'{v:,.1f}'
            ax.annotate(label, (x, v), xytext=(0, 6), textcoords='offset points', ha='center', color=INK, fontsize=11)
        ax.annotate(delta(phase, key), (1, vals[1]), xytext=(0, 23), textcoords='offset points',
                    ha='center', fontsize=12, fontweight='bold',
                    color=INK if key in ('ipc', 'alu') else '#19724b' if key == 'reads' else '#a03b28')
    fig.text(.045, .035 if phase == 'prefill' else .06,
             'Bar heights use actual values and separate unit scales. Labels above Q2 show relative change vs Q4 (%).', fontsize=10, color=MUTED)
    save(fig, f'{phase}-actual-values')


def roofline_figure():
    fig,ax=plt.subplots(figsize=(10.8,5.3))
    fig.subplots_adjust(left=.095,right=.97,bottom=.255,top=.79)
    fig.text(.06,.935,'Higher arithmetic intensity, lower tensor throughput',fontsize=17,weight='bold',color=INK)
    fig.text(.06,.87,'70B prefill gate/up · GPU 0 · dense INT8 Tensor Core path · Q4 and Q2 on one plot',fontsize=10.5,color=MUTED)
    memory_color,compute_color,boundary_color='#447D98','#537650','#B64D4D'
    point_styles={Q4:('#0072B2','o'),Q2:('#D55E00','D')}
    xlim,ylim=(30,4000),(15,430)
    rows={fmt:roof(fmt) for fmt in (Q4,Q2)}
    peaks={fmt:float(row['peak_work_ops_per_s'])/1e12 for fmt,row in rows.items()}
    bandwidths={fmt:float(row['peak_traffic_bytes_per_s'])/1e12 for fmt,row in rows.items()}
    ridges={fmt:peaks[fmt]/bandwidths[fmt] for fmt in (Q4,Q2)}
    low_ridge,high_ridge=min(ridges.values()),max(ridges.values())
    x=np.unique(np.concatenate([np.geomspace(*xlim,900),[low_ridge,high_ridge]]))
    lower_roof=np.minimum.reduce([np.minimum(peaks[fmt],bandwidths[fmt]*x) for fmt in (Q4,Q2)])
    ax.fill_between(x,ylim[0],lower_roof,where=x<=low_ridge,color='#EBF3F8',zorder=0)
    ax.fill_between(x,ylim[0],lower_roof,where=x>=high_ridge,color='#EDF4EA',zorder=0)
    ax.fill_between(x,ylim[0],lower_roof,where=(x>=low_ridge)&(x<=high_ridge),color='#F1F0E9',zorder=0)
    for fmt,style in ((Q4,'-'),(Q2,'-.')):
        ax.plot(x,x*bandwidths[fmt],color=memory_color,linestyle='--',linewidth=1.4,zorder=2)
        ax.axhline(peaks[fmt],color=compute_color,linestyle=style,linewidth=1.45,zorder=2)
        ax.axvline(ridges[fmt],color=boundary_color,linestyle=':',linewidth=1.3,zorder=2)
    ax.text(3820,358,f'Compute: Q4 {peaks[Q4]:.1f} · Q2 {peaks[Q2]:.1f} TOPS',ha='right',fontsize=9,color=compute_color)
    ax.text(460,249,f'{low_ridge:.0f}–{high_ridge:.0f} ops/B',fontsize=9,color=boundary_color)
    ax.text(47,43,'DRAM ≈729 GB/s',fontsize=9.5,color=memory_color,rotation=26)
    ax.text(37,17.5,'Memory ceiling lower',fontsize=9.5,color=memory_color)
    ax.text(485,17.5,'Compute ceiling lower',fontsize=9.5,color=compute_color)
    label_positions={Q4:(850,154),Q2:(1660,47)}
    for fmt in (Q4,Q2):
        row=rows[fmt]
        intensity=float(row['arithmetic_intensity_ops_per_byte'])
        tops=float(row['work_ops_per_s'])/1e12
        color,marker=point_styles[fmt]
        ax.scatter(intensity,tops,s=92,color=color,marker=marker,edgecolor='white',linewidth=1.0,zorder=5)
        ax.annotate(f'{fmt} · FFN gate/up\n{intensity:,.0f} ops/byte; {tops:.1f} TOPS',
                    (intensity,tops),xytext=label_positions[fmt],ha='center',va='center',
                    fontsize=10,color=color,weight='bold',
                    arrowprops={'arrowstyle':'-','color':color,'lw':.9,'shrinkA':4,'shrinkB':6},zorder=6)
    ax.set(xscale='log',yscale='log',xlim=xlim,ylim=ylim,
           xlabel='Tensor arithmetic intensity (INT8 operations / DRAM byte)',ylabel='Executed tensor throughput (TOPS)')
    ax.set_xticks([30,100,300,1000,3000], ['30','100','300','1,000','3,000'])
    ax.set_yticks([20,50,100,200,400], ['20','50','100','200','400'])
    ax.grid(which='major',lw=.5,alpha=.5,color='#C6C6C6')
    ax.tick_params(which='minor',length=2,color='#AAAAAA')
    fig.legend(handles=[
        Line2D([0],[0],color=memory_color,ls='--',label='DRAM ceiling'),
        Line2D([0],[0],color=compute_color,ls='-',label='Q4 compute ceiling'),
        Line2D([0],[0],color=compute_color,ls='-.',label='Q2 compute ceiling'),
        Line2D([0],[0],color=boundary_color,ls=':',label='Compute / memory boundaries'),
    ],loc='lower center',bbox_to_anchor=(.515,.085),ncol=4,frameon=False,fontsize=8.5,
        handlelength=2.5,columnspacing=1.45)
    fig.text(.06,.055,'Points and ceilings retain the original measurements. Nearby boundaries reflect each capture’s clock-dependent peak.',fontsize=9,color=MUTED)
    fig.text(.06,.022,'Region shading follows the lower of the two capture roofs; the narrow neutral band lies between their boundaries.',fontsize=9,color=MUTED)
    save(fig,'prefill-tensor-roofline')


def decode_register_figure():
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.0))
    fig.subplots_adjust(left=.065,right=.98,top=.775,bottom=.15,wspace=.40,hspace=.72)
    fig.text(.045,.95,'Decode: more instruction work and register pressure',fontsize=18,weight='bold',color=INK)
    fig.text(.045,.90,'70B gate/up · GPU 0 · N = 8 · median of five matched launches',fontsize=10.8,color=MUTED)
    fig.text(.045,.845,'A  Registers, block limits, and achieved occupancy',fontsize=11.5,weight='bold',color=INK)
    fig.text(.045,.478,'B  Instruction work, memory traffic, and kernel duration',fontsize=11.5,weight='bold',color=INK)
    specs=[
        ('registers','Registers per thread','Registers / thread',0,220,[0,50,100,150,200]),
        ('blocks','Register-limited blocks','Maximum blocks / SM',0,12,[0,2,4,6,8,10,12]),
        ('occupancy','Achieved occupancy','Active warps (% of peak)',2,48,[0,10,20,30,40]),
        ('instructions','Executed instructions','Million instructions',1,265,[0,50,100,150,200,250]),
        ('reads','DRAM reads','MB (decimal)',1,195,[0,50,100,150]),
        ('time','Kernel duration','Microseconds',2,595,[0,100,200,300,400,500]),
    ]
    for ax,(key,title,unit,digits,ceiling,ticks) in zip(axes.flat,specs):
        vals=[value('decode',key,fmt) for fmt in (Q4,Q2)]
        if key=='time':vals=[v*1000 for v in vals]
        ax.bar([0,1],vals,width=.55,color=[PALETTE[Q4],PALETTE[Q2]],zorder=3)
        ax.set_title(title,loc='left',pad=10,fontsize=11,weight='bold',color=INK)
        ax.set_ylabel(unit,fontsize=9.5,color=MUTED)
        ax.set_xticks([0,1],[Q4,Q2],fontsize=9.5)
        ax.set_yticks(ticks)
        ax.set_ylim(0,ceiling)
        ax.set_xlim(-.6,1.6)
        ax.grid(axis='y',color=GRID,zorder=0)
        ax.tick_params(axis='both',length=0,labelcolor=MUTED,labelsize=9)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        for x,v in enumerate(vals):
            ax.annotate(f'{v:,.{digits}f}',(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10.5,color=INK)
        ax.annotate(delta('decode',key),(1,vals[1]),xytext=(0,21),textcoords='offset points',
                    ha='center',fontsize=11,weight='bold',color='#19724b' if key=='reads' else INK)
    fig.text(.045,.082,'Actual values on separate, zero-based scales. Q2 labels show percentage change relative to Q4.',fontsize=9.5,color=MUTED)
    fig.text(.045,.043,'Higher register use and lower occupancy accompany the slowdown; the independent latency cost of occupancy is not isolated.',fontsize=9.5,color=INK)
    save(fig,'decode-register-pressure')


def operator_roofline_figure():
    styles={
        'FFN gate/up':('#0072B2','o'),'FFN down':('#D55E00','^'),
        'Q projection':('#7B3294','s'),'K projection':('#009E73','D'),
        'V projection':('#B76390','v'),'Attn. output':('#258FA6','h'),
        'FlashAttention':('#C13E3F','*'),'LM head':('#333333','P'),
    }
    positions={
        ('prefill',Q4):{
            'FFN gate/up':(110,15),'Q projection':(13,21),'K projection':(17,1.9),
            'V projection':(2.8,1.5),'Attn. output':(27,31),'FFN down':(2.8,5.0),
            'FlashAttention':(.36,.9)},
        ('prefill',Q2):{
            'FFN gate/up':(340,18),'Q projection':(77,25),'K projection':(150,3.6),
            'Attn. output':(38,3.1),'FFN down':(13,4.9),'FlashAttention':(.55,.35)},
        ('decode',Q4):{
            'FFN gate/up':(6.5,14),'Q projection':(1.8,11),'K projection':(3.8,1.2),
            'V projection':(.85,.85),'Attn. output':(4.8,25),'FFN down':(1.4,4.1),
            'FlashAttention':(.11,.22),'LM head':(3.2,2.1)},
        ('decode',Q2):{
            'FFN gate/up':(83,14),'Q projection':(34,25),'K projection':(62,2.2),
            'Attn. output':(15,.8),'FFN down':(27,1.35),'FlashAttention':(.42,.032)},
    }
    peak=float(OPERATOR_VALIDATION['peak_fp32_flops_per_s'])/1e12
    bandwidth=float(OPERATOR_VALIDATION['peak_dram_bytes_per_s'])/1e12
    ridge=peak/bandwidth
    assert len(OPERATOR_POINTS)==OPERATOR_VALIDATION['plotted_operator_configuration_groups']==31
    assert sum(int(p['samples']) for p in OPERATOR_POINTS)==OPERATOR_VALIDATION['captured_launches_used']==120

    def panel(ax,phase):
        xlim,ylim=(.03,1800),(.015,100)
        x=np.unique(np.append(np.geomspace(*xlim,900),ridge))
        bound=np.minimum(peak,bandwidth*x)
        ax.fill_between(x,ylim[0],bound,where=x<=ridge,color='#EBF3F8',zorder=0)
        ax.fill_between(x,ylim[0],bound,where=x>=ridge,color='#EDF4EA',zorder=0)
        ax.plot(x,bandwidth*x,color='#447D98',ls='--',lw=1.4,zorder=1)
        ax.axhline(peak,color='#537650',lw=1.4,zorder=1)
        ax.axvline(ridge,color='#B64D4D',ls=':',lw=1.3,zorder=1)
        ax.text(1500,47,f'{peak:.1f} TFLOP/s',ha='right',color='#537650',fontsize=9)
        ax.text(.3,.37,f'{bandwidth*1000:.0f} GB/s',rotation=29,color='#447D98',fontsize=9)
        ax.text(ridge*1.15,67,f'{ridge:.1f} FLOPs/B',color='#B64D4D',fontsize=8.8)
        ax.text(.042,.020,'Memory ceiling lower',color='#447D98',fontsize=8.8)
        ax.text(72,.020,'Compute ceiling lower',color='#537650',fontsize=8.8)
        for fmt in (Q4,Q2):
            points=[p for p in OPERATOR_POINTS if p['phase']==phase and p['format']==fmt]
            for p in points:
                assert p['operator'] in styles and int(p['samples']) in range(1,6)
                if p['operator']!='FlashAttention':assert int(p['n'])==(512 if phase=='prefill' else 8)
                assert p['gpu']=='0' or (phase=='decode' and fmt==Q4 and p['operator']=='LM head' and p['gpu']=='1')
                color,marker=styles[p['operator']]
                ax.scatter(float(p['fp32_flops_per_byte']),float(p['fp32_flops_per_s'])/1e12,
                           s=90 if marker=='*' else 67,marker=marker,
                           facecolors=color if fmt==Q4 else 'none',edgecolors=color,
                           linewidths=1.3 if fmt==Q2 else .75,zorder=4 if fmt==Q4 else 5)
            for op,position in positions[phase,fmt].items():
                group=[p for p in points if p['operator']==op]
                assert group,(phase,fmt,op)
                color,_=styles[op]
                label=('Q4' if fmt==Q4 else 'Q2')+' '+op
                if op=='LM head':label+=' †'
                if phase=='decode' and op=='FlashAttention':label+='\n(3 configs)'
                for i,p in enumerate(group):
                    ax.annotate(label if i==0 else '',
                        (float(p['fp32_flops_per_byte']),float(p['fp32_flops_per_s'])/1e12),
                        xytext=position,ha='center',va='center',fontsize=8.9,weight='bold',color=color,
                        arrowprops={'arrowstyle':'-','color':color,'lw':.85,
                                    'linestyle':'-' if fmt==Q4 else '--','shrinkA':3,'shrinkB':5},zorder=6)
        ax.set(xscale='log',yscale='log',xlim=xlim,ylim=ylim,
               xlabel='Executed scalar FP32 FLOPs / DRAM byte',ylabel='Executed scalar FP32 TFLOP/s')
        ax.set_title(f'{phase.title()} · Q4_K_M and Q2_K',loc='left',fontsize=12,weight='bold',pad=12,color=INK)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        ax.grid(which='major',lw=.5,alpha=.5,color='#C6C6C6')
        ax.tick_params(which='minor',length=2,color='#AAAAAA')
        return len([p for p in OPERATOR_POINTS if p['phase']==phase])

    legend=[
        Line2D([0],[0],color=INK,marker='o',ls='none',label='Q4_K_M: filled'),
        Line2D([0],[0],color=INK,marker='o',markerfacecolor='none',ls='none',label='Q2_K: hollow'),
        Line2D([0],[0],color='#447D98',ls='--',label='DRAM ceiling'),
        Line2D([0],[0],color='#537650',label='FP32 ceiling'),
        Line2D([0],[0],color='#B64D4D',ls=':',label='Compute / memory boundary'),
    ]
    fig,axes=plt.subplots(2,1,figsize=(10.8,10.0))
    fig.subplots_adjust(left=.10,right=.965,top=.855,bottom=.19,hspace=.46)
    fig.text(.06,.968,'Other kernels · Q4 and Q2 on a common roofline',fontsize=18,weight='bold',color=INK)
    fig.text(.06,.931,'Scalar FP32 view · 70B · 8 clients · 2K input / 512 output',fontsize=10.5,color=MUTED)
    fig.text(.06,.901,'Color / shape identifies the operator; filled markers are Q4 and hollow markers are Q2.',fontsize=10,color=MUTED)
    count=sum(panel(ax,phase) for ax,phase in zip(axes,('prefill','decode')))
    assert count==31
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.53,.071),ncol=3,frameon=False,fontsize=9,columnspacing=1.5)
    fig.text(.06,.053,'GPU0 points; † Q4 LM head is GPU1. FlashAttention retains all three decode launch configurations per format.',fontsize=9,color=MUTED)
    fig.text(.06,.030,'31 operator/configuration groups from 120 launches. Coincident points retain their measured coordinates; labels use leader lines.',fontsize=8.8,color=MUTED)
    fig.text(.06,.009,'Scalar FP32 counts exclude Tensor Core, FP16, integer and special-function work. Missing kernels are listed in the report.',fontsize=8.8,color=MUTED)
    save(fig,'operator-roofline-combined')
    for phase in ('prefill','decode'):
        fig,ax=plt.subplots(figsize=(10.8,6.1))
        fig.subplots_adjust(left=.10,right=.965,top=.77,bottom=.255)
        fig.text(.06,.94,'Other kernels · scalar FP32 roofline',fontsize=18,weight='bold',color=INK)
        fig.text(.06,.875,'70B · 8 clients · 2K input / 512 output · filled Q4 / hollow Q2',fontsize=10.5,color=MUTED)
        panel(ax,phase)
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.53,.07),ncol=3,frameon=False,fontsize=9)
        fig.text(.06,.049,'GPU0; † Q4 decode LM head is GPU1. Each point retains one operator/configuration; no missing coordinates are fabricated.',fontsize=8.8,color=MUTED)
        fig.text(.06,.018,'Scalar FP32 counts exclude Tensor Core, FP16, integer and special-function work. Rated per-GPU ceilings: 38.7 TFLOP/s, 768 GB/s.',fontsize=8.8,color=MUTED)
        save(fig,f'operator-roofline-{phase}')


def instruction_figure():
    fig,axes = plt.subplots(1,2,figsize=(12.4,4.4))
    fig.subplots_adjust(left=.065,right=.98,top=.71,bottom=.25,wspace=.28)
    fig.text(.045,.94,'More multiply-add and conversion instructions',fontsize=18,weight='bold',color=INK)
    fig.text(.045,.875,'70B gate/up · GPU 0 · median of five matched launches · Relative change above each Q2 bar',fontsize=10.5,color=MUTED)
    for ax,phase in zip(axes,('prefill','decode')):
        groups = {fmt:instruction_categories(phase,fmt) for fmt in (Q4,Q2)}
        x = np.arange(2)
        for fmt,offset in ((Q4,-.19),(Q2,.19)):
            vals = [groups[fmt][op]/1e6 for op in ('FFMA','I2FP')]
            ax.bar(x+offset,vals,width=.34,color=PALETTE[fmt],label=fmt,zorder=3)
            for i,v in enumerate(vals):
                ax.annotate(f'{v:,.1f}',(x[i]+offset,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10,color=INK)
                if fmt == Q2:
                    op = ('FFMA','I2FP')[i]
                    ax.annotate(signed(pct(groups[Q4][op],groups[Q2][op])),(x[i]+offset,v),xytext=(0,21),
                                textcoords='offset points',ha='center',fontsize=11,weight='bold',color='#a03b28')
        ax.set_title(f'{phase.title()}: {instruction_share(phase):.1f}% of the extra instructions',loc='left',fontsize=11.5,weight='bold',pad=11,color=INK)
        ax.set_xticks(x,['FFMA\nFloating-point multiply-add','I2FP\nInteger-to-float conversion'],fontsize=10)
        ax.set_ylabel('Executed instructions (millions)',fontsize=10,color=MUTED)
        ax.set_xlim(-.58,1.58)
        ax.set_ylim(0,max(groups[Q2][op]/1e6 for op in ('FFMA','I2FP'))*1.55)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        ax.grid(axis='y',color=GRID,zorder=0)
        ax.tick_params(axis='both',length=0,labelcolor=MUTED)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,fontsize=10)
    fig.text(.045,.025,'Separate phase scales; actual counts. The panel title is the combined FFMA + I2FP share of the net instruction increase.',fontsize=9.5,color=MUTED)
    save(fig,'instruction-overhead-actual-values')


def instruction_horizontal_figure():
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.4))
    fig.subplots_adjust(left=.115, right=.965, top=.83, bottom=.22, hspace=.62)
    fig.text(.045, .95, 'More multiply-add and conversion instructions', fontsize=18, weight='bold', color=INK)
    fig.text(.045, .905, '70B gate/up · GPU 0 · median of five matched launches · Relative change beside each Q2 bar',
             fontsize=10.5, color=MUTED)
    ops = ('FFMA', 'I2FP')
    y = np.arange(len(ops))
    for ax, phase in zip(axes, ('prefill', 'decode')):
        groups = {fmt: instruction_categories(phase, fmt) for fmt in (Q4, Q2)}
        for fmt, offset in ((Q4, -.19), (Q2, .19)):
            vals = [groups[fmt][op] / 1e6 for op in ops]
            ax.barh(y + offset, vals, height=.32, color=PALETTE[fmt], label=fmt, zorder=3)
            for i, v in enumerate(vals):
                ax.annotate(f'{v:,.1f}', (v, y[i] + offset), xytext=(6, 0), textcoords='offset points',
                            ha='left', va='center', fontsize=10, color=INK)
                if fmt == Q2:
                    ax.annotate(signed(pct(groups[Q4][ops[i]], groups[Q2][ops[i]])),
                                (v, y[i] + offset), xytext=(46, 0), textcoords='offset points',
                                ha='left', va='center', fontsize=11, weight='bold', color='#a03b28')
        ax.set_title(f'{phase.title()}: {instruction_share(phase):.1f}% of the extra instructions',
                     loc='left', fontsize=11.5, weight='bold', pad=13, color=INK)
        ax.set_yticks(y, ops, fontsize=11)
        ax.set_xlabel('Executed instructions (millions)', fontsize=10, color=MUTED, labelpad=9)
        ax.set_xlim(0, max(groups[Q2][op] / 1e6 for op in ops) * 1.6)
        ax.set_ylim(1.58, -.58)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(axis='x', color=GRID, zorder=0)
        ax.tick_params(axis='both', length=0, labelcolor=MUTED)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .11), ncol=2, frameon=False, fontsize=10)
    fig.text(.045, .075, 'FFMA: floating-point multiply-add. I2FP: integer-to-float conversion.', fontsize=9.5, color=MUTED)
    fig.text(.045, .035, 'Separate phase scales; actual counts. Titles give the FFMA + I2FP share of the net instruction increase.',
             fontsize=9.5, color=MUTED)
    save(fig, 'instruction-overhead-horizontal')


def md_table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:']*(len(header)-1)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def phase_table(phase, keys):
    rows = []
    for key in keys:
        _, _, label, digits = METRICS[key]
        a, b = (value(phase, key, fmt) for fmt in (Q4, Q2))
        if phase == 'decode' and key == 'time':
            a, b, label, digits = a*1000, b*1000, 'Kernel duration (µs)', 3
        change = signed(pct(a,b))
        if key in ('tensor', 'alu', 'occupancy'):
            change = (change + f' ({b-a:+.2f} pp)').replace('-', '−') if a else '0.00 pp; relative change undefined'
        rows.append([label, f'{a:,.{digits}f}', f'{b:,.{digits}f}', change])
    return md_table(['Metric', Q4, Q2, 'Q2 change vs Q4'], rows)


def make_report():
    pre = phase_table('prefill', ['time','instructions','reads','tensor','alu','ipc'])
    dec = phase_table('decode', ['time','instructions','reads','occupancy','registers','blocks','tensor'])
    roof_rows = []
    for key, label, div, digits in [
        ('arithmetic_intensity_ops_per_byte','Tensor arithmetic intensity (ops/byte)',1,1),
        ('work_ops_per_s','Executed tensor throughput (TOPS)',1e12,3),
        ('work_pct_of_peak','Tensor compute (% of path peak)',1,3),
        ('traffic_bytes_per_s','DRAM read + write bandwidth (GB/s)',1e9,3),
        ('traffic_pct_of_peak','DRAM throughput (% of DRAM peak)',1,3),
    ]:
        a,b=(float(roof(fmt)[key])/div for fmt in (Q4,Q2))
        change=signed(pct(a,b))
        if 'pct' in key: change += f' ({b-a:+.3f} pp)'.replace('-', '−')
        roof_rows.append([label,f'{a:,.{digits}f}',f'{b:,.{digits}f}',change])
    roof_table=md_table(['Metric',Q4,Q2,'Q2 change vs Q4'],roof_rows)
    gpu_rows=[]
    for phase in ['prefill','decode']:
        for gpu in [0,1]:
            gpu_rows.append([phase.title(),str(gpu), f'{value(phase,"time",Q4,gpu)*1000:,.3f}',
                             f'{value(phase,"time",Q2,gpu)*1000:,.3f}',delta(phase,'time',gpu),
                             delta(phase,'instructions',gpu),delta(phase,'reads',gpu)])
    gpu_table=md_table(['Phase','GPU','Q4 time (µs)','Q2 time (µs)','Time change','Instructions','DRAM reads'],gpu_rows)
    layout_table=md_table(['Layout property','Q4_K','Q2_K'],[
        ['Weights per scale/offset group','32','16'],['Groups per 256 weights','8','16']])
    instruction_rows=[]
    for phase in ('prefill','decode'):
        a,b=(instruction_categories(phase,fmt) for fmt in (Q4,Q2))
        for op in ('FFMA','I2FP','Other opcodes','All instructions'):
            instruction_rows.append([f'{phase.title()} / {op}',f'{a[op]/1e6:,.3f}',f'{b[op]/1e6:,.3f}',
                                     f'{(b[op]-a[op])/1e6:+,.3f}',signed(pct(a[op],b[op]))])
    instruction_table=md_table(['Phase / instruction','Q4 (M)','Q2 (M)','Extra (M)','Change'],instruction_rows)
    source_rel='../../results/cuda-context-study/diagnostics/q2-q4-full-sections'
    pinned='https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src'
    text=f'''# Runtime bottlenecks of Q2 and Q4 quantization

*Llama 70B · two RTX A6000 GPUs · eight clients · 2,048 input / 512 output tokens*

**Quantization-related arithmetic and conversion overhead limits Q2's runtime benefit.** GPU0 prefill gate/up reads 30.0% fewer DRAM bytes yet executes 69.6% more instructions and takes 49.2% longer. Decode follows the same pattern (Table 2). Opcode counts and kernel code connect the extra instructions to finer quantization groups and scale/offset processing.

Gate/up comparisons use five-launch medians from the full-section captures. GPU0 is shown unless labeled otherwise; Table 6 checks both GPUs. Q4_K_M and Q2_K are model recipes; the selected gate/up tensors use Q4_K and Q2_K respectively.

*Table 1. Prefill gate/up, GPU0; M = 28,672, N = 512, K = 8,192.*

{pre}

![Prefill IPC, ALU and Tensor pipeline activity, instructions, DRAM reads and duration](figures/prefill-actual-values.png)

*Figure 1. Top: IPC, ALU and Tensor pipeline activity. Bottom: instructions, DRAM reads and duration. Bars use actual values on separate, zero-based axes; Q2 labels show relative percentage changes from Q4.*

**Prefill analysis.** IPC rises 10.7% and ALU activity rises 33.7% (+7.41 percentage points), while Tensor activity falls 26.2% (−9.32 points). Higher activity accompanies 69.6% more instructions and longer execution. Table 4 attributes 74.0% of extra instructions to multiply-add and integer-to-float conversion, supporting arithmetic/conversion overhead. The measurements do not isolate each category's runtime cost; pipeline percentages describe separate resources and must not be summed as a partition of runtime.

<!-- PAGEBREAK -->

**Decode also spends more time despite lower DRAM demand.** Q2's kernel duration rises from 321.664 to 402.464 µs. Its 41.6% reduction in DRAM reads therefore coexists with a 25.1% execution-time increase.

*Table 2. Decode gate/up, GPU0; M = 28,672, N = 8, K = 8,192.*

{dec}

![Decode register pressure, occupancy, work and duration](figures/decode-register-pressure.png)

*Figure 2. Top row: registers per thread, the register-limited maximum blocks per SM, and achieved occupancy. Bottom row: instructions, DRAM reads and duration. Actual values and relative changes are shown on separate axes. The arrangement compares observations without assigning the slowdown to occupancy alone.*

**Decode analysis.** Figure 2A shows register use increasing from 125 to 151 per thread (+20.8%), the register-limited block count falling from eight to six (−25.0%), and achieved occupancy falling from 31.79% to 23.83% (−25.0%; −7.96 percentage points). Figure 2B shows 35.2% more instructions and 25.1% longer duration despite 41.6% fewer DRAM reads. Together, these document greater instruction work and register pressure. Occupancy's independent latency contribution is not isolated. Neither kernel records local-memory spill traffic in these captures.

**The same direction appears on GPU1.** Prefill takes {delta('prefill','time',1).lstrip('+')} longer with 30.1% fewer DRAM reads; decode takes {delta('decode','time',1).lstrip('+')} longer with 41.6% fewer DRAM reads. Instruction-count increases are identical across the two GPUs. The absolute timings are retained in Table 6.

<!-- PAGEBREAK -->

**Why Q2 executes more instructions.** Q2_K uses smaller scale/offset groups, doubling the number of independently scaled groups per 256 weights. The [pinned format definitions]({pinned}/ggml-common.h#L293) establish this layout difference.

*Table 3. Quantization groups in the selected gate/up weight tensors.*

{layout_table}

Each group represents weights as w = s × q + b; its dot product is s × Σ(q × x) + b × Σx. The [Q2 CUDA implementation]({pinned}/ggml-cuda/mmq-vec-dot.cuh#L675) combines more separately scaled partial products and offset corrections than the [Q4 path]({pinned}/ggml-cuda/mmq-vec-dot.cuh#L313). These operations are fused into quantized multiplication and require additional arithmetic and conversions. The measured opcode counts support this mechanism.

*Table 4. GPU0 instruction counts, in millions. FFMA = floating-point multiply-add; I2FP = integer-to-float conversion. Other opcodes include all remaining categories.*

{instruction_table}

![Actual instruction counts and percentage increases](figures/instruction-overhead-actual-values.png)

*Figure 3. Actual opcode counts; percentage changes above Q2 use Q4 as the reference. Each panel title reports the two categories' combined share of the net instruction increase.*

FFMA and I2FP contribute 352.322M of 475.927M extra prefill instructions (**74.0%**) and 29.360M of 46.248M extra decode instructions (**63.5%**). The same counts occur on GPU1. These are shares of additional instructions; the corresponding shares of runtime have not been isolated. The remaining categories reconcile exactly to the overall instruction counter.

<!-- PAGEBREAK -->

**The Tensor Core roofline provides supporting context.** In prefill, Q2 moves to higher arithmetic intensity but lower tensor throughput: approximately 989 → 1,441 INT8 operations per DRAM byte and 106.5 → 80.3 TOPS. Its lower DRAM demand does not become higher tensor arithmetic throughput. The opcode breakdown and kernel source provide the direct evidence for the added arithmetic/conversion work.

*Table 5. Prefill gate/up, GPU0; dense INT8 Tensor Core arithmetic path.*

{roof_table}

![Prefill Tensor Core roofline](figures/prefill-tensor-roofline.png)

*Figure 4. Q4 (blue circle) and Q2 (orange diamond) share one plot. Dashed blue DRAM ceilings, green compute ceilings and dotted red boundaries follow the reference figure's styling while retaining the original INT8 measurements. DRAM traffic includes reads and writes; TOPS measures executed tensor operations.*

**Roofline analysis.** Q2's arithmetic intensity increases by 45.7%, yet tensor throughput decreases by 24.6%. DRAM throughput falls from about 14.8% to 7.6% of peak, which does not support a simple saturated-DRAM-bandwidth explanation for this prefill comparison. The plotted points are on the compute-ceiling side of their tensor-path rooflines, but their distance below that ceiling does not identify a unique limiting resource. The instruction and pipeline counters provide the additional evidence for the interpretation.

The roofline covers the selected prefill kernel's dense INT8 Tensor Core work. Nominal Q2/Q4 storage precision is distinct from that arithmetic path. Scalar arithmetic, integer unpacking/correction, scheduling and other kernels are not fully described by this tensor view. The MMVQ decode kernels have zero measured tensor work, so this Tensor Core plot is not applied to decode or to the whole serving workload.

<!-- PAGEBREAK -->

**Other kernels share a scalar FP32 roofline.** Figure 5 overlays Q4 and Q2 within each phase, including all 31 available operator/configuration groups from 120 saved launches.

![Other kernels with Q4 and Q2 overlaid in each phase](figures/operator-roofline-combined.png)

*Figure 5. Prefill above, decode below; Q4 filled, Q2 hollow. Colors/shapes identify operators. All points are GPU0 except Q4's decode LM head (†, GPU1). FlashAttention retains three decode configurations per format.*

Coordinates use predicated-on scalar FP32 work (FADD + FMUL + 2 × FFMA) per DRAM read/write byte and per second. Tensor Core, FP16, integer and special-function work are excluded. Higher executed FLOP/s can reflect extra arithmetic. These exploratory captures differ from the full-section reruns; FlashAttention includes fused softmax.

**Missing counters:** Q2 V projection and LM head; both prefill LM heads; activation quantization, RMSNorm/RoPE, SiLU, residual additions, KV-cache updates and other helpers. [Coordinates](operator-roofline-data.csv) retain kernel/configuration and sample counts; the [source method]({source_rel}/roofline-operators.md) documents validation and rated per-GPU ceilings.

<!-- PAGEBREAK -->

**Both GPUs corroborate the central finding.** The direction and magnitude of the changes are similar across devices, with approximately 49% longer prefill kernels and 25–26% longer decode kernels for Q2.

*Table 6. Five-launch medians from the same full-section dataset. All change columns compare Q2 against Q4.*

{gpu_table}

**Measurement and reporting conventions.** The saved validation records confirm all eight expected captures: two formats × two phases × two GPUs. Each capture has eight warmup requests and eight profiled requests, and pools three gate plus two up launches with matched geometry and phase. Prefill width N = 512 is a processing chunk within the 2,048-token input; decode width N = 8 counts token vectors in the matrix operation. The server retains its 41/39-layer split and FP16 KV cache. CUDA graphs are disabled for annotations; clocks are automatic and cache control is none.

Percentage change = 100 × (Q2 value / Q4 value − 1). Calculations use unrounded source values, with display rounding applied afterward. For percentage-valued counters, pp denotes the absolute percentage-point difference. A positive duration change is a slowdown; a negative DRAM-read change is a traffic reduction. Relative change is undefined when both values are zero. MB and GB use decimal units; M instructions means one million instructions. Figures use separate scales because the metrics have different physical units.

**Scope of the finding.** These are kernel-level profiler replay measurements, not request latency or end-to-end throughput. The detailed gate/up comparison uses five launches from one capture per case; the broader operator roofline uses 1–5. Neither provides independent run-to-run uncertainty. GPU1 corroborates the gate/up pattern without establishing its universality across hardware or workloads. Opcode counts identify additional instruction categories, but their separate latency costs are not isolated. The other-kernel roofline describes only each kernel's scalar FP32 arithmetic.

**Report takeaway.** Q2_K's finer quantization groups increase partial-product processing, conversions and scale/offset corrections. FFMA and I2FP account for 74.0% of additional prefill instructions and 63.5% in decode. The selected kernels read fewer DRAM bytes but execute more instructions and run longer. The roofline supports the memory/compute interpretation; the instruction trace and kernel implementation identify the mechanism.

**Source data and reproducibility.** This report uses the existing [full-section capture report]({source_rel}/README.md), [hardware metric medians]({source_rel}/all-hardware-metrics.csv), [opcode instances]({source_rel}/instruction-instances.csv), [derived roofline values]({source_rel}/roofline-values.csv), [operator coordinates]({source_rel}/operator-roofline-points.csv) and their saved validation records. Format and kernel references use llama.cpp commit 3f5e94d7c2ab2267fe39852051777fe30c1f49ef. The source report links raw Nsight captures. No new inference or profiling was run.

The adjacent [report-data.json](report-data.json), [comparison-data.csv](comparison-data.csv), [opcode-comparison-data.csv](opcode-comparison-data.csv), [instruction-attribution-data.csv](instruction-attribution-data.csv) and [operator-roofline-data.csv](operator-roofline-data.csv) preserve figure/table inputs. [evidence.json](evidence.json) records source hashes. Rebuild with `python3 paper/runtime-bottlenecks-report/build.py` from the repository root; dependencies are Matplotlib, NumPy, Markdown and ReportLab, with existing document libraries discovered automatically.
'''
    (HERE/'runtime-bottlenecks-report.md').write_text(text)
    return text


def exports():
    selected=[]
    for phase in ['prefill','decode']:
        for gpu in [0,1]:
            for key,(metric,divisor,label,_) in METRICS.items():
                a,b=(value(phase,key,fmt,gpu) for fmt in (Q4,Q2))
                selected.append({'phase':phase,'gpu':gpu,'metric':metric,'label':label,
                                 'Q4_K_M':a,'Q2_K':b,'relative_change_percent':pct(a,b),
                                 'absolute_change':b-a,'source_divisor':divisor,'statistic':'median','launches_per_format':5})
    with (HERE/'comparison-data.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]))
        writer.writeheader();writer.writerows(selected)
    opcodes,attribution=[],[]
    for phase in ('prefill','decode'):
        for gpu in (0,1):
            a,b=(opcode_counts(phase,fmt,gpu) for fmt in (Q4,Q2))
            extra=sum(b.values())-sum(a.values())
            for op in sorted(a.keys()|b.keys()):
                q4,q2=a.get(op,0),b.get(op,0)
                opcodes.append({'phase':phase,'gpu':gpu,'opcode':op,'Q4_instructions':q4,'Q2_instructions':q2,
                                'extra_instructions':q2-q4,'relative_change_percent':pct(q4,q2),
                                'share_of_net_extra_percent':100*(q2-q4)/extra,
                                'metric':'sass__inst_executed_per_opcode','statistic':'median','launches_per_format':5})
            a,b=(instruction_categories(phase,fmt,gpu) for fmt in (Q4,Q2))
            for category in a:
                attribution.append({'phase':phase,'gpu':gpu,'category':category,'Q4_instructions':a[category],
                                    'Q2_instructions':b[category],'extra_instructions':b[category]-a[category],
                                    'relative_change_percent':pct(a[category],b[category]),
                                    'share_of_net_extra_percent':100*(b[category]-a[category])/extra,
                                    'FFMA_plus_I2FP_share_percent':instruction_share(phase,gpu)})
    for name,rows in [('opcode-comparison-data.csv',opcodes),('instruction-attribution-data.csv',attribution)]:
        with (HERE/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (HERE/'operator-roofline-data.csv').write_bytes((SOURCE/'operator-roofline-points.csv').read_bytes())
    (HERE/'operator-roofline-validation.json').write_bytes((SOURCE/'operator-roofline-validation.json').read_bytes())
    (HERE/'report-data.json').write_text(json.dumps({'comparisons':selected,'prefill_gpu0_roofline':[roof(Q4),roof(Q2)],
        'opcode_comparisons':opcodes,'instruction_attribution':attribution,
        'operator_roofline_points':OPERATOR_POINTS,'operator_roofline_validation':OPERATOR_VALIDATION,
        'group_layouts':{'Q4_K':{'weights_per_group':32,'groups_per_256_weights':8},
                         'Q2_K':{'weights_per_group':16,'groups_per_256_weights':16}},
        'instruction_counting':'sass__inst_executed_per_opcode; per-opcode medians sum exactly to sm__inst_executed.sum in all eight cases. An absent opcode within a complete available table contributes zero.'},indent=2)+'\n')
    paths=[SOURCE/name for name in ['all-hardware-metrics.csv','instruction-instances.csv','roofline-values.csv','validation.json','README.md',
                                  'operator-roofline-points.csv','operator-roofline-validation.json','roofline-operators.md']]
    paths += [ROOT/'vendor/llama.cpp/ggml/src'/name for name in ['ggml-common.h','ggml-cuda/mmq-vec-dot.cuh',
                                                             'ggml-cuda/vecdotq.cuh','ggml-cuda/mmq.cuh']]
    (HERE/'evidence.json').write_text(json.dumps({'dataset':'q2-q4-full-sections','aggregation':'five-launch medians',
        'percentage_formula':'100 * (Q2 / Q4 - 1)',
        'instruction_share_formula':'100 * (extra FFMA + extra I2FP) / extra total instructions',
        'implementation_commit':'3f5e94d7c2ab2267fe39852051777fe30c1f49ef',
        'inputs':[{'path':str(p.relative_to(ROOT)),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},indent=2)+'\n')


def document_outputs(md):
    for suffix,weight,style in [('', 'normal','normal'),('Bold','bold','normal'),('Italic','normal','italic'),('BoldItalic','bold','italic')]:
        path=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight=weight,style=style))
        pdfmetrics.registerFont(TTFont('Report'+suffix,path))
    pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBoldItalic')
    body=ParagraphStyle('Body',fontName='Report',fontSize=9.2,leading=13.4,textColor=colors.HexColor(INK),spaceAfter=8)
    title=ParagraphStyle('Title',parent=body,fontName='ReportBold',fontSize=20,leading=24,spaceAfter=10)
    caption=ParagraphStyle('Caption',parent=body,fontSize=7.6,leading=10.4,textColor=colors.HexColor(MUTED),spaceAfter=7)
    cell=ParagraphStyle('Cell',parent=body,fontSize=7.5,leading=10,spaceAfter=0)
    headcell=ParagraphStyle('HeadCell',parent=cell,fontName='ReportBold',textColor=colors.white)
    width=528

    def inline(node):
        s=ET.tostring(node,encoding='unicode')
        s=s[s.index('>')+1:s.rfind('</')]
        return s.replace('<code>','<font name="Report" size="8">').replace('</code>','</font>')

    story=[]
    chunks=md.split('<!-- PAGEBREAK -->')
    for page,chunk in enumerate(chunks):
        if page: story.append(PageBreak())
        root=ET.fromstring('<root>'+markdown.markdown(chunk,extensions=['tables'])+'</root>')
        nodes=list(root)
        i=0
        while i<len(nodes):
            n=nodes[i]
            if n.tag=='h1': story.append(Paragraph(inline(n),title))
            elif n.tag=='p' and n.find('img') is not None:
                img=Image(str(HERE/n.find('img').attrib['src']))
                img.drawHeight*=width/img.drawWidth; img.drawWidth=width
                group=[img]
                if i+1<len(nodes) and nodes[i+1].tag=='p' and inline(nodes[i+1]).startswith('<em>Figure'):
                    i+=1;group.append(Paragraph(inline(nodes[i]),caption))
                story.append(KeepTogether(group))
            elif n.tag=='p':
                content=inline(n)
                style=caption if content.startswith('<em>') else body
                p=Paragraph(content,style)
                if content.startswith('<em>Table'): p.keepWithNext=True
                story.append(p)
            elif n.tag=='table':
                rows=[]
                for rindex,r in enumerate(n.iter('tr')):
                    row=[]
                    for c in r:
                        p=Paragraph(inline(c),headcell if rindex==0 else cell)
                        row.append(p)
                    rows.append(row)
                widths={3:[240,144,144],4:[209,76,76,167],5:[190,80,80,90,88],7:[52,35,84,84,80,96,97]}[len(rows[0])]
                assert sum(widths)==width
                table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('BACKGROUND',(0,0),(-1,0),colors.HexColor(INK)),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4f8'),colors.white]),
                    ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor(GRID)),
                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                    ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ]))
                story.extend([table,Spacer(1,10)])
            else: raise ValueError(n.tag)
            i+=1

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(MUTED));canvas.setFont('Report',7.3)
        if doc.page>1:canvas.drawString(42,764,'Runtime bottlenecks · Q2_K vs Q4_K_M')
        canvas.drawString(42,26,'70B GPU profiling · kernel counters and operator rooflines')
        canvas.drawRightString(570,26,str(doc.page));canvas.restoreState()

    SimpleDocTemplate(str(HERE/'runtime-bottlenecks-report.pdf'),pagesize=(612,792),
        leftMargin=42,rightMargin=42,topMargin=43,bottomMargin=43,
        title='Runtime bottlenecks of Q2 and Q4 quantization',author='').build(story,onFirstPage=footer,onLaterPages=footer)
    html=markdown.markdown(md,extensions=['tables'])
    css='''body{font-family:system-ui,sans-serif;color:#142c43;max-width:1080px;margin:48px auto;padding:0 28px;line-height:1.6}h1{line-height:1.2;font-size:34px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0 26px}th{background:#142c43;color:white;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}td{padding:9px;text-align:right;border-bottom:1px solid #dfe6ec}tr:nth-child(even){background:#f0f4f8}img{max-width:100%;height:auto}a{color:#245ba8}em{color:#526274;font-size:14px}code{font-size:12px}@media print{body{margin:0;max-width:none}table,img{break-inside:avoid}}'''
    (HERE/'runtime-bottlenecks-report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Runtime bottlenecks report</title><style>'+css+'</style><body>'+html+'</body></html>')


def bundle():
    files=[p for p in HERE.rglob('*') if p.is_file() and not p.name.startswith('preview-')
           and p.suffix in ('.py','.pdf','.png','.svg','.md','.html','.json','.csv')]
    with zipfile.ZipFile(HERE/'runtime-bottlenecks-report-bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path,path.relative_to(HERE))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, help='Build a new RTX PRO 6000 report from this study')
    parser.add_argument('--captures', type=Path, help='RTX capture root; default: STUDY/profiles/c8-p2048')
    parser.add_argument('--output', type=Path, help='Separate output directory; default beneath the RTX study')
    args = parser.parse_args()
    if args.study_root:
        sys.path.insert(0, str(ROOT / 'benchmark'))
        from rtxpro6000_publication import build_bottlenecks
        print(build_bottlenecks(args.study_root, args.captures or args.study_root / 'profiles/c8-p2048', args.output))
        sys.exit(0)
    if args.output or args.captures:
        parser.error('--output and --captures require --study-root')
    load_historical_data()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    validation=json.loads((SOURCE/'validation.json').read_text())
    assert validation['complete'] and validation['completed_captures']==8
    bar_figure('prefill');bar_figure('decode');decode_register_figure();instruction_figure();roofline_figure()
    instruction_horizontal_figure()
    operator_roofline_figure()
    exports()
    document_outputs(make_report())
    bundle()
    print(HERE/'runtime-bottlenecks-report.pdf')
