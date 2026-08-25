"""Generate comprehensive experiment report: Markdown + PNGs + Jupyter Notebook."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/docs/experiment_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight', 'figure.facecolor': 'white'})

METRICS = ['auroc', 'auprc', 'f1_macro', 'mcc']
METRIC_LABELS = {'auroc': 'AUROC', 'auprc': 'AUPRC', 'f1_macro': 'F1 (Macro)', 'mcc': 'MCC'}
COLORS = {'Ours': '#1a73e8', 'GENA': '#ea4335', 'CNN': '#34a853', 'Mamba2': '#f9ab00',
          'Ours Frozen': '#669df6', 'Ours Finetune': '#1a73e8',
          'GENA Frozen': '#f28b82', 'GENA Finetune': '#ea4335'}

# ═══════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════

def load_cv(path):
    with open(path) as f: return json.load(f)['metrics']

BASE = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints"
exp1 = {
    'Ours Frozen':   load_cv(f"{BASE}/cds_intergenic/ours_frozen/cv_summary.json"),
    'Ours Finetune': load_cv(f"{BASE}/cds_intergenic/ours_finetune/cv_summary.json"),
    'GENA Frozen':   load_cv(f"{BASE}/cds_intergenic/gena_frozen/cv_summary.json"),
    'GENA Finetune': load_cv(f"{BASE}/cds_intergenic/gena_finetune/cv_summary.json"),
    'CNN':           load_cv(f"{BASE}/cds_intergenic/cnn_train/cv_summary.json"),
    'Mamba2':        load_cv(f"{BASE}/cds_intergenic/mamba2_train/cv_summary.json"),
}

# Few-shot: from 5-fold aggregated data
fewshot_base = f"{BASE}/cds_intergenic_fewshot"
fewshot = {}
for model, label in [('ours','Ours'), ('gena','GENA')]:
    for pct in ['10pct','20pct','30pct','40pct']:
        aurocs, auprcs, f1s, mccs = [], [], [], []
        for fold in range(5):
            d = f"{fewshot_base}/{model}_{pct}" if fold == 0 else f"{fewshot_base}/{model}_{pct}_fold{fold}"
            m = json.load(open(f"{d}/metrics.json"))
            aurocs.append(m['auroc']); auprcs.append(m['auprc'])
            f1s.append(m['f1_macro']); mccs.append(m['mcc'])
        fewshot[f"{label} {pct.replace('pct','%')}"] = {
            'auroc': {'mean': np.mean(aurocs), 'std': np.std(aurocs)},
            'auprc': {'mean': np.mean(auprcs), 'std': np.std(auprcs)},
            'f1_macro': {'mean': np.mean(f1s), 'std': np.std(f1s)},
            'mcc': {'mean': np.mean(mccs), 'std': np.std(mccs)},
        }

# 100% 用完整数据的 5 折结果
fewshot['Ours 100%'] = exp1['Ours Finetune']
fewshot['GENA 100%'] = exp1['GENA Finetune']

# Hard CDS (5-fold CV)
hard_base = f"{BASE}/cds_intergenic_hard"
exp3 = {}
for name in ['ours_frozen','ours_finetune','gena_frozen','gena_finetune']:
    m = json.load(open(f"{hard_base}/{name}/cv_summary.json"))['metrics']
    label = name.replace('_',' ').title()
    exp3[label] = m

# sORF (5-fold CV)
sorf_base = f"{BASE}/cds_intergenic_sorf"
exp4 = {}
for name in ['ours_frozen','ours_finetune','gena_frozen','gena_finetune']:
    m = json.load(open(f"{sorf_base}/{name}/cv_summary.json"))['metrics']
    label = name.replace('_',' ').title()
    exp4[label] = m

# ═══════════════════════════════════════════════════
# CHARTS — One per experiment (2x2 subplots for 4 metrics)
# ═══════════════════════════════════════════════════

def single_chart(data_dict, title, filename, metric, metric_label, chart_type='bar'):
    """单个指标的一张图（柱状图或折线图，few-shot 折线含 100% 断裂标记）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(data_dict.keys())
    means = [data_dict[l][metric]['mean'] for l in labels]
    stds  = [data_dict[l][metric].get('std', 0) for l in labels]
    colors = [COLORS.get(l, '#666666') for l in labels]

    if chart_type == 'bar':
        bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=3,
                      color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    else:  # line: few-shot，40%→100% 等距，线段中点用双竖杠隔断（竖杠间无线段）
        x_map = {10: 10, 20: 20, 30: 30, 40: 40, 100: 50}
        gap_half = 0.3
        series_data = {}
        for series in ['Ours', 'GENA']:
            triples = []
            for k, v in data_dict.items():
                if series in k:
                    pct = int(k.split()[-1].replace('%', ''))
                    triples.append((pct, v[metric]['mean'], v[metric].get('std', 0)))
            triples.sort(key=lambda t: t[0])
            series_data[series] = triples
        for series, color, marker in [('Ours', COLORS['Ours'], 'o'),
                                      ('GENA', COLORS['GENA'], 's')]:
            triples = series_data[series]
            pcts = [t[0] for t in triples]
            xs = [x_map[t[0]] for t in triples]
            ys = [t[1] for t in triples]
            es = [t[2] for t in triples]
            # 线段（无 marker，断点只是线段端点）
            if 40 in pcts and 100 in pcts:
                i40 = pcts.index(40)
                i100 = pcts.index(100)
                x40, y40 = xs[i40], ys[i40]
                x100, y100 = xs[i100], ys[i100]
                x_mid = (x40 + x100) / 2
                x_b1 = x_mid - gap_half
                x_b2 = x_mid + gap_half
                slope = (y100 - y40) / (x100 - x40)
                y_b1 = y40 + slope * (x_b1 - x40)
                y_b2 = y40 + slope * (x_b2 - x40)
                ax.plot(xs[:i40+1] + [x_b1], ys[:i40+1] + [y_b1], color=color, lw=2)
                ax.plot([x_b2] + xs[i100:], [y_b2] + ys[i100:], color=color, lw=2)
            else:
                ax.plot(xs, ys, color=color, lw=2)
            # 数据点 marker（只在真实数据点）+ 误差线
            ax.errorbar(xs, ys, yerr=es, fmt='none', ecolor=color, capsize=3, elinewidth=1.5)
            ax.scatter(xs, ys, marker=marker, s=49, color=color, zorder=3, label=series)
        # 双竖杠（隔断标记，较短）
        y_lim = ax.get_ylim()
        bar_len = (y_lim[1] - y_lim[0]) * 0.03
        for series, color in [('Ours', COLORS['Ours']), ('GENA', COLORS['GENA'])]:
            triples = series_data[series]
            pcts = [t[0] for t in triples]
            if 40 in pcts and 100 in pcts:
                i40 = pcts.index(40)
                i100 = pcts.index(100)
                x_mid = (x_map[40] + x_map[100]) / 2
                y_mid = (triples[i40][1] + triples[i100][1]) / 2
                ax.plot([x_mid - gap_half, x_mid - gap_half],
                        [y_mid - bar_len, y_mid + bar_len], color=color, lw=2)
                ax.plot([x_mid + gap_half, x_mid + gap_half],
                        [y_mid - bar_len, y_mid + bar_len], color=color, lw=2)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xticklabels(['10%', '20%', '30%', '40%', '100%'])
        ax.set_xlabel('Training Data (%)', fontsize=10)
        ax.legend(fontsize=9)

    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════
# GENERATE
# ═══════════════════════════════════════════════════

md = []
def w(s=''): md.append(s + '\n')

w('# CDS/Intergenic 二分类实验总结报告')
w()

# ── Experiment 1 ──
w('## 实验 1：常规五折交叉验证')
w()
w('| 模型 | AUROC | AUPRC | F1 (Macro) | MCC |')
w('|------|-------|-------|-----------|-----|')
for name, m in exp1.items():
    w(f"| {name} | {m['auroc']['mean']:.4f} ±{m['auroc']['std']:.4f} | {m['auprc']['mean']:.4f} ±{m['auprc']['std']:.4f} | {m['f1_macro']['mean']:.4f} ±{m['f1_macro']['std']:.4f} | {m['mcc']['mean']:.4f} ±{m['mcc']['std']:.4f} |")
w()
for metric in METRICS:
    fn = f'exp1_{metric}.pdf'
    single_chart(exp1, f'Experiment 1: CDS/Intergenic 5-Fold CV — {METRIC_LABELS[metric]}',
                 fn, metric, METRIC_LABELS[metric])
    w(f'![Experiment 1 {METRIC_LABELS[metric]}]({fn})')
w()

# ── Experiment 2 ──
w('## 实验 2：Few-Shot 梯度抽样（五折交叉验证）')
w()
w('| 模型 | 比例 | AUROC | AUPRC | F1 (Macro) | MCC |')
w('|------|------|-------|-------|-----------|-----|')
for name, m in fewshot.items():
    w(f"| {name} | — | {m['auroc']['mean']:.4f} ±{m['auroc']['std']:.4f} | {m['auprc']['mean']:.4f} ±{m['auprc']['std']:.4f} | {m['f1_macro']['mean']:.4f} ±{m['f1_macro']['std']:.4f} | {m['mcc']['mean']:.4f} ±{m['mcc']['std']:.4f} |")
w()
for metric in METRICS:
    fn = f'exp2_{metric}.pdf'
    single_chart(fewshot, f'Experiment 2: Few-Shot Scaling — {METRIC_LABELS[metric]}',
                 fn, metric, METRIC_LABELS[metric], chart_type='line')
    w(f'![Experiment 2 {METRIC_LABELS[metric]}]({fn})')
w()

# ── Experiment 3 ──
w('## 实验 3：Hard CDS/Intergenic（短序列 ≤300bp）')
w()
w('| 模型 | AUROC | AUPRC | F1 (Macro) | MCC |')
w('|------|-------|-------|-----------|-----|')
for name, m in exp3.items():
    w(f"| {name} | {m['auroc']['mean']:.4f} | {m['auprc']['mean']:.4f} | {m['f1_macro']['mean']:.4f} | {m['mcc']['mean']:.4f} |")
w()
for metric in METRICS:
    fn = f'exp3_{metric}.pdf'
    single_chart(exp3, f'Experiment 3: Hard CDS/Intergenic — {METRIC_LABELS[metric]}',
                 fn, metric, METRIC_LABELS[metric])
    w(f'![Experiment 3 {METRIC_LABELS[metric]}]({fn})')
w()

# ── Experiment 4 ──
w('## 实验 4：人类 sORF 跨物种迁移')
w()
w('> **注意**：Ours Finetune 的 MCC 标准差极大（±0.26），5 折中有 3 折模型塌缩到预测全为同一类别（MCC=0）。真菌 BPE tokenizer 无法编码人类 mRNA，训练极不稳定。GENA 的字符级 tokenizer 相对稳健。')
w()
w('| 模型 | AUROC | AUPRC | F1 (Macro) | MCC |')
w('|------|-------|-------|-----------|-----|')
for name, m in exp4.items():
    w(f"| {name} | {m['auroc']['mean']:.4f} | {m['auprc']['mean']:.4f} | {m['f1_macro']['mean']:.4f} | {m['mcc']['mean']:.4f} |")
w()
for metric in METRICS:
    fn = f'exp4_{metric}.pdf'
    single_chart(exp4, f'Experiment 4: Human sORF Transfer — {METRIC_LABELS[metric]}',
                 fn, metric, METRIC_LABELS[metric])
    w(f'![Experiment 4 {METRIC_LABELS[metric]}]({fn})')
w()

# ═══════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════
with open(f"{OUTPUT_DIR}/report.md", 'w') as f:
    f.writelines(md)
print(f"Markdown report: {OUTPUT_DIR}/report.md")
print(f"Charts: {OUTPUT_DIR}/")
print(f"Generated {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pdf')])} PNG charts")
