"""Generate splice site experiment report: Markdown + PNGs."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/docs/experiment_report_splice"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight', 'figure.facecolor': 'white'})

BASE = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints/splice_v2"
METRICS = ['macro_f1', 'mcc', 'auroc_macro', 'auprc_macro']
METRIC_LABELS = {'macro_f1': 'F1 (Macro)', 'mcc': 'MCC', 'auroc_macro': 'AUROC (Macro)', 'auprc_macro': 'AUPRC (Macro)'}
CLASS_NAMES = ['Donor', 'Acceptor', 'Non-Site']
PER_CLASS_METRICS = ['F1', 'Recall', 'AUROC', 'AUPRC']
PER_CLASS_LABELS = {'F1': 'F1', 'Recall': 'Recall', 'AUROC': 'AUROC', 'AUPRC': 'AUPRC'}
COLORS = {'Ours full-ft': '#1a73e8', 'Ours frozen': '#669df6',
          'GENA full-ft': '#ea4335', 'GENA frozen': '#f28b82',
          'CNN': '#34a853', 'Mamba2_1layer': '#f9ab00',
          'Ours': '#1a73e8', 'GENA': '#ea4335'}

def load_cv(path):
    with open(path) as f: return json.load(f)['metrics']

# ── Experiment 1: 5-fold CV, 6 models ──
exp1 = {
    'Ours full-ft':      load_cv(f"{BASE}/fullft_lr2e-5/cv_summary.json"),
    'Ours frozen':       load_cv(f"{BASE}/frozen/cv_summary.json"),
    'GENA full-ft':      load_cv(f"{BASE}/baseline_gena/cv_summary.json"),
    'GENA frozen':       load_cv(f"{BASE}/baseline_gena_frozen/cv_summary.json"),
    'CNN':               load_cv(f"{BASE}/cnn/cv_summary.json"),
    'Mamba2_1layer':     load_cv(f"{BASE}/baseline_mamba2_1layer/cv_summary.json"),
}

# ── Experiment 2: few-shot (aggregate per-fold, compute macro AUROC from per-class) ──
def aggregate_fewshot(model_dir, pct):
    aurocs, auprcs, f1s, mccs = [], [], [], []
    for fold in range(5):
        m = json.load(open(f"{BASE}/fewshot/{model_dir}_{pct}pct/fold_{fold}/metrics.json"))
        # macro AUROC = mean of 3 per-class AUROC
        auroc = (m['Donor_AUROC'] + m['Acceptor_AUROC'] + m['Non-Site_AUROC']) / 3
        aurocs.append(auroc)
        auprcs.append(m['auprc_macro'])
        f1s.append(m['macro_f1'])
        mccs.append(m['mcc'])
    return {
        'auroc_macro': {'mean': np.mean(aurocs), 'std': np.std(aurocs)},
        'auprc_macro': {'mean': np.mean(auprcs), 'std': np.std(auprcs)},
        'macro_f1': {'mean': np.mean(f1s), 'std': np.std(f1s)},
        'mcc': {'mean': np.mean(mccs), 'std': np.std(mccs)},
    }

fewshot = {}
for pct in [10, 20, 30, 40]:
    fewshot[f'Ours {pct}%'] = aggregate_fewshot('ours_fullft', pct)
    fewshot[f'GENA {pct}%'] = aggregate_fewshot('gena_fullft', pct)
# 100% from fullft and baseline_gena
fewshot['Ours 100%'] = exp1['Ours full-ft']
fewshot['GENA 100%'] = exp1['GENA full-ft']

# ── Charts ──
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


def per_class_chart(exp_data, class_name, filename, sub, sub_label):
    """单个类别的单个指标柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    key = f'{class_name}_{sub}'
    labels = list(exp_data.keys())
    valid_labels = [l for l in labels if key in exp_data[l]]
    means = [exp_data[l][key]['mean'] for l in valid_labels]
    stds  = [exp_data[l][key].get('std', 0) for l in valid_labels]
    colors = [COLORS.get(l, '#666666') for l in valid_labels]

    bars = ax.bar(range(len(valid_labels)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(valid_labels)))
    ax.set_xticklabels(valid_labels, rotation=25, ha='right', fontsize=9)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel(sub_label, fontsize=10)
    ax.set_title(f'{class_name} — {sub_label}', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches='tight')
    plt.close()

# ── Markdown ──
md = []
def w(s=''): md.append(s + '\n')

w('# 剪接位点预测实验总结报告')
w()
w('> **评估指标**: F1 (Macro), MCC, AUROC (Macro), AUPRC (Macro)  —  五折交叉验证')

# Exp 1
w('## 实验 1：常规五折交叉验证（三分类）')
w()
w('| 模型 | F1 (Macro) | MCC | AUROC (Macro) | AUPRC (Macro) |')
w('|------|-----------|-----|---------------|---------------|')
for name, m in exp1.items():
    w(f"| {name} | {m['macro_f1']['mean']:.4f} ±{m['macro_f1']['std']:.4f} | {m['mcc']['mean']:.4f} ±{m['mcc']['std']:.4f} | {m['auroc_macro']['mean']:.4f} ±{m['auroc_macro']['std']:.4f} | {m['auprc_macro']['mean']:.4f} ±{m['auprc_macro']['std']:.4f} |")
w()
w('> **注**: (1) Mamba2_1layer 的 fold 3 最初在 LR=1e-3 下塌缩（F1=0.52），降 LR 至 5e-4 重训后恢复 0.89；(2) Ours full-ft 原 LR=5e-6 过于保守（best_epoch≈26），经 LR 扫描确认为 2e-5 最优，重训后 F1 从 0.886 提升至 0.897。')
w()
for metric in METRICS:
    fn = f'exp1_{metric}.pdf'
    single_chart(exp1, f'Experiment 1: Splice Site 5-Fold CV — {METRIC_LABELS[metric]}',
                 fn, metric, METRIC_LABELS[metric])
    w(f'![Experiment 1 {METRIC_LABELS[metric]}]({fn})')
w()

# Per-class 分析图（Donor / Acceptor / Non-Site × 4 指标）
w('### 各类别分析')
w()
for cls in CLASS_NAMES:
    for sub in PER_CLASS_METRICS:
        fn = f'exp1_perclass_{cls}_{sub}.pdf'
        per_class_chart(exp1, cls, fn, sub, PER_CLASS_LABELS[sub])
        w(f'![{cls} {PER_CLASS_LABELS[sub]}]({fn})')
    w()

# Exp 2
w('## 实验 2：Few-Shot 梯度抽样（五折交叉验证）')
w()
w('| 模型 | 比例 | F1 (Macro) | MCC | AUROC (Macro) | AUPRC (Macro) |')
w('|------|------|-----------|-----|---------------|---------------|')
for name, m in fewshot.items():
    w(f"| {name.replace(' Ours',' Ours').replace(' GENA',' GENA')} | {'—' if '%' in name else ''} | {m['macro_f1']['mean']:.4f} ±{m['macro_f1']['std']:.4f} | {m['mcc']['mean']:.4f} ±{m['mcc']['std']:.4f} | {m['auroc_macro']['mean']:.4f} ±{m['auroc_macro']['std']:.4f} | {m['auprc_macro']['mean']:.4f} ±{m['auprc_macro']['std']:.4f} |")
w()
for metric in METRICS:
    fn = f'exp2_{metric}.pdf'
    single_chart(fewshot, f'Experiment 2: Few-Shot Scaling — {METRIC_LABELS[metric]}',
                 fn, metric, METRIC_LABELS[metric], chart_type='line')
    w(f'![Experiment 2 {METRIC_LABELS[metric]}]({fn})')
w()

with open(f"{OUTPUT_DIR}/report.md", 'w') as f:
    f.writelines(md)
print(f"Markdown: {OUTPUT_DIR}/report.md")
print(f"Charts: {OUTPUT_DIR}/")
print(f"Generated {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pdf')])} PNGs")
