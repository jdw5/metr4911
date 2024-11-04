import matplotlib.pyplot as plt
import numpy as np

labels = ['8-core CPU', 'Complete Folding', 'Partial Folding', 'P100', 'A100', 'No Folding']
execution_times = [229782, 19600, 4900, 1020.45, 608.73, 10]

sorted_indices = np.argsort(execution_times)[::-1]
labels = [labels[i] for i in sorted_indices]
execution_times = [execution_times[i] for i in sorted_indices]

# UQ Purple color
uq_purple = '#51247A'
uq_purple_light = '#8D6AB8'

plt.figure(figsize=(14, 8))
bars = plt.bar(labels, execution_times, color=uq_purple, edgecolor=uq_purple_light, linewidth=1.5)

plt.xlabel('Method', fontsize=14, labelpad=10)
plt.ylabel('Execution Time (ns)', fontsize=14, labelpad=10)
plt.xticks(rotation=15, ha='right', fontsize=10)
plt.yscale('log')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             rotation=0, color='black', backgroundcolor='white',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3))

plt.tight_layout()
plt.style.use('seaborn-whitegrid')
plt.gcf().set_facecolor('#f0f0f0')

plt.ylim(top=plt.ylim()[1] * 1.2)

plt.savefig('chart_ex_time.png', dpi=300, bbox_inches='tight')

plt.show()
