import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv("saliency_results/list_all_result_values.csv", index_col=0)
columns = data.columns
ticks = range(1, 4)

# plot
#define list of subplot titles
title_list = ['TP fixation(F)', 'TP heatmap(H)', 'TPTN fixation(F)', 'TPTN heatmap(H)', 'AUC fixation(F)', 'AUC heatmap(H)', 'IG fixation(F)', 'CC heatmap(H)']
columns_first = ['TPfixation', 'TPheatmap', 'TPTNfixation', 'TPTNheatmap', 'AUCfixation', 'AUCheatmap', 'IGfixation', 'CCheatmap']

subtitle_list = ['Shuffled', 'RA', 'Normal']
meanlineprops = dict(linestyle='--', linewidth=2.5, color='red')

fig = plt.figure()
ax = []
bplots = []
for p in range(0,8):
    d = data[[columns_first[p] +" " + subtitle_list[0], columns_first[p] +" " + subtitle_list[1], columns_first[p]]]
    confidence1 = stats.norm.interval(confidence=0.95, loc=data[[columns_first[p]+" " + subtitle_list[0]]].mean(),scale=stats.sem(data[[columns_first[p]+" " + subtitle_list[0]]]))
    confidence2 = stats.norm.interval(confidence=0.95, loc=data[[columns_first[p]+" " + subtitle_list[1]]].mean(),scale=stats.sem(data[[columns_first[p]+" " + subtitle_list[1]]]))
    confidence3 = stats.norm.interval(confidence=0.95, loc=data[[columns_first[p]]].mean(),scale=stats.sem(data[[columns_first[p]]]))

    ax.append(fig.add_subplot(4,2,p+1))
    bplots.append(ax[p].boxplot(d, vert=0,notch = True,usermedians=d.mean(), medianprops = meanlineprops,showcaps=False,
                  patch_artist=True,conf_intervals = (confidence1,confidence2,confidence3),showfliers=False,whis=(100,0)))
    ax[p].set_yticklabels(subtitle_list) 
    ax[p].set_title(title_list[p])

colors = ["beige","yellow","lightgreen",]
for bplot in bplots: # for each boxplot
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adjust output
plt.tight_layout()


plt.show()