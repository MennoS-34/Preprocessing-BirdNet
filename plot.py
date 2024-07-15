import matplotlib.pyplot as plt

# Data from the log
data = {
    5: {'ARI': 0.022073278546084495, 'NMI': 0.034194524164800365},
    6: {'ARI': 0.01751071575935065, 'NMI': 0.028590690310875615},
    7: {'ARI': 0.017527211453624167, 'NMI': 0.029903113234382227},
    8: {'ARI': 0.0062744544288334935, 'NMI': 0.03467381877390489},
    9: {'ARI': 0.005657587008751611, 'NMI': 0.03055552984537518},
    10: {'ARI': 0.010699671120120949, 'NMI': 0.027124584644935168},
    11: {'ARI': 0.004647723030630703, 'NMI': 0.027200454245673418},
    12: {'ARI': 0.0027575522370018426, 'NMI': 0.027364096451008398},
    14: {'ARI': 0.004913032519467589, 'NMI': 0.027817131799456427},
    16: {'ARI': 0.005462432208248151, 'NMI': 0.026320805619952634},
    18: {'ARI': 0.0017135859372402384, 'NMI': 0.027897006168741963},
    20: {'ARI': 0.0036145370887466287, 'NMI': 0.03340867295603104}
}

# Extracting the values for plotting
k_values = list(data.keys())
ari_values = [data[k]['ARI'] for k in k_values]
nmi_values = [data[k]['NMI'] for k in k_values]

# Setting font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# Creating the plot with bright colors
plt.figure(figsize=(12, 8))
plt.plot(k_values, nmi_values, 'o-', color='orange', label='NMI', markersize=8, linewidth=2)
plt.plot(k_values, ari_values, 's-', color='dodgerblue', label='ARI', markersize=8, linewidth=2)
plt.xlabel('Number of clusters (k)', fontsize=18)
plt.ylabel('Score', fontsize=18)
plt.title('ARI and NMI Scores for Different Numbers of Clusters', fontsize=20, fontweight='bold')
plt.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('formal_plot_bright.png', dpi=300)
plt.show()

