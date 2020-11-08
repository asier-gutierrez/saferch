import os
from collections import Counter
import matplotlib.pyplot as plt


def flaw_bar_plot(G, output):
    flaw_counter = Counter([node[1]['flaws'] for node in G.nodes(data=True)])
    labels = list()
    xs = list()
    for k, v in flaw_counter.items():
        labels.append(k)
        xs.append(v)

    fig = plt.figure(figsize=(6, 6))
    plt.title("Flaws distribution")
    plt.xlabel("Flaws")
    plt.ylabel("Number of nodes")
    plt.bar(range(len(xs)), xs, tick_label=labels, color='#8ebad9')
    plt.savefig(os.path.join(output, 'flaws_barplot.png'))
