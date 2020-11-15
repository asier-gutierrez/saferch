import os
import argparse
from collections import Counter
from graph import load, connected
import matplotlib.pyplot as plt


def flaw_bar_plot(G, output):
    flaw_counter = Counter([node[1]['flaws'] for node in G.nodes(data=True)])
    max_k = max(flaw_counter.keys())
    max_v = max(flaw_counter.values())
    plot(flaw_counter, output, max_k, max_v)


def plot(flaw_counter, output, max_x, max_y):
    labels = list()
    xs = list()
    for idx in range(max_x + 1):
        v = flaw_counter.get(idx, 0)
        labels.append(idx)
        xs.append(v)

    fig = plt.figure(figsize=(6, 6))
    plt.title("Breaches distribution")
    plt.xlabel("Number of breaches")
    plt.ylabel("Number of nodes")
    plt.ylim(0, max_y+20)
    plt.bar(range(len(xs)), xs, tick_label=labels, color='#8ebad9')
    plt.savefig(os.path.join(output, 'flaws_barplot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot multiple bar plots with the same size of axes.')
    parser.add_argument('paths', type=str, nargs='+', help='Paths to graphs.')
    args = parser.parse_args()

    graphs_paths = list()
    max_k, max_v = 0, 0
    for path in args.paths:
        # Load
        G = load(os.path.join(path, 'graph.net'))

        # Get connected graph
        G = connected(G, strategy="connect")

        # Get flaws counter
        flaw_counter = Counter([node[1]['flaws'] for node in G.nodes(data=True)])

        # Get max k,v
        max_k = max(max_k, max(flaw_counter.keys()))
        max_v = max(max_v, max(flaw_counter.values()))

        # Append to list
        graphs_paths.append([G, flaw_counter, path])

    for G, flaw_counter, path in graphs_paths:
        plot(flaw_counter, path, max_k, max_v)
