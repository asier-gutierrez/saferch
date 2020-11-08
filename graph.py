import os
import json
import networkx as nx
import itertools
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
from ndlib.utils import multi_runs
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from collections import Counter

CMAP = cm.Set3
CMAP_HEAT = cm.jet_r


def relations_graph_attach(G, relations):
    for relation in relations:
        relation = set(relation)
        if len(relation) > 1:
            for e1, e2 in itertools.combinations(relation, 2):
                if G.has_edge(e1, e2):
                    G[e1][e2]['weight'] = G[e1][e2]['weight'] + 1
                else:
                    G.add_edge(e1, e2, weight=1)


def save(G, path):
    nx.write_gml(G, path)


def load(path):
    return nx.read_gml(path)


def connected(G, strategy):
    connected_components = nx.connected_components(G)
    if strategy == 'population':
        nodes = max(connected_components, key=len)
        G = G.subgraph(nodes)
    elif strategy == 'connect':
        connected_components = list(connected_components)
        connect = list()
        for component in connected_components:
            component = list(component)
            nodes_n_edges = [G.number_of_edges(node) for node in component]
            max_node_idx = np.argmax(nodes_n_edges)
            node = component[max_node_idx]
            connect.append(node)
        for e1, e2 in itertools.combinations(connect, 2):
            G.add_edge(e1, e2, weight=1)
    else:
        NotImplementedError('Strategy for obtaining connected graph not implemented.')
    return G


def analysis(G, output):
    data = dict()
    # data['clustering'] = nx.clustering(G, weight='weight')
    # data['pagerank'] = nx.pagerank(G, weight='weight')
    data['average_clustering'] = nx.average_clustering(G, weight='weight')
    data['shortest_path_average'] = nx.average_shortest_path_length(G, weight='weight')
    data['degree_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')

    with open(os.path.join(output, "analysis.json"), 'w') as out:
        json.dump(data, out)

    T = nx.maximum_spanning_tree(G, weight='weight')
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(T, node_color='#8ebad9')
    plt.savefig(os.path.join(output, "mst.png"))


def draw(G, output):
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(G, node_color='#8ebad9')
    plt.savefig(os.path.join(output, 'draw.png'))


def community_analysis(G, output):
    gn_communities = next(nx.algorithms.community.centrality.girvan_newman(G))

    nodes = np.array(G.nodes())
    colors = list(np.zeros(len(nodes)))

    for community_id, communities in enumerate(gn_communities):
        for idx in np.where(np.isin(nodes, list(communities)))[0]:
            colors[idx] = CMAP(community_id)

    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(G, node_color=colors)
    plt.savefig(os.path.join(output, 'gn_community.png'))


def simulate_spread(G, steps, threshold, infected_probability, simulation_beta, simulation_gamma, simulation_alpha,
                    output):
    infected_nodes = []
    exposed_nodes = [node[0] for node in G.nodes(data=True) if node[1]['flaws'] > threshold]

    # Take 1 at random and the rest with probability
    idx = np.random.choice(list(range(len(exposed_nodes))), size=1, replace=False)[0]
    infected_nodes.append(exposed_nodes.pop(idx))

    infected_nodes_count = int(len(exposed_nodes) * infected_probability)
    if infected_nodes_count != 0:
        infected_idxs = np.random.choice(list(range(len(exposed_nodes))), infected_nodes_count)
        infected_nodes.extend(list(np.take(exposed_nodes, infected_idxs)))
        exposed_nodes = list(np.delete(exposed_nodes, infected_idxs))

    susceptible_nodes = [node for node_idx, node in enumerate(G.nodes()) if
                         node_idx not in [*exposed_nodes, *infected_nodes]]

    # Create model
    model = ep.SEIRModel(G)

    # Create configuration
    cfg = mc.Configuration()

    # Infection probability.
    cfg.add_model_parameter('beta', simulation_beta)
    # Infected -> Remove probability
    cfg.add_model_parameter('gamma', simulation_gamma)
    # Latent period
    cfg.add_model_parameter('alpha', simulation_alpha)

    # Set both infected and exposed nodes.

    # cfg.add_model_parameter('fraction_infected', 0.01)
    cfg.add_model_initial_configuration("Susceptible", susceptible_nodes)
    cfg.add_model_initial_configuration("Exposed", exposed_nodes)
    cfg.add_model_initial_configuration("Infected", infected_nodes)

    # Set the configuration
    model.set_initial_status(cfg)

    # Run multiple models
    trends = multi_runs(model, execution_number=10, iteration_number=steps, infection_sets=None, nprocesses=5)

    dt = DiffusionTrend(model, trends)
    dt.plot(filename=os.path.join(output, 'diffusion_trend.png'))

    dp = DiffusionPrevalence(model, trends)
    dp.plot(filename=os.path.join(output, 'diffusion_prevalence.png'))

    # Other run to show
    infected_nodes = list()
    for _ in range(10):
        model.reset()
        iterations = model.iteration_bunch(steps)
        for iteration in iterations:
            for k, v in iteration['status'].items():
                if v == 3:
                    infected_nodes.append(k)

    return Counter(infected_nodes)


def draw_infected(G, infected_nodes, output):
    max_score = max(infected_nodes.values())
    infected_nodes = {k: v / max_score for k, v in infected_nodes.items()}
    node_colors = [CMAP_HEAT(infected_nodes.get(node, 0)) for node in G.nodes()]
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(G, node_color=node_colors)
    plt.colorbar(cm.ScalarMappable(norm=None, cmap=CMAP_HEAT), cax=fig.add_axes([0.93, 0.60, 0.04, 0.38]))
    plt.savefig(os.path.join(output, 'infected_draw.png'))
