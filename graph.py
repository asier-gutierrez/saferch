import os, json
import networkx as nx
import itertools
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence
from bokeh.io import export_png
import matplotlib.pyplot as plt
from selenium import webdriver
import numpy as np


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
    data['clustering'] = nx.clustering(G, weight='weight')
    # data['degree_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')
    if nx.is_connected(G):
        data['shortest_path_average'] = nx.average_shortest_path_length(G, weight='weight')
    data['pagerank'] = nx.pagerank(G, weight='weight')

    with open(os.path.join(output, "analysis.json"), 'w') as out:
        json.dump(data, out)

    T = nx.maximum_spanning_tree(G, weight='weight')
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(T)
    plt.savefig(os.path.join(output, "mst.png"))


def draw(G, output):
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(G)
    plt.savefig(os.path.join(output, 'draw.png'))


def community_analysis(G, output):
    nx.algorithms.community.centrality.girvan_newman(G)
    nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    # TODO draw community analysis


def simulate_spread(G, steps, threshold, output):
    exposed_nodes = [node_idx for node_idx, node in enumerate(G.nodes(data=True)) if node[1]['flaws'] > threshold]
    model = ep.SEIRModel(G)

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', 0.01)
    cfg.add_model_parameter('gamma', 0.005)
    cfg.add_model_parameter('alpha', 0.05)
    cfg.add_model_parameter("fraction_infected", 0.05)
    cfg.add_model_initial_configuration("Exposed", exposed_nodes)
    model.set_initial_status(cfg)

    iterations = model.iteration_bunch(steps)
    trends = model.build_trends(iterations)

    dt = DiffusionTrend(model, trends)
    p = dt.plot(width=800, height=800)
    export_png(p, filename=os.path.join(output, 'diffusion_trend.png'),
               webdriver=webdriver.Firefox(executable_path=os.environ['GECKO_DRIVER']))

    dp = DiffusionPrevalence(model, trends)
    p2 = dp.plot(width=800, height=800)
    export_png(p2, filename=os.path.join(output, 'diffusion_prevalence.png'),
               webdriver=webdriver.Firefox(executable_path=os.environ['GECKO_DRIVER']))

    infected_nodes = set()
    for iteration in iterations:
        for k, v in iteration['status'].items():
            if v == 3:
                infected_nodes.add(k)

    return list(infected_nodes)


def draw_infected(G, infected_nodes, output):
    node_colors = ['blue' if node not in infected_nodes else 'red' for node in G.nodes()]
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(G, node_color=node_colors)
    plt.savefig(os.path.join(output, 'infected_draw.png'))
