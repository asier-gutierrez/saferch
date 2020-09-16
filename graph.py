import os, json
import networkx as nx
import itertools
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence
import matplotlib.pyplot as plt


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
    nx.write_pajek(G, path)


def load(path):
    return nx.Graph(nx.read_pajek(path))


def analysis(G, output):
    data = dict()
    data['clustering'] = nx.clustering(G, weight='weight')
    #data['degree_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')
    if nx.is_connected(G):
        data['shortest_path_average'] = nx.average_shortest_path_length(G, weight='weight')
    data['pagerank'] = nx.pagerank(G, weight='weight')

    with open(os.path.join(output, "analysis.json"), 'w') as out:
        json.dump(data, out)

    T = nx.maximum_spanning_tree(G, weight='weight')
    fig = plt.figure(figsize=(20, 20))
    nx.draw_kamada_kawai(T)
    plt.savefig(os.path.join(output, "mst.png"))


def community_analysis(G, output):
    nx.algorithms.community.centrality.girvan_newman(G)
    nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    # TODO draw community analysis


def simulate_spread(G, steps, threshold):
    exposed_nodes = [node for node in G.nodes(data=True) if node[1]['flaws'] > threshold]
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
    p = dt.plot(width=400, height=400)
    show(p)

    dp = DiffusionPrevalence(model, trends)
    p2 = dp.plot(width=400, height=400)
    show(p2)
