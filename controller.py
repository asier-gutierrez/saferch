import os, json
import networkx as nx
from connection import scrap_relations, check_mail
from graph import relations_graph_attach, analysis, community_analysis, simulate_spread, save, load, draw, connected, \
    draw_infected
from ploting import flaw_bar_plot


def domain2graph(domain, depth):
    G = nx.Graph()
    relations = scrap_relations(domain, depth)
    relations_graph_attach(G, relations)
    return G


def graph2flaws(G):
    flaws = dict()
    for mail in G.nodes():
        flaws[mail] = check_mail(mail)
    nx.set_node_attributes(G, flaws, 'flaws')
    return G


def graph_save(G, path):
    save(G, path)


def graph_load(path):
    try:
        return load(path)
    except:
        print("Graph not found.")


def graph_connected(G, strategy):
    G = connected(G, strategy)
    return G


def graph_analysis(G, output):
    analysis(G, output)


def graph_draw(G, output):
    draw(G, output)


def graph_community_analysis(G, output):
    community_analysis(G, output)


def graph_simulate_spread(G, steps, threshold, output):
    return simulate_spread(G, steps, threshold, output)


def graph_draw_infected(G, infected_nodes, output):
    draw_infected(G, infected_nodes, output)


def graph_flaw_bar_plot(G, output):
    flaw_bar_plot(G, output)
