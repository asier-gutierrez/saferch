import os
import argparse
from controller import domain2graph, graph2flaws, graph_analysis, graph_community_analysis, graph_simulate_spread, \
    graph_save, graph_load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Given a domain obtains a communication graph and simulates virus spreading.")
    parser.add_argument("domain", type=str, help="Domain to lookup.")
    parser.add_argument("--geckodriver-path", type=str,
                        help="Path to geckodriver. It is not necessary if it is present in Path.")
    parser.add_argument("--depth", type=int, help="Depth limit to scrap in order to build the graph.", default=25)
    parser.add_argument("--from-compute-step", type=int,
                        help="Step from which start computations. 0 From start, 1 from domain2graph, and 2 from graph2flaws.",
                        default=0)
    parser.add_argument("--community-analysis", action='store_true')
    parser.add_argument("--spread-simulation-steps", type=int, help="Steps for the virus spread simulation.",
                        default=10)
    parser.add_argument("--threshold", type=int, help="Threshold to be used in order to select Exposed nodes.",
                        default=2)
    args = parser.parse_args()

    OUT_PATH = os.path.join('out', args.domain)
    GRAPH_PATH = os.path.join(OUT_PATH, 'graph.net')

    # Make output directory
    os.makedirs(OUT_PATH, exist_ok=True)

    if args.geckodriver_path:
        os.environ["GECKO_DRIVER"] = args.geckodriver_path

    if args.from_compute_step < 1:
        # Obtain graph
        G = domain2graph(args.domain, args.depth)
    else:
        G = graph_load(GRAPH_PATH)

    if args.from_compute_step < 2:
        # Obtain flaws from email addresses
        G = graph2flaws(G)
    else:
        G = graph_load(GRAPH_PATH)

    graph_save(G, GRAPH_PATH)

    # Graph description
    graph_analysis(G, OUT_PATH)

    # Community detection
    if args.community_analysis:
        graph_community_analysis(G, OUT_PATH)

    # Virus spread simulation
    graph_simulate_spread(G, args.spread_simulation_steps, args.threshold)
