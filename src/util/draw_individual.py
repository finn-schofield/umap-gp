import pygraphviz as pgv
from deap import gp


def draw_individual(individual, datafile, run_type):

    g = pgv.AGraph()
    g.graph_attr['label'] = "{}:{}".format(datafile, run_type).upper()
    for i, expr in enumerate(individual):

        nodes, edges, labels = gp.graph(expr)
        node_map = dict()
        for j in range(len(nodes)):
            current = nodes[j]
            new_node = int(str(i+1)+str(current))
            nodes[j] = new_node
            node_map[current] = new_node

        for j in range(len(edges)):
            edges[j] = (node_map[edges[j][0]], node_map[edges[j][1]])

        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for j in range(len(nodes)):
            n = g.get_node(node_map[j])

            # round constants to 4dp for aesthetics :-)
            try:
                labels[j] = float(labels[j])
                labels[j] = str(round(labels[j], 4))
            except:
                pass

            n.attr["label"] = labels[j]
    return g
