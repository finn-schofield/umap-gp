from threadpoolctl import threadpool_limits

import warnings
import sys
import os
import re
import csv

import numpy as np
import pandas as pd
from sklearn.utils import parallel_backend

import vector_tree as vt
import multi_tree as mt

import matplotlib.pyplot as plt

from deap import gp
from deap import base
from deap import creator

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import pairwise_distances

from umap import UMAP
from umap.umap_ import fuzzy_simplicial_set
from eval import umap_cost

from ea_simple_elitism import eaSimple

from selection import *
from util.draw_individual import draw_individual

from scipy.stats import pearsonr

import run_data


rd = run_data.RunData()
CXPB = 0.8
MUTPB = 0.2
ELITISM = 10
CMPLX = "nodes_total"  # complexity measure of individuals
BCKT = no_bucketing  # bucketing used for lexicographic parsimony pressure
BCKT_VAL = 5  # bucketing parameter
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
MT_CX = "aic"  # crossover for multi-tree {'aic', 'ric', 'sic'}

# MAX_E = None    # used for normalising with nrmse
# MIN_E = None


def evaluate(individual, toolbox, data, embedding, metric):
    """
    Evaluates an individuals fitness. The fitness is the clustering performance on the data
    using the specified metric.

    :param individual: the individual to be evaluated
    :param toolbox: the evolutionary toolbox
    :param data: the data to be used to evaluate the individual
    :param embedding: the UMAP embedding to measure against
    :param metric: the metric to be used to evaluate embedding difference
    :return: the fitness of the individual
    """

    X = REP.process_data(individual, toolbox, data)


    if metric == 'spearmans':
        return (1.0 - spearmans(embedding, X))/2.0,
    elif metric == 'pearsons':
        # #lol GP makin' constantz
        # if np.any(np.all(X == X[0,:],axis=0)):
        #     return 1.,
        # corrcoef = np.corrcoef(X.T, embedding.T)[0, 1]
        # #print(corrcoef)
        # return (1.0 - np.abs(corrcoef)) / 2.0,

        f = pearsons(X.T, embedding.T)
        return 1-f,
    elif metric == "mse":
        return MSE(embedding, X, squared=False),
    elif metric == "nrmse":
        errors = MSE(embedding, X, multioutput='raw_values')
        total_error = 0
        for error in enumerate(errors):
            total_error += error[1]/(MAX_E[error[0]] - MIN_E[error[0]])
        return total_error,
    elif metric == "umap_cost":
        return umap_cost(X, v),

    else:
        raise Exception("invalid metric: {}".format(metric))


def pearsons(data_t, embedding_t):
    abs_pearson = np.zeros(len(data_t))
    for of in range(len(data_t)):
        # for constant features, clearly zero correlation to them, no?
        if (data_t[of, :] == data_t[of, :][0]).all():
            abs_pearson[of] = 0.
        else:
            pearson = np.corrcoef(data_t[of], embedding_t[of])[0, 1]
            abs_pearson[of] = np.abs(pearson)
    return abs_pearson.sum() / len(data_t)


def spearmans(o1, o2):
    d = np.abs(o1 - o2) ** 2
    n = len(o1)
    rho = 1 - (6 * d.sum() / (n * ((n ** 2) - 1)))
    return rho


def eval_complexity(individual, measure):
    """
    Evaluates the complexity of an individual using the given measure.

    :param individual: the individual to be evaluated
    :param measure: the measure of individual complexity
    :return: individuals complexity
    """

    if REP is mt:
        con_fts = [str(tree) for tree in individual]
    elif REP is vt:
        con_fts = vt.parse_tree(individual)
    else:
        raise Exception("Invalid representation")

    if measure == "cf_count":
        complexity = len(con_fts)
    elif measure == "unique_fts":
        unique_fts = set()
        pat = re.compile("f[\\d]+")

        for cf in con_fts:
            unique_fts.update(re.findall(pat, cf))

        complexity = len(unique_fts)
    elif measure == "nodes_avg" or measure == "nodes_total":
        total_nodes = 0
        for cf in con_fts:
            total_nodes += 1  # root node
            for i in range(len(cf)):
                # each node is preceded by either a comma or an opening bracket (except root)
                if cf[i] == ',' or cf[i] == '(':
                    total_nodes += 1
        if measure == "nodes_avg":
            complexity = total_nodes / len(con_fts)
        else:
            complexity = total_nodes
    else:
        raise Exception("Invalid complexity metric: %s" % measure)

    return complexity


def plot_silhouette(silhouette, title):
    plt.hist(silhouette, bins=30)
    plt.title(title)
    plt.show()


def write_ind_to_file(ind, run_num, results):
    """
    Writes the attributes of an individual to a csv file.

    :param run_num: the number of the current run
    :param ind: the individual
    :param results: a dictionary of results, titles to values
    """

    line_list = []

    # add constructed features to lines
    if REP is mt:
        for cf in [str(tree) for tree in ind]:
            line_list.append(cf + "\n")
    elif REP is vt:
        for cf in vt.parse_tree(ind):
            line_list.append(cf + "\n")
    else:
        raise Exception("Invalid representation")

    line_list.append("\n")

    fname = "{}/{}_ind.txt".format(rd.outdir, run_num)
    if not os.path.exists(fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    fl = open(fname, 'w')
    fl.writelines(line_list)
    fl.close()

    csv_columns = results.keys()
    csv_file = "{}/{}_results.txt".format(rd.outdir, run_num)

    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(results)


def init_toolbox(toolbox, pset, n_trees):
    """
    Initialises the toolbox with evolutionary operators.

    :param toolbox: the toolbox to initialise
    :param pset: primitive set for evolution
    """
    if REP is mt:
        mt.init_toolbox(toolbox, pset, MT_CX, n_trees)
    if REP is vt:
        mt.init_toolbox(toolbox, pset, MT_CX)

    if rd.use_parsimony:
        toolbox.register("eval_complexity", eval_complexity, measure=CMPLX)
        toolbox.register("bucket", BCKT, BCKT_VAL)
        toolbox.register("select", parsimony_tournament, tournsize=7, toolbox=toolbox)
    else:
        toolbox.register("select", tools.selTournament, tournsize=7)


def init_stats():
    """
    Initialises a MultiStatistics object to capture data.

    :return: the MultiStatistics object
    """
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    if CMPLX == 'unique_fts':
        unique_stats = tools.Statistics(lambda ind: eval_complexity(ind, "nodes_total"))
        stats = tools.MultiStatistics(fitness=fitness_stats, unique_fts=unique_stats)
    else:
        nodes_stats = tools.Statistics(lambda ind: eval_complexity(ind, "nodes_total"))
        stats = tools.MultiStatistics(fitness=fitness_stats, total_nodes=nodes_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return stats


def final_evaluation(best, data, labels, umap, toolbox, print_output=True):
    """
    Performs a final performance evaluation on an individual.

    :param best: the individual to evaluate
    :param data: the dataset associated with the individual
    :param labels: the labels of the dataset
    :param umap: umap embedding
    :param toolbox: the evolutionary toolbox
    :param print_output: whether or not to print output
    :return: a dictionary of results, titles to values
    """

    X = REP.process_data(best, toolbox, data)
    print(X)
    print(umap.embedding_)

    X_a, X_b = zip(*X)
    # plt.plot(X_a, X_b, 'bo')
    plt.scatter(X_a, X_b, c=labels, marker='o', s=20)
    # plt.xlim(np.min(X_a), np.max(X_a))
    # plt.xlim(np.min(X_b), np.max(X_b))
    plt.title('GP')
    plt.show()

    X_a, X_b = zip(*umap.embedding_)
    # plt.plot(X_a, X_b, 'bo')
    plt.scatter(X_a, X_b, c=labels, marker='o',s=20)
    # plt.xlim(np.min(X_a),np.max(X_a))
    # plt.xlim(np.min(X_b), np.max(X_b))
    plt.title('UMAP')
    plt.show()
    best_spearmans = evaluate(best, toolbox, data, umap.embedding_, "spearmans")[0]
    best_mse = evaluate(best, toolbox, data, umap.embedding_, "mse")[0]

    unique = eval_complexity(best, "unique_fts")
    nodes = eval_complexity(best, "nodes_total")

    if print_output:
        print("Unique features: %d\n" % unique)
        print("Total nodes: %d\n" % nodes)
        print("Best MSE: %f \n" % best_mse)
        print("Best Spearmans: %f \n" % best_spearmans)

    return {"unique-fts": unique, "total-nodes": nodes, "best-mse": best_mse,
            "best-spearmans": best_spearmans}


def plot_stats(logbook):
    """
    Generates plots of the statistics gathered from the evolutionary process.
    :param logbook: a logbook of the statistics.
    """
    gen = logbook.select("gen")
    fit_max = logbook.chapters["fitness"].select("max")
    nodes_avg = logbook.chapters["total_nodes"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, nodes_avg, "r-", label="Average Nodes")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def main():

    random.seed(rd.seed)
    umap = UMAP(n_components=rd.n_dims, random_state=rd.seed).fit(rd.data)

    if rd.measure == "nrmse":
        global MAX_E
        global MIN_E
        MAX_E = np.amax(umap.embedding_.T, 1)
        MIN_E = np.amin(umap.embedding_.T, 1)

    if rd.measure == "umap_cost":
        global v
        v = fuzzy_simplicial_set(
            rd.data,
            15,
            np.random.RandomState(rd.seed),
            "euclidean"
        )[0].todense()
        print("UMAP Embedding Cost: {}".format(umap_cost(umap.embedding_, v)))



    num_classes = len(set(rd.labels))
    print("%d classes found." % num_classes)
    # distance_vector = pairwise_distances(rd.data)

    pset = gp.PrimitiveSet("MAIN", rd.num_features, prefix="f")
    pset.context["array"] = np.array
    REP.init_primitives(pset, rd.use_ercs)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # set up toolbox
    toolbox = base.Toolbox()
    init_toolbox(toolbox, pset, rd.n_dims)

    toolbox.register("evaluate", evaluate, toolbox=toolbox, data=rd.data,
                     metric=rd.measure, embedding=umap.embedding_)

    pop = toolbox.population(n=rd.pop)
    hof = tools.HallOfFame(1)

    stats = init_stats()

    pop, logbook = eaSimple(pop, toolbox, CXPB, MUTPB, ELITISM, rd.gens, stats, halloffame=hof, verbose=True)

    # TODO: re-implement outputting of run data

    # for chapter in logbook.chapters:
    #     logbook_df = pd.DataFrame(logbook.chapters[chapter])
    #     logbook_df.to_csv("%s_%d.csv" % (chapter, run_num), index=False)

    best = hof[0]
    res = final_evaluation(best, rd.data, rd.labels, umap, toolbox)
    # evaluate(best, toolbox, data, num_classes, 'silhouette_pre', distance_vector=distance_vector,
    #          plot_sil=True)
    write_ind_to_file(best, rd.seed, res)

    # TODO: fix string passed to individuals
    draw_individual(best, rd.dataset, "").draw("{}/{}-{}-best.png".format(rd.outdir, rd.seed, rd.dataset))

    return pop, stats, hof


"""
[seed] [data file] [{parsimony, noparsimony}]
"""
if __name__ == "__main__":
    run_data.init_data(rd)
    main()



