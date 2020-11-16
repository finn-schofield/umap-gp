import argparse
from read_data import read_data


class RunData(object):
    def __init__(self):
        self.elitism = 1

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def init_data(rd):
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="Integer for seeding random generator", type=int)
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="wine")
    parser.add_argument("--dir", help="Dataset directory", type=str, default="../data")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int, default=1000)
    parser.add_argument("-p", "--pop", help="Size of population", type=int, default=100)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("--erc", dest='use_ercs', help="Use ephemeral random constants", action='store_true')
    parser.add_argument("--parsimony", help="use parsimony pressure", dest='use_parsimony', action='store_true')
    parser.add_argument("--dim", dest="n_dims", type=int, default=2)
    parser.add_argument("-m", "--measure", help="Measure to be used for fitness", type=str, default="spearmans",
                        choices=["spearmans", "mse"])

    parser.set_defaults(use_parsimony=False)
    parser.set_defaults(use_ercs=False)

    args = parser.parse_args()
    print(args)
    update_experiment_data(rd, args)

    rd.all_data = read_data("%s/%s.data" % (args.dir, args.dataset))
    rd.data = rd.all_data["data"]
    rd.labels = rd.all_data["labels"]
    rd.num_instances = rd.data.shape[0]
    rd.num_features = rd.data.shape[1]


def update_experiment_data(data, ns):
    dict = vars(ns)
    for i in dict:
        setattr(data, i, dict[i])
        # data[i] = dict[i]

