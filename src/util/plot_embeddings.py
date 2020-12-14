import sys
import os
import matplotlib.pyplot as plt
from read_data import read_data


def plot_embedding(embedding, labels, output_fname):
    X_a, X_b = zip(*embedding)
    plt.scatter(X_a, X_b, c=labels, marker='o', s=20)
    # plt.xlim(np.min(X_a), np.max(X_a))
    # plt.xlim(np.min(X_b), np.max(X_b))
    plt.title('GP')
    plt.savefig(output_fname)


def plot_all_in_dir(dir, dataset):
    i = 1
    while os.path.isfile("{}/{}_emb.data" .format(dir, i)):
        data = read_data("{}/{}_emb.data" .format(dir, i))
        plot_embedding(data['data'], data['labels'], "{}/{}_{}_emb.png" .format(OUTPUT_DIR, dataset, i))
        i += 1


def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))])


def main():
    count = 0

    for dataset in get_immediate_subdirectories(ROOT_DIR):
        dataset_dir = os.path.join(ROOT_DIR, dataset)

        count += 1
        plot_all_in_dir(dataset_dir, dataset)

    print("\n{} runs processed".format(count))


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]
    if len(sys.argv) == 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        OUTPUT_DIR = ROOT_DIR

    main()
