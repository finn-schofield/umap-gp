import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import csv

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

PLOT = False


def main():
    table_file = open("{}/table.csv".format(OUTPUT_DIR), 'w+')
    table_file.write("method,dataset,mean\n")
    for fitness in get_immediate_subdirectories(ROOT_DIR):

        current = os.path.join(ROOT_DIR, fitness)

        summary_file = open("{}/{}-summary.txt".format(OUTPUT_DIR, fitness), "w+")
        count = 0

        for dataset in get_immediate_subdirectories(current):
            summary_file.write("{1} {0} {1}\n\n".format(dataset.upper(), ("=" * 5)))
            dataset_dir = os.path.join(current, dataset)
            plot_data = dict()

            for run_type in get_immediate_subdirectories(dataset_dir):
                summary_file.write(run_type.upper()+"\n")
                run_dir = os.path.join(dataset_dir, run_type)
                fname = "{}:{}".format(dataset, run_type)

                count += 1
                stats = process_run(run_dir)
                if stats is None:
                    continue

                print("processed {}, {} runs".format(fname, len(stats)))
                output_loc = "%s/%s" % (current, fname)
                for method in ['gp']:  # , 'umap']:
                    table_file.write("{}{},{},{}\n".format(fitness, run_type, dataset,
                                                                round(stats["{}-acc".format(method)].mean(), 3)))
                stats.to_csv(output_loc + ".csv")

                # now summarise the results
                # plot_column(stats["total-nodes"].tolist(), 10, fname)
                plot_data[run_type] = stats["total-nodes"].tolist()
                pd.options.display.max_columns = 40

                summary = stats.describe(include='all').drop("count", axis=0)
                print(summary, file=summary_file)
                print(file=summary_file)
            if PLOT:
                plot_graph(plot_data, dataset)
        print("\n{} runs processed".format(count))
    table_file.close()
    table_file = pd.read_csv("{}/table.csv".format(OUTPUT_DIR))
    print(table_file)
    columns = sorted(set(table_file['dataset']))
    ind = []
    for dims in [1, 2, 3, 10]:
        for method in ['umap_cost', 'nrmse', 'umap']:
            ind.append("{}{}".format(method, dims))

    df = pd.DataFrame('0', columns=columns, index=ind)

    for _, row in table_file.iterrows():
        df[row['dataset']][row['method']] = "{:.3f}".format(row['mean'])

    print(df)
    open("{}/table.txt".format(OUTPUT_DIR), 'w+').write(df.to_latex())



def plot_graph(plots, dataset):

    fig, axes = plt.subplots(len(plots), sharex=True)
    axis = 0

    for run_type, data in plots.items():
        axes[axis].hist(data, 10, facecolor='blue', alpha=0.5)
        axes[axis].set_title("{} : {}".format(dataset, run_type))
        axis += 1
    plt.savefig("{}/{}-nodesdist.png".format(OUTPUT_DIR, dataset))


def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))])


def process_run(path):

    stats = pd.DataFrame()
    i = 1
    while os.path.isfile("%s/%d_results.txt" % (path, i)):
        result = pd.read_csv("%s/%d_results.txt" % (path, i))
        result.insert(0, 'run', i)
        stats = pd.concat([stats, result], axis=0)
        i += 1
    if stats.empty:
        return None
    stats = stats.set_index('run')
    # print("processed {} runs.".format(i))
    return stats


def plot_column(data, num_bins, title):
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]

    if len(sys.argv) == 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        OUTPUT_DIR = ROOT_DIR

    main()