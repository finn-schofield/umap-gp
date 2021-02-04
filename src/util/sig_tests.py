from scipy.stats import mannwhitneyu, wilcoxon
import sys
import os
import pandas as pd


def main():
    datafiles = dict()
    mn_rows = []
    wc_rows = []

    for file in sorted(os.listdir(ROOT_DIR)):
        if file.endswith(".csv"):
            df = pd.read_csv("{}/{}".format(ROOT_DIR, file))

            datafiles[file] = (df['umap-acc'].to_list(), df['gp-acc'].to_list())

    for dataname, dataframes in datafiles.items():

        wc_row = [dataname]
        mn_row = [dataname]

        null_data = dataframes[0]
        alt_data = dataframes[1]

        try:
            _, wc_p = wilcoxon(alt_data, null_data)
            _, mn_p = mannwhitneyu(null_data, alt_data)
        except ValueError as e:
            if "length" in str(e):
                raise ValueError("{} lengths not equal".format(dataname)) from e
            else:
                wc_p = 1.0
                mn_p = 1.0

        wc_p = round(wc_p, 5)
        mn_p = round(mn_p, 5)

        wc_row.append(wc_p)
        mn_row.append(mn_p)

        wc_rows.append(wc_row)
        mn_rows.append(mn_row)

    columns = ['datafile', 'acc']

    wc = pd.DataFrame(wc_rows, columns=columns)
    wc = wc.set_index('datafile')

    mn = pd.DataFrame(mn_rows, columns=columns)
    mn = mn.set_index('datafile')

    fl = open("{}/{}significance.txt".format(ROOT_DIR, 'acc'), 'w')
    fl.write("Wilcoxon Results:\n\n")
    fl.write(str(wc))

    fl.write("\n\nMann-Whitney Results:\n\n")
    fl.write(str(mn))


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]
    main()
