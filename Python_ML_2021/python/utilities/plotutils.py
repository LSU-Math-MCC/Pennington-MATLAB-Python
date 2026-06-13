import math

import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib.pyplot import figure
import numpy as np


def normalize_series(series, range=[0, 1]):
    x_min, x_max = series.min(), series.max()
    return series.map(lambda x: (x - x_min)/(x_max - x_min)*(range[1]-range[0]) + range[0])

"""
    Draw scatter plot grid from list of numeric column names (param_list) from dataframe (df) 
with axes_labels = [x_label, y_label].
"""
def param_grid_plot(df, param_list, axes_labels, pointsize_label=None, ncols=2, sf=3, save_name=None):
    #plt.clf()
    #plt.cla()
    plt.style.use('seaborn-whitegrid')

    x_label, y_label = axes_labels
    y_min, y_max = df[y_label].min(), df[y_label].max()
    total_rows = np.sum([math.ceil(len(param_cnames) / ncols) for param_cnames in param_list])
    plt.figure(figsize=[sf * ncols * 2.1, sf * total_rows * 2.4])

    prev_row = 0
    for param_cnames in param_list:
        nrows = math.ceil(len(param_cnames) / ncols)
        for i in range(nrows):
            for j in range(ncols):
                # Don't draw plot without parameter
                if i*ncols + j >= len(param_cnames):
                    continue
                # The index argument for subplot begins at 1 and counts left to right
                ax = plt.subplot(total_rows, ncols, (prev_row + i)*ncols + j + 1, aspect='equal')
                param_cname = param_cnames[i*ncols + j]
                ax.set_title(param_cname, fontsize=sf*10)
                options = np.unique(df[param_cname])
                colors = {opt: c for opt, c in zip(options, ['r', 'b', 'y', 'purple', 'green', 'orange'])}
                for opt in options:
                    df_ = df.loc[df[param_cname] == opt]
                    ax.scatter(df_[x_label],
                               df_[y_label],
                               color=colors[opt],
                               alpha=.25,
                               s=sf*7 if pointsize_label is None
                                    else normalize_series(df_[pointsize_label], [0, sf*7])
                               )
                ax.legend(colors.keys(), loc='lower left', fontsize=sf*7, markerscale=2.5)
                ax.plot([y_min, y_max], [y_min, y_max], color='grey', linestyle=':', linewidth=sf)
        prev_row += nrows

    if isinstance(save_name, str):
        plt.savefig(f'reports/{save_name}.png', format='png')
    return plt


def hist_df(df, columns=None, bins=10):
    if columns is None:
        columns = df.columns

    plt.clf()
    df.hist(column=columns, bins=bins)
    # # fig = plt.gcf()
    # # plt.subplots_adjust(bottom=0.5)
    #
    # text = """1 Provided by mtecResults
    # 2 Mean time per unit distance between two points
    # 3 SA Finished 670 out of 4716 Males
    # 4 SA Finished 754 out of 4756 Males"""
    #
    # plt.figtext(0.1, 0.01, text, ha='left',
    #                            va='bottom',
    #                            fontsize=8
    #                            )
    # # plt.get_axes()[len(plt.get_axes()) - 1].annotate(text, (0.5, 0.01),
    # #                            xycoords='figure fraction', ha='left',
    # #                            va='bottom',
    # #                            fontsize=8
    # #                            )
    # plt.tight_layout()
    # # plt.subplots_adjust(top=0.5)
    # # fig.subplots_adjust(top=.7, bottom = .7)
    plt.show()


def graph_predicted_vs_actual(predicted, actual):
    lo = min(predicted.values.min(), actual.values.min()) * 0.9
    hi = max(predicted.values.max(), actual.values.max()) * 1.1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([lo, hi], [lo, hi], color='grey')

    avg = actual.values.mean()

    sst = np.sum((actual.values - actual.values.mean()) ** 2)
    sse = np.sum((predicted.values - actual.values.mean()) ** 2)
    ssr = np.sum((actual.values - predicted.values) ** 2)
    r2 = 1 - ssr/sst

    print(f"sst: {sst}, sse: {sse}, ssr: {ssr}, r2:{r2}")
    ax.plot([lo, hi], [avg, avg], color='grey')
    ax.scatter(actual.values, predicted.values, s=3)
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.text(0.99, 0.01, 'footnote text\nd\nd', horizontalalignment='right')
    fig.show()

    pass