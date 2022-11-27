import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1], title=""):
    """
    Scatter plot feat1 vs feat2.
    Assumes +/- binary labels.
    Plots first and second columns by default.
    Args:
      - df: dataframe with feat1, feat2, and labels
      - feat1: column name of first feature
      - feat2: column name of second feature
      - labels: column name of labels
      - weights: [w1, w2, b]
    """
    colors = pd.Series(["b" if label > 0 else "r" for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind="scatter", c=colors)

    xmin, xmax = ax.get_xlim()
    c, a, b = weights

    def y(x):
        return (-a / b) * x - c / b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color="black")
    ax.add_line(line)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.savefig("./scatterplot.png")
    plt.show()


def main():
    # import data
    inpath = os.path.join(os.getcwd(), sys.argv[1])
    outpath = os.path.join(os.getcwd(), sys.argv[2])
    df = pd.read_csv(inpath, header=None)
    data = np.asmatrix(df, dtype="float64")
    features, labels = data[:, :-1], data[:, -1]

    # initialize weights
    w = np.zeros(shape=(1, 3))
    ws = np.empty(shape=[0, 3])  # 2D

    for _ in range(20):
        for feature, label in zip(features, labels):
            feature = np.insert(feature, 0, 1)  # add bias column
            fx = np.dot(w, feature.transpose())
            if label * fx <= 0:
                w += (feature * label.item(0, 0)).tolist()
        ws = np.vstack((ws, w))

    visualize_scatter(df, weights=w[-1])

    results_df = pd.DataFrame(data=ws)
    # reorder columns
    results_df = results_df[[results_df.columns[i] for i in [1, 2, 0]]]
    results_df.to_csv(outpath, index=False, header=False)


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
