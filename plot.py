import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re


def get_tukey_plots(df, title, figsize=(3,3), save=True, path = '/mnt/c/Users/marti/OneDrive/Plocha/vscht/magister/treti_rocnik/mvi/mvi-sp'):
    """Generate Tukey plots for the given metric
    df is a dataframe obtained from running array job workflow, split by dataset - may later include split by ds uin this fn
    Allowed metrics: 'f1', 'roc_auc', 'mcc', 'accuracy', 'precision', 'recall'"""

    df["label"] = (
        df["split_type"] + "_" +
        df["model"]
    )

    figure, ax = plt.subplots(figsize=figsize)

    best_config = df.groupby("label")["score"].mean().idxmax()
    tukey = pairwise_tukeyhsd(endog=df["score"],groups=df['label'], alpha=0.05)
    tukey.plot_simultaneous(figsize=figsize, comparison_name=best_config, ax=ax)
    plt.title(f"{title}")
    plt.tight_layout()

    if save:
        safe_title = re.sub(r"[^\w\s-]", "", title).replace(" ", "_")
        save_path = path + f"{safe_title}.png"

        # Save plot as PNG
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

        print(f"Plot saved as: {save_path}")



metrics = ["f1","matthews_corrcoef","precision","recall","roc_auc"]
datasets = ["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q9Y468","Q12809","Q16637"]

for ds in datasets:
    for metric in metrics:
        df = pd.read_csv(f"./output/benchmarking/{ds}/{metric}/results.txt",sep="\t")

        print(f"Generating Tukey plots for dataset {ds}...")
        get_tukey_plots(df, ds + f'_{metric}', save=True, figsize=(4,3), path = './plots/')