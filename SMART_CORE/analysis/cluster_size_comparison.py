import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    cluster_size_bio_df = pd.read_csv('/home/jwood/OneDrive/SMART/SMART output/Champion_Output/cluster_size_comparison/Cluster Size Comparison_OLI Biology.csv')
    cluster_size_bio_df.plot.box(title="Comparison of Cluster Size (OLI Introduction to Biology)")
    plt.xlabel = ('Skill Model')
    plt.ylabel = ('Number of Assessment Items per Skill')
    plt.show(block=True)
    plt.close()

    cluster_size_chem_df = pd.read_csv('/home/jwood/OneDrive/SMART/SMART output/Champion_Output/cluster_size_comparison/Cluster Size Comparison_OLI Chemistry.csv')
    cluster_size_chem_df.plot.box(title="Comparison of Cluster Size (OLI General Chemistry I)")
    plt.xlabel = ('Skill Model')
    plt.ylabel = ('Number of Assessment Items per Skill')
    plt.show(block=True)
    plt.close()

if __name__ == '__main__':
    main()
