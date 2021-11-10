import pickle
import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    # Currently only supports 2D visualization
    with open("labels_and_latents_export.pkl", "rb") as f:
        labels, latents = pickle.load(f)

    if latents.shape[1] > 2:
        pca = TSNE(n_components=2)
        latents = pca.fit_transform(latents)
        # print(pca.explained_variance_ratio_)

    for label in set(labels):
        subsetX = latents[labels==label]
        plt.scatter(subsetX[:, 0], subsetX[:, 1], label=f"Class {label}")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()