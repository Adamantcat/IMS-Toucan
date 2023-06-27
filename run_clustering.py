from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
matplotlib.use("Agg")


def read_features(filename):
    df = pd.read_csv(filename, sep=";")
    return df

def read_json(filename):
    with open(filename, 'r') as file:
        features = json.load(file)
    return features


def get_adjacency_matrix(features_dict):
   # features = features_df.to_numpy()[:,2:] # we don't want filename and frameTime for clustering
    features = []
    for f in features_dict.keys():
        features.append(features_dict[f]['features'])
    features_df = pd.DataFrame.from_dict(features, orient='columns')

    print(features_df.columns)
    adj_matrix = kneighbors_graph(features, n_neighbors=10).toarray()
    return adj_matrix

def cluster(features_df, k=10, save_file_path=None):
    adj_matrix = get_adjacency_matrix(features_df)
    labels = spectral_clustering(adj_matrix, n_clusters=k, assign_labels="cluster_qr")
    print(labels)
    labels_df = features_df.assign(label=labels)

    if save_file_path is not None:
        labels_df.to_csv(save_file_path, sep=";", columns=['name', 'label'])

    return labels_df

def reduce(data):
    X = data.to_numpy()[:,2:-1]
    Y = data['label']

    # print(X)
    # print(X.shape)
    # print(Y)

def plot(data, title, save_file_path, legend, colors, plot_tsne=True):

    X = data.to_numpy()[:,2:-1]
    labels = data['label']
    filenames = data['name']
    names = [sub.replace('/mount/arbeitsdaten/textklang/synthesis/styles', '') for sub in filenames]

    tsne = TSNE(n_jobs=-1, learning_rate="auto", init="pca", verbose=1, n_iter_without_progress=20000, n_iter=60000)
    pca = PCA(n_components=2)
    
    if plot_tsne:
        projected_data = tsne.fit_transform(X)
    else:
        projected_data = pca.fit_transform(X)

    if colors is None:
        colors = cm.gist_rainbow(np.linspace(0, 1, len(set(labels))))
    label_to_color = dict()
    for index, label in enumerate(sorted(list(set(labels)))):
        label_to_color[label] = colors[index]

    labels_to_points_x = dict()
    labels_to_points_y = dict()
    for label in labels:
        labels_to_points_x[label] = list()
        labels_to_points_y[label] = list()
    for index, label in enumerate(labels):
        labels_to_points_x[label].append(projected_data[index][0])
        labels_to_points_y[label].append(projected_data[index][1])

    fig, ax = plt.subplots()
    for label in sorted(list(set(labels))):
        x = np.array(labels_to_points_x[label])
        y = np.array(labels_to_points_y[label])
        ax.scatter(x=x,
                    y=y,
                    c=label_to_color[label],
                    label=label,
                    alpha=0.9)
    if legend:
        ax.legend()
    fig.tight_layout()
    ax.axis('off')
    fig.subplots_adjust(top=0.9, bottom=0.0, right=1.0, left=0.0)
    ax.set_title(title)
    if save_file_path is not None:
        plt.savefig(save_file_path)
    else:
        plt.show()
    plt.close()

# EXPERIMENTAL: find number of clusters based on data   
def find_best_k(adj_matrix):
    # from https://towardsdatascience.com/spectral-clustering-aba2640c0d5b
    # create the graph laplacian
    diag_matrix = np.diag(adj_matrix.sum(axis=1))
    laplace_matrix = diag_matrix - adj_matrix

    # find the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(laplace_matrix)

    # sort
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # use Fiedler value to find best cut to separate data
    k = vecs[:,1] > 0 # TODO: somehow this always returns true

    return k



if __name__ == '__main__':
    #features = read_features("/mount/arbeitsdaten/textklang/synthesis/styles/Clustering/features.is09_emotion.csv")
    features = read_json("poems_features_eGeMAPS_mfcc20.json")
    # print(len(features.keys()))
    # print(features.head)
    # print(features.size)
    # print(features.columns)

    adjacency_matrix = get_adjacency_matrix(features)
    # print(adjacency_matrix)
    # k = find_best_k(adjacency_matrix)
    # print(k)
    #labels = cluster(features, k=5, save_file_path="/mount/arbeitsdaten/textklang/synthesis/styles/Clustering/labels.is09_emo.csv")
    #print(labels)
    #plot(labels, title="EMO Verses PCA", save_file_path="vis/is09_emo_pca.png", legend=True, colors=None, plot_tsne=False)