import numpy as np
import torch
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from remap_label import remap_labels
from sklearn.metrics import precision_score,f1_score,recall_score,confusion_matrix
import umap
from cycler import cycler

def visualizer_hook(umapper, umap_embeddings, labels, *args):
    logging.info("UMAP plot")
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20,15))
    plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()

def fcm_k_means(output_train_features,output_test_features,label_train,label_test,evaluate_on_fcm):

    ncenters = 7
    init_matrix = np.zeros((label_train.size, label_train.max()+1))
    init_matrix[np.arange(label_train.size),label_train] = 1

    if evaluate_on_fcm:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                 output_train_features.transpose(1,0), ncenters, 2, error=0.000005, maxiter=1000000, init=init_matrix.transpose(1,0))
        u,u0,d,jm,p,fpc = fuzz.cluster.cmeans_predict(
            output_test_features.transpose(1,0),cntr,2,error=0.000005,maxiter=1000000)
        pred = u.argmax(axis=0)
    else:
        kmeans = KMeans(7, n_init=1000)
        pred = kmeans.fit_predict(output_test_features)

    accuracy,remapped_labels = remap_labels(pred,label_test)
    print('accuracy clustering:',accuracy)
    print("Precision: %2.6f" % precision_score(label_test, remapped_labels, average='micro'))
    print("Recall: %2.6f" % recall_score(label_test, remapped_labels, average='micro'))
    print("F1 Score: %2.6f" % f1_score(label_test, remapped_labels, average='micro'))
    print("Confusion Matrix:\n", confusion_matrix(label_test, remapped_labels), '\n')

    return remapped_labels

def get_list_label(data):
    label = []
    for data,label in data:
        label.append(label)
    return label = np.array(label)


def visualize_embedding_pred_n_gt(output_test_features,label_test,remaped_label):
    umapper = umap.UMAP()
    embeddings = umapper.fit_transform(output_test_features)

    visualizer_hook(umapper,embeddings,label_test)
    visualizer_hook(umapper,embeddings,remaped_label)
