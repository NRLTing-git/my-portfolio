import requests
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import matplotlib.pyplot as plt

def dcos(vec1, vec2):
    """Compute cosine distance between vec1 and vec2

    If `vec1` and `vec2` are same-sized matrices, an ndarray of the cosine
    distance of corresponding rows will be returned instead.

    Parameters
    ----------
    vec1 : ndarray
        First vector
    vec2 : ndarray
        Second vector

    Returns
    -------
    float
        cosine distance of `vec1` and `vec2`
    """
    return 1 - cosine_similarity(vec1, vec2)

def get_confusion(actual, results, all_labels):
    """Accept the label of the correct class, the returned results as indices
    to the objects and all labels, and return the confusion matrix as a
    pandas DataFrame."""
    conf = {
        'relevant': {
            'relevant': 0,
            'irrelevant': 0
        },
        'irrelevant': {
            'relevant': 0,
            'irrelevant': 0
        },
    }

    for i, j in enumerate(all_labels):
        if i in results:
            if j == actual:
                conf['relevant']['relevant'] += 1
            else:
                conf['irrelevant']['relevant'] += 1
        else:
            if j == actual:
                conf['relevant']['irrelevant'] += 1
            else:
                conf['irrelevant']['irrelevant'] += 1

    return pd.DataFrame(conf)

def nearest_k(query, objects, k, dist):
    """Return the indices to objects most similar to query

    Parameters
    ----------
    query : ndarray
        query object represented in the same form vector representation as the
        objects
    objects : ndarray
        vector-represented objects in the database; rows correspond to
        objects, columns correspond to features
    k : int
        number of most similar objects to return
    dist : function
        accepts two ndarrays as parameters then returns their distance

    Returns
    -------
    most_similar : ndarray
        Indices to the most similar objects in the database
    """
    return np.argsort(dist(query, objects).flatten(), kind="stable")[:k]

def precision(confusion):
    """Return precision from given confusion matrix"""
    return confusion.iat[0,0]/confusion.iloc[0,:].sum()

def recall(confusion):
    """Return recall from given confusion matrix"""
    return confusion.iat[0,0]/confusion.iloc[:,0].sum()

def f_measure(precision, recall, beta=1):
    """Return F-measure from given precision, recall, and beta"""
    return ((1 + beta**2) * (precision * recall)
            / ((beta**2 * precision) + recall))

def pr_curve(query, objects, dist, actual, all_labels):
    """Draw PR curve

    Parameters
    ----------
    query: array-like
        find objects similar to this query
    objects: ndarray
        database of objects to search in
    dist: function
        function that returns the distance of two input `ndarray`s
    actual: int
        class label
    all_labels: array-like
        label of each object in the database

    Returns
    -------
    matplotlib.Axes
        rendered PR curve
    """
    all_labels = np.asarray(all_labels)
    results = nearest_k(query, objects, len(all_labels), dist)
    rs = (all_labels[results] == actual).cumsum()
    N = (all_labels == actual).sum()
    precisions = rs / np.arange(1, len(rs) + 1)
    recalls = rs / N
    recalls = [0] + recalls.tolist()
    precisions = [1] + precisions.tolist()

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.step(recalls, precisions, where="post")
    ax.fill_between(recalls, precisions, step="post", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title('Figure 2. Precision-Recall Curve')
    aucpr = auc_pr(query, objects, dist,
                    actual,all_labels)
    ax.text(0.3, 0.3, f"AUC-PR: {aucpr:.4f}",
            bbox=dict(alpha=0.5))
    txt="""
    The PR curve retains a high precision when k is low,
    and quickly dips afterwards. The curve shows that after
    getting the top 3 artists which were true positives,
    the next 28 mentors were spread out until k reaches the maximum."""
    plt.figtext(0, -0.15, txt, wrap=True, horizontalalignment='left',
                fontsize=12)
    return ax

def auc_pr(query, objects, dist, actual, all_labels):
    """Compute area under the PR curve

    Parameters
    ----------
    query: array-like
        find objects similar to this query
    objects: numpy.ndarray
        database of objects to search in
    dist: function
        function that returns the distance of two input `ndarray`s
    actual: int
        class label
    all_labels: array-like
        label of each object in the database

    Returns
    -------
    float
        area under the PR curve
    """
    from scipy.integrate import trapz

    all_labels = np.asarray(all_labels)
    results = nearest_k(query, objects, len(all_labels), dist)
    rs = (all_labels[results] == actual).cumsum()
    N = (all_labels == actual).sum()
    precisions = rs / np.arange(1, len(rs) + 1)
    recalls = rs / N
    recalls = [0] + recalls.tolist()
    precisions = [1] + precisions.tolist()
    return trapz(precisions, recalls)