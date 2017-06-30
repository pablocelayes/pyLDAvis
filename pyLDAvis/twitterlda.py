"""
pyLDAvis sklearn
===============
Helper functions to visualize sklearn's LatentDirichletAllocation models
"""

import funcy as fp
import pyLDAvis
import numpy as np
from os.path import join
from os import listdir

def _get_doc_lengths(model_data_path):
    with open(join(model_data_path, 'DocLengths.txt')) as f:
        return [int(l.strip()) for l in f.readlines() if l]


def _get_term_freqs(model_data_path):
    with open(join(model_data_path, 'TermFreqs.txt')) as f:
        return [int(l.strip()) for l in f.readlines() if l]

def _get_sample_docs(model_data_path,  n_topics, n_docs=10):
    labeledtweets_path = join(model_data_path, 'TextWithLabel')

    rawtweets_path = '/'.join(model_data_path.split('/')[:-1])\
                        .replace('ModelRes', 'rawtweets')

    clusters = [[] for i in range(n_topics)]
    for userfname in listdir(labeledtweets_path):
        labeled_tweets = open(join(labeledtweets_path, userfname)).readlines()
        raw_tweets = open(join(rawtweets_path, userfname)).readlines()

        for raw, labeled in zip(raw_tweets, labeled_tweets):
            _, topic, terms = labeled.split(':')
            topic_ind = int(topic.split('=')[-1])
            # naive relevance just by proportion of terms coming from main topic
            # TODO: implement something better
            terms = terms.strip().split()
            n_terms = len(terms) or 1
            n_topic_terms = len([t for t in terms if t.split('/')[-1] == str(topic_ind)])
            relevance = n_topic_terms * 1.0 / n_terms
            clusters[topic_ind].append((raw.strip(), relevance))

    for t in range(n_topics):
        clusters[t] = sorted(clusters[t], key= lambda x: -x[1])[:n_docs]    
    
    return clusters    

def _get_vocab(model_data_path):
    with open(join(model_data_path, 'uniWordMap.txt')) as f:
        return [l.strip() for l in f.readlines() if l]


def _row_norm(dists):
    # row normalization function required
    # for doc_topic_dists and topic_term_dists
    return dists / dists.sum(axis=1)[:, None]


def _get_doc_topic_dists(model_data_path, n_topics):
    with open(join(model_data_path, 'DocTopics.txt')) as f:
        doc_topics = [int(l.strip()) for l in f.readlines() if l]
    n_docs = len(doc_topics)

    doc_topic_dists = np.zeros((n_docs, n_topics))

    for i, j in enumerate(doc_topics):
        doc_topic_dists[i,j] = 1

    return doc_topic_dists


def _get_topic_term_dists(model_data_path):
    return np.genfromtxt(join(model_data_path, 'TopicTermDists.txt'))


def _extract_data(model_data_path, ignore_topics=[], ignore_terms=[]):
    vocab = _get_vocab(model_data_path)
    doc_lengths = _get_doc_lengths(model_data_path)
    term_freqs = _get_term_freqs(model_data_path)
    topic_term_dists = _get_topic_term_dists(model_data_path)
    n_topics = topic_term_dists.shape[0]
    sample_docs = _get_sample_docs(model_data_path, n_topics)

    assert len(term_freqs) == len(vocab), \
        ('Term frequencies and vocabulary are of different sizes, {} != {}.'
         .format(term_freqs.shape[0], len(vocab)))

    # column dimensions of document-term matrix and topic-term distributions
    # must match first before transforming to document-topic distributions
    doc_topic_dists = _get_doc_topic_dists(model_data_path, n_topics)

    if ignore_topics:
        ignore_topic_inds = [i-1 for i in ignore_topics]
        include_inds = [i for i in range(n_topics) if i not in ignore_topic_inds]
        topic_term_dists = topic_term_dists[include_inds,:]

        filter_docs = (doc_topic_dists[:, ignore_topic_inds].sum(axis=1) == 0)
        include_docs = [i for i, v in enumerate(filter_docs) if v]

        doc_topic_dists = doc_topic_dists[filter_docs,:]
        doc_topic_dists = doc_topic_dists[:,include_inds]

        doc_lengths = [doc_lengths[i] for i in include_docs]

    if ignore_terms:
        _vocab = []
        ignore_term_inds = []
        for i, t in enumerate(vocab):
            if t in ignore_terms:
                ignore_term_inds.append(i)
            else:
                _vocab.append(t)
        include_term_inds = [i for i in range(len(vocab)) if i not in ignore_term_inds]
        term_freqs = [f for i, f in enumerate(term_freqs) if i not in ignore_term_inds]
        topic_term_dists = _row_norm(topic_term_dists[:,include_term_inds]) 

        vocab = _vocab

    return {'vocab': vocab,
            'doc_lengths': doc_lengths,
            'term_frequency': term_freqs,
            'sample_docs': sample_docs,
            'doc_topic_dists': doc_topic_dists.tolist(),
            'topic_term_dists': topic_term_dists.tolist()}


def prepare(model_data_path, ignore_topics=[], ignore_terms=[], **kwargs):
    """Create Prepared Data from sklearn's LatentDirichletAllocation and CountVectorizer.

    Parameters
    ----------
    model_data_path : Path where TwitterLDA stored it's data output

    Returns
    -------
    prepared_data : PreparedData
          the data structures used in the visualization


    Example
    --------
    For example usage please see this notebook:
    http://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/sklearn.ipynb

    See
    ------
    See `pyLDAvis.prepare` for **kwargs.
    """
    opts = fp.merge(_extract_data(model_data_path, ignore_topics,  ignore_terms), kwargs)
    opts['sort_topics'] = False
    return pyLDAvis.prepare(**opts)
