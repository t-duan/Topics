#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing Text Data, Creating Matrices and Cleaning Corpora
***************************************************************

Functions of this module are for **preprocessing purpose**. You can read text \
files, `tokenize <https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)>`_ \
and segment documents (if a document is chunked into smaller segments, each segment \
counts as one document), create and read `document-term matrices <https://en.wikipedia.org/wiki/Document-term_matrix>`_, \
determine and remove features. Recurrent variable names are based on the following \
conventions:

    * ``corpus`` means an iterable containing at least one ``document``.
    * ``document`` means one single string containing all characters of a text \
    file, including whitespaces, punctuations, numbers, etc.
    * ``dkpro_document`` means a pandas DataFrame containing tokens and additional \
    information, e.g. *part-of-speech tags* or *lemmas*, produced by `DARIAH-DKPro-Wrapper <https://github.com/DARIAH-DE/DARIAH-DKPro-Wrapper>`_.
    * ``tokenized_corpus`` means an iterable containing at least one ``tokenized_document`` \
    or ``dkpro_document``.
    * ``tokenized_document`` means an iterable containing tokens of a ``document``.
    * ``document_labels`` means an iterable containing names of each ``document`` \
    and must have as much elements as ``corpus`` or ``tokenized_corpus`` does.
    * ``document_term_matrix`` means either a pandas DataFrame with rows corresponding to \
    ``document_labels`` and columns to types (distinct tokens in the corpus), whose \
    values are token frequencies, or a pandas DataFrame with a MultiIndex \
    and only one column corresponding to word frequencies. The first column of the \
    MultiIndex corresponds to a document ID (based on ``document_labels``) and the \
    second column to a type ID. The first variant is designed for small and the \
    second for large corpora.
    * ``token2id`` means a dictionary containing a token as key and an unique identifier \
    as key, e.g. ``{'first_document': 0, 'second_document': 1}``.

Contents
********
    * :func:`add_token2id` adds a token to a ``document_ids`` or ``type_ids`` dictionary \
    and assigns an unique identifier.
    * :func:`read_matrix_market_file()` reads a `Matrix Market <http://math.nist.gov/MatrixMarket/formats.html#MMformat>`_ \
    file for `Gensim <https://radimrehurek.com/gensim/>`_. 
    * :func:`read_token2id()` reads a ``document_ids`` or ``type_ids`` dictionary \
    from a CSV file.
"""


from gensim.corpora import MmCorpus
import os
import pandas as pd
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(name)s: %(message)s')


def add_token2id(token, token2id):
    """Adds token to token2id dictionary.

    With this function you can append a ``token`` to an existing ``token2id`` \
    dictionary. If ``token2id`` has *x* elements with *n* identifiers, ``token`` \
    will be element *x + 1* with identifier *n + 1*. 

    Args:
        token (str): Token.
        token2id (dict): A dictionary with tokens as keys and identifiers as values.

    Returns:
        An extended token2id dictionary.
    
    Raises:
        ValueError, if ``token`` has alread an ID in ``token2id``.
        
    Example:
        >>> token = 'example'
        >>> token2id = {'text': 0}
        >>> len(add_token2id(token, token2id)) == 2
        True
    """
    if token in token2id.keys():
        raise KeyError("{} has already an ID in token2id. Access its value with token2id[token].".format(token))
    token2id[token] = len(token2id) + 1
    return token2id


def read_matrix_market_file(filepath):
    """Reads a Matrix Market file for Gensim.

    With this function you can read a Matrix Market file to process it with \
    `Gensim <https://radimrehurek.com/gensim/>`_.

    Args:
        filepath (str): Path to Matrix Market file.

    Returns:
        Matrix Market model for Gensim.
    """
    if os.path.splitext(filepath)[1] != '.mm':
        raise ValueError("The file {} is not a Matrix Market file.".format(filepath))
    return MmCorpus(filepath)



def read_token2id(filepath):
    """Reads a token2id dictionary from CSV file.

    With this function you can read a CSV-file containing a document or type dictionary.

    Args:
        filepath (str): Path to CSV file.

    Returns:
        A dictionary.
    
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
        ...     tmpfile.write(b"0,this\\n1,is\\n2,an\\n3,example") and True
        ...     tmpfile.flush()
        ...     read_token2id(tmpfile.name)
        True
        {0: 'this', 1: 'is', 2: 'an', 3: 'example'}
    """
    dictionary = pd.read_csv(filepath, header=None)
    dictionary.index = dictionary[0]
    dictionary = dictionary[1]
    return dictionary.to_dict()
