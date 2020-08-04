import os
import pickle , json
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import pandas as pd
from app.search_engine import text_data, dataframe

def preprocess_data(search_keys, data):

    ## TFIDF vectorize the data
    tf_idf_vectorizer = pickle.load(open('./models/output/tf_idf_vectorizer.pkl', 'rb'))
    vectorized_query = tf_idf_vectorizer.transform([search_keys]).toarray()
    vectorized_data = tf_idf_vectorizer.transform(data).toarray()

    ## Dimension reduction of data
    pca_reduction_model = pickle.load(open('./models/output/pca_method.pkl', 'rb'))

    dimension_reduced_query = pca_reduction_model.transform(vectorized_query)
    dimension_reduced_data = pca_reduction_model.transform(vectorized_data)

    return dimension_reduced_query, dimension_reduced_data

# Calculate TF-IDF matrix
def tf_idf_cluster(search_keys, data):
    ## Load from the saved vectorizer later
    tfidf_vectorizer = TfidfVectorizer(stop_words={"english"})
    tfidf_weights_matrix = tfidf_vectorizer.fit_transform(data)
    search_query_weights = tfidf_vectorizer.transform([search_keys])

    return search_query_weights, tfidf_weights_matrix

def cluster_data(query):
    ## Load the model while initializing
    cluster_model = pickle.load(open('./models/output/spectral_model.pkl', 'rb'))

    # print("cluster model is :{}" .format(cluster_model))

    ## Prepare data for model fitting 
    # query_vector , _ = tf_idf_cluster(query, text_data)
    query_vector, data_vectors = preprocess_data(query, text_data)
    # print("query vector ", query_vector)
    # print("query vector shape ", query_vector.shape)

    ## Concatenate query vector and the text corpus vectors along row (since, spectral cluster doesn't have predict method)
    concatenated_data = np.concatenate((query_vector, data_vectors), axis = 0 )

    ## pass data to model and get the cluster number
    cluster_labels = cluster_model.fit_predict(concatenated_data)

    # print("cluster_label is : {}" .format(cluster_labels[0]))

    ## Send the data related to the predicted cluster
    cluster_dataframe = dataframe.loc[dataframe['y'] == cluster_labels[0]]
    
    # print(cluster_dataframe)
    # print("cluster data has {} records" .format(cluster_dataframe.shape))

    return cluster_dataframe

