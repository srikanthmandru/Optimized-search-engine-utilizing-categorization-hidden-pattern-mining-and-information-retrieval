### This is type of content-based recommender system

### Steps:

# - represent the documents in form of vectors

# - find the cosine similarity between the documents and form a similarity matrix

# - prepare the document-term matrix (indexing) for fast access

# - get the most similar documents 


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

## Load the data
dataframe = pd.read_csv('./models/output/results_dataframe.csv')

# Cast as a list of values for calculating weights
text_data = dataframe['processed_text'].values.tolist()


# Calculate TF-IDF matrix
def tf_idf(search_keys, data):
    ## Load from the saved vectorizer later
    ## TFIDF vectorize the data
    tf_idf_vectorizer = pickle.load(open('./models/output/tf_idf_vectorizer.pkl', 'rb'))
    tfidf_weights_matrix = tf_idf_vectorizer.transform(data).toarray()
    search_query_weights = tf_idf_vectorizer.transform([search_keys]).toarray()
    

    ## Dimension reduction of data
    pca_reduction_model = pickle.load(open('./models/output/pca_method.pkl', 'rb'))

    dimension_reduced_query = pca_reduction_model.transform(search_query_weights)
    dimension_reduced_data = pca_reduction_model.transform(tfidf_weights_matrix)

    return dimension_reduced_query, dimension_reduced_data

# Calculate the cosine similarity between search query and TF-IDF vectors
def cos_similarity(search_query_weights, tfidf_weights_matrix):
    cosine_sim = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    similarity_list = cosine_sim[0]

    return similarity_list

# Calculate number of relevant vectors
def calculate_num_vectors(cosine_similarity):

    num = 0
    for i in cosine_similarity:
        if i != 0.0:
            num += 1
    return num

# Calculate the most relevant vectors
def most_similar(similarity_list, N):

    most_similar = []

    while N > 0:
        tmp_index = np.argmax(similarity_list)
        most_similar.append(tmp_index)
        similarity_list[tmp_index] = 0
        N -= 1

    return most_similar

# Create weights at specific index for quick retrieval
def create_matrix_dict(cosine_similarity):

    matrix_dict = {}

    iter_counter = 0
    for i in cosine_similarity:
        matrix_dict[iter_counter] = i
        iter_counter += 1

    return matrix_dict

# -----------
# Return the articles with relevant search term
def return_relevant_articles(search_term, cluster_dataframe = None):

    # Create local variables
    # convert_documents to vector representations
    if (cluster_dataframe.shape[0] != 0 ):
        cluster_text_data = cluster_dataframe['processed_text'].values.tolist()
        search, matrix = tf_idf(search_term, cluster_text_data)
        dataframe_copy = cluster_dataframe
    else:
        search, matrix = tf_idf(search_term, text_data)
        dataframe_copy = dataframe
    
    # Find the cosine similarity
    cosine_sim_list = cos_similarity(search, matrix)
    
    # Get the number of relevant documents
    num_relevant_vectors = calculate_num_vectors(cosine_sim_list)
    
    # Prepare the " indexing " (one of stages in web information retrieval) for faster retrieval 
    # (Similar concept is also used by the Google, namely stored as document-term matrix)
    dictionary = create_matrix_dict(cosine_sim_list)
    
    # Get the most similar items
    list_of_most_similar = most_similar(cosine_sim_list, num_relevant_vectors)

    df = pd.DataFrame()

    for index in list_of_most_similar:

        article = dataframe_copy.iloc[index]

        if df.empty:

            to_dataframe = article.to_frame()
            df = to_dataframe.T

        else:
            to_dataframe = article.to_frame()
            df = pd.concat([df, to_dataframe.T], join='outer')

    ### Specify the required columns here
    columns = dataframe_copy.columns

    return df[columns]