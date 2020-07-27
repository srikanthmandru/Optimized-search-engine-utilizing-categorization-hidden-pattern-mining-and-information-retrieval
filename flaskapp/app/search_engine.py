### Steps:

# - represent the documents in form of vectors

# - find the cosine similarity between the documents

# - prepare the document-term matrix (indexing) for fast access

# - get the most similar documents 


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

## Load the data
dataframe = pd.read_csv('./models/output/results_dataframe.csv')

# Cast as a list of values for calculating weights
text_data= dataframe['processed_text'].values.tolist()

# Calculate TF-IDF matrix
def tf_idf(search_keys, data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_weights_matrix = tfidf_vectorizer.fit_transform(data)
    search_query_weights = tfidf_vectorizer.transform([search_keys])

    return search_query_weights, tfidf_weights_matrix

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
# Return the recipes with relevant search term
def return_relevant_recipes(search_term):

    # Create local variables
    # convert_documents to vector representations
    search, matrix = tf_idf(search_term, text_data)
    
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

        recipe = dataframe.iloc[index]

        if df.empty:

            to_dataframe = recipe.to_frame()
            df = to_dataframe.T

        else:
            to_dataframe = recipe.to_frame()
            df = pd.concat([df, to_dataframe.T], join='outer')

    ### Specify the required columns here
    columns = dataframe.columns

    return df[columns]