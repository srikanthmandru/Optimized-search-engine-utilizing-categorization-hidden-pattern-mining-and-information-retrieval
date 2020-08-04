
""" @author: srikanthmandru """

from app import app 
from flask import Flask, render_template, jsonify, request
import os
import pickle , json
import numpy as np
import sklearn
import pandas as pd

from app.search_engine import return_relevant_articles
from app.clustering import cluster_data


#################################################
# Flask Routes
#################################################

@app.route("/")
def welcome():
    return "API home page!!!"

@app.route("/api/search/<query>")
def search_query(query=None):

    try:
        # Send the query term and get cluster data from the clustering model
        cluster_dataframe = cluster_data(query)

        # Pass the cluster data to get the similar data points
        results = return_relevant_articles(query, cluster_dataframe= cluster_dataframe)
        
        # print('results dataframe is :' ,results)
        
        ### Work around with the json convertion
        # df_to_dict = results_subset.to_dict('r')
        # data = json.dumps(df_to_dict, ensure_ascii=False, indent=4)

        ### Convert the results to json for serialization data representation
        data = results.to_json(orient='records')

        return (
            data
        )

    except Exception as e:
        return (
            f"{e}"
    )

    
    
    