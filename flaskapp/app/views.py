
""" @author: srikanthmandru """

from app import app 
from flask import Flask, render_template, jsonify, request
import os
import pickle , json
import numpy as np
import sklearn
import pandas as pd

from app.search_engine import return_relevant_recipes

## Load the model while initializing
model = pickle.load(open('./models/output/spectral_model.pkl', 'rb'))


#################################################
# Flask Routes
#################################################

@app.route("/")
def welcome():
    return render_template('index.html')

@app.route("/api/search/<query>")
def search_query(query=None):

    try:

        results = return_relevant_recipes(query)
        
        # df_to_dict = results_subset.to_dict('r')
        # data = json.dumps(df_to_dict, ensure_ascii=False, indent=4)

        data = results.to_json(orient='records')

        return (
            data
        )

    except Exception as e:
        return (
            f"{e}"
    )

    
    
    