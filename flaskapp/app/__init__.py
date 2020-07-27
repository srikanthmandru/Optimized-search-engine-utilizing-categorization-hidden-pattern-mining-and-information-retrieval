### Initialize the app 
from flask import Flask
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

from app import views
