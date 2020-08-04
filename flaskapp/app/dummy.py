import pandas as pd
import numpy as np

## Load the data
dataframe = pd.read_csv('../models/output/results_dataframe.csv')

dummy_array = np.array([10, 56])
# print((dataframe["y"] == 56).values)

print(dataframe[(dataframe["y"] == dummy_array[0]).values])