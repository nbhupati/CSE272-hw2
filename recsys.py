import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Load data
dataframe = pd.read_json('Magazine_Subscriptions.json', lines=True)

# Drop unnecessary columns
dataframe = dataframe.drop(columns=['reviewTime', 'reviewText', 'verified', 'style', 'vote', 'reviewerName', 'unixReviewTime', 'summary'])

# Split data into training and testing sets
train_data, test_data = train_test_split(dataframe, test_size=0.2, random_state=1)


# Prepare training data
train_piv = train_data.pivot_table(values='overall', index='reviewerID', columns='asin').fillna(0)

# Generate pivot table
pivot_table = train_piv.apply(np.sign)

item_sim = np.dot(pivot_table.T.values, pivot_table.T.values.T)

pred = np.random.rand(*pivot_table.T.values.shape)
learning_rate = 0.001
num_iter = 5

for _ in range(num_iter):
    pred += learning_rate * (item_sim.T.dot(pivot_table.T.values) / np.array([np.abs(item_sim.T).sum(axis=1)]).T)

known = pivot_table.T.values

# Evaluation functions
def mae(prediction, known):
    prediction = prediction[known.nonzero()].flatten()
    known = known[known.nonzero()].flatten()
    return mean_absolute_error(prediction, known)

def rmse(prediction, known):
    prediction = prediction[known.nonzero()].flatten()
    known = known[known.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, known))

# Calculate scores
mae_score = mae(pred, known)
rmse_score = rmse(pred, known)

# Print results
print("Mean Absolute Error:", mae_score)
print("Root Mean Square Error:", rmse_score)