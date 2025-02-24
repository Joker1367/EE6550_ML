import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

'''''''''''''''''''''''''''''''''
        Functions
'''''''''''''''''''''''''''''''''

def compute_prior(train_data):
    labels = train_data.iloc[:, 0]
    prior = labels.value_counts(normalize = True)

    return prior.to_dict()

def compute_likelihood(train_data):
    features = train_data.columns[1:]
    classes = train_data.iloc[:, 0].unique()

    
    params = {}
    for cls in classes:
        class_data = train_data[train_data.iloc[:, 0] == cls][features]
        mean_vec = class_data.mean().values
        covariance_matrix = class_data.cov().values

        #print("mean vector = ", mean_vec)
        #print("covariance matrix = ", covariance_matrix)

        params[cls] = (mean_vec, covariance_matrix)

    return params

def MAP_classifier(prior, likelihood, test_data):
    features = test_data.columns[1:]
    prediction = []

    for index, row in test_data.iterrows():
        x_vec = row[features].values
        class_prob = {}

        for cls, (mean_vec, covariance_matrix) in likelihood.items():
            prior_prob = prior[cls]
            likelihood_prob = multivariate_normal.pdf(x_vec, mean = mean_vec, cov = covariance_matrix)
            posterior_prob = likelihood_prob * prior_prob
            class_prob[cls] = posterior_prob

        #print(f"probability class for test case {index} is {class_prob}")
        prediction.append(max(class_prob, key = class_prob.get)) 

    return prediction

def ML_classifier()

def evaluate(test_data, prediction):
    answer = test_data.iloc[:, 0]
    accuracy = np.mean(answer == prediction)

    return accuracy

'''''''''''''''''''''''''''''''''
        Main Program
'''''''''''''''''''''''''''''''''

# Read data & train / test split
df = pd.read_csv("wine.csv")

test_data = df.groupby(df.columns[0], group_keys = False).apply(lambda x: x.sample(n=20, random_state=69))
train_data = df.drop(test_data.index)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Compute prior & likelihood
prior = compute_prior(train_data)
likelihood = compute_likelihood(train_data)

# MAP classifier
prediction = MAP_classifier(prior, likelihood, test_data)

# Evaluation
accuracy = evaluate(test_data, prediction)
print("accuracy = ", accuracy)

