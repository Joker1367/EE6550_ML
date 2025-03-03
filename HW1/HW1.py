import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    
    likelihood = {}
    for cls in classes:
        pdf = []
        class_data = train_data[train_data.iloc[:, 0] == cls][features]
        for _, col in class_data.items():
            mean = np.mean(col)
            std  = np.std(col)
            pdf.append(norm(mean, std).pdf)

        likelihood[cls] = pdf

    return likelihood

def MAP_classifier(prior, likelihood, test_data):
    features = test_data.columns[1:]
    prediction = []

    for idx, row in test_data.iterrows():
        x_vec = row[features].values
        class_prob = {}

        for cls, pdfs in likelihood.items():
            prior_prob = prior[cls]
            likelihood_prob = 1
            for idx, pdf in enumerate(pdfs):
                likelihood_prob = likelihood_prob * pdf(x_vec[idx])

            posterior_prob = likelihood_prob * prior_prob
            class_prob[cls] = posterior_prob
            #print(f'prior = {prior_prob}, likelihood = {likelihood_prob}, posterior = {posterior_prob}')

        prediction.append(max(class_prob, key = class_prob.get)) 

    return prediction

def ML_classifier(likelihood, test_data):
    features = test_data.columns[1:]
    prediction = []

    for idx, row in test_data.iterrows():
        x_vec = row[features].values
        class_prob = {}

        for cls, pdfs in likelihood.items():
            #prior_prob = prior[cls]
            likelihood_prob = 1
            for idx, pdf in enumerate(pdfs):
                likelihood_prob = likelihood_prob * pdf(x_vec[idx])

            posterior_prob = likelihood_prob
            class_prob[cls] = posterior_prob
            #print(f'prior = {prior_prob}, likelihood = {likelihood_prob}, posterior = {posterior_prob}')

        prediction.append(max(class_prob, key = class_prob.get))
    
    return prediction

def evaluate(test_data, prediction):
    answer = test_data.iloc[:, 0]
    accuracy = np.mean(answer == prediction)

    cm = confusion_matrix(answer, prediction)

    return accuracy, cm

def PCA_visualize(test_data):
    features_idx = test_data.columns[1:]
    X_test = test_data[features_idx].values
    y_test = test_data.iloc[:, 0].values

    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_test)

    # Convert PCA result into a DataFrame
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Label'] = y_test  # Add labels for coloring

    plt.figure(figsize=(8, 6))
    for label in np.unique(y_test):
        subset = pca_df[pca_df['Label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'], label=f'Class {label}', alpha=0.7)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Test Data")
    plt.legend()
    plt.grid(True)
    plt.show()

'''''''''''''''''''''''''''''''''
        Main Program
'''''''''''''''''''''''''''''''''

# Read data & train / test split
df = pd.read_csv("wine.csv")

test_data = df.groupby(df.columns[0], group_keys = False).apply(lambda x: x.sample(n=20, random_state=1))
train_data = df.drop(test_data.index)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Compute prior & likelihood
prior = compute_prior(train_data)
likelihood = compute_likelihood(train_data)

# MAP / ML classifier
prediction_MAP = MAP_classifier(prior, likelihood, test_data)
prediction_ML = ML_classifier(likelihood, test_data)

# Evaluation
accuracy_MAP, cm_MAP = evaluate(test_data, prediction_MAP)
accuracy_ML, cm_ML = evaluate(test_data, prediction_ML)

plt.figure(1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_MAP)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix (classifier = MAP, acc = {accuracy_MAP})')

plt.figure(2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_ML)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix (classifier = ML, acc = {accuracy_ML})')

plt.show()

# PCA visualization of test data
PCA_visualize(test_data)



