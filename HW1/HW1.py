import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


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

def Visualize_PCA(train, test):

    features_idx = train.columns[1:]
    train_data = train[features_idx].values
    train_labels = train.iloc[:, 0].values
    test_data = test[features_idx].values
    test_labels = test.iloc[:, 0].values

    # normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)
    X_test_scaled = scaler.transform(test_data)

    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
    X_test_pca_2d = pca_2d.transform(X_test_scaled)

    # 3D PCA
    pca_3d = PCA(n_components=3)
    X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)
    X_test_pca_3d = pca_3d.transform(X_test_scaled)

    # === 2D ===
    plt.figure(figsize=(10, 5))
    
    # train
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=train_labels, cmap='viridis', alpha=0.6)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization (2D) - Train Data")

    # test
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_pca_2d[:, 0], X_test_pca_2d[:, 1], c=test_labels, cmap='viridis', alpha=0.6)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization (2D) - Test Data")
    
    plt.show()

    # === 3D ===
    fig = plt.figure(figsize=(10, 5))

    # train
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_train_pca_3d[:, 0], X_train_pca_3d[:, 1], X_train_pca_3d[:, 2], c=train_labels, cmap='viridis', alpha=0.6)
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.set_zlabel("PCA 3")
    ax1.set_title("PCA Visualization (3D) - Train Data")

    # test
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_test_pca_3d[:, 0], X_test_pca_3d[:, 1], X_test_pca_3d[:, 2], c=test_labels, cmap='viridis', alpha=0.6)
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_zlabel("PCA 3")
    ax2.set_title("PCA Visualization (3D) - Test Data")

    plt.show()

def evaluate_feature_contributions(prior, likelihood, test_data):
    answer = test_data.iloc[:, 0]
    features = test_data.columns[1:]  
    feature_contributions = {}  

    for removed_feature in features:
        prediction = []
        reduced_features = [f for f in features if f != removed_feature]  

        for _, row in test_data.iterrows():
            x_vec = row[reduced_features].values
            class_prob = {}

            for cls, pdfs in likelihood.items():
                prior_prob = prior[cls]
                reduced_pdfs = [pdf for i, pdf in enumerate(pdfs) if features[i] != removed_feature]
                likelihood_prob = np.prod([pdf(x) for pdf, x in zip(reduced_pdfs, x_vec)])
                posterior_prob = likelihood_prob * prior_prob
                class_prob[cls] = posterior_prob

            prediction.append(max(class_prob, key=class_prob.get))

        feature_contributions[removed_feature] = np.mean(np.array(answer) == np.array(prediction))

    return feature_contributions

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

# Data visualization
Visualize_PCA(train_data, test_data)

# Feature contribution
feature_contribution = evaluate_feature_contributions(prior, likelihood, test_data)
sorted_contributions = sorted(feature_contribution.items(), key=lambda x: x[1])
print("Feature Contributions (sorted from most important to least important):")
for feature, contribution in sorted_contributions:
    print(f"{feature}: {contribution:.4f}")


