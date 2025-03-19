import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

random.seed(42)

'''''''''''''''
    Data Preprocessing
'''''''''''''''
def load_images_as_vectors(source_dir, img_size=(32, 32)):
    """讀取所有灰階圖片並轉換為向量"""
    data = []
    labels = []
    
    classes = os.listdir(source_dir)
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            img = Image.open(img_path).convert("L").resize(img_size)  # 轉換為灰階 (L mode)
            img_vector = np.array(img).flatten()  # 展平成 1D 向量
            
            data.append(img_vector)
            labels.append(cls)
    
    return np.array(data), np.array(labels)

def split_data(X, y, train_ratio=0.8):
    indices = list(range(len(y)))
    random.shuffle(indices)  # 隨機打亂
    
    train_count = int(len(y) * train_ratio)
    train_idx, val_idx = indices[:train_count], indices[train_count:]
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def one_hot_encode(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    one_hot_encoder = OneHotEncoder(sparse_output=False)  
    y_one_hot = one_hot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    return y_one_hot

'''
    Neural Network
'''
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.weights = []
        self.biases = []

        np.random.seed(42)
        
        self.weights.append(np.random.randn(self.input_size, self.hidden_sizes[0]))
        self.biases.append(np.zeros(self.hidden_sizes[0]))
        
        for i in range(1, len(self.hidden_sizes)):
            self.weights.append(np.random.randn(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.biases.append(np.zeros(self.hidden_sizes[i]))
        
        self.weights.append(np.random.randn(self.hidden_sizes[-1], self.output_size))
        self.biases.append(np.zeros(self.output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        x = np.clip(x, -500, 500)  # 將x限制在[-500, 500]範圍內
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                activation = self.softmax(z)
            else:
                activation = self.relu(z)
            self.activations.append(activation)
        
        return self.activations[-1]

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        output = self.activations[-1]
        output_error = output - y
        
        for i in range(len(self.weights)-1, -1, -1):
            if i == len(self.weights) - 1:
                delta = output_error
            else:
                delta = delta.dot(self.weights[i+1].T) * self.relu_derivative(self.activations[i+1] > 0)  
                
            self.weights[i] -= learning_rate * self.activations[i].T.dot(delta) / m
            self.biases[i] -= learning_rate * np.sum(delta, axis=0) / m

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train, learning_rate)
            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(y_train, output)
                print(f"Epoch {epoch}: Loss = {loss}")

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

def eval(predict, answer):
    correct_count = sum(p == a for p, a in zip(predict, answer))
    accuracy = correct_count / len(answer)
    return accuracy

train_directory = "Data/Data_train"
X, y = load_images_as_vectors(train_directory)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
y_one_hot = one_hot_encode(y)

X_train, y_train, X_val, y_val = split_data(X_pca, y_one_hot)

input_size = 2  
hidden_sizes = [10]  
output_size = 3

nn_2 = NeuralNetwork(input_size, hidden_sizes, output_size)
nn_2.train(X_train, y_train, epochs=1000, learning_rate=0.0003)

y_pred = nn_2.forward(X_val)
prediction = np.argmax(y_pred, axis = 1)
answer = np.argmax(y_val, axis = 1)

accuracy = eval(prediction, answer)

print("accuracy = ", accuracy)

