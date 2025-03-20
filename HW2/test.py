import os
import random
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

random.seed(42)

def load_images_as_vectors(source_dir, img_size=(32, 32)):
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

# One-hot encoding 函數
def one_hot_encode(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_one_hot = one_hot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    return y_one_hot

# 分割訓練和驗證資料
def split_data(X, y, train_ratio=0.8):
    indices = list(range(len(y)))
    random.shuffle(indices)  # 隨機打亂
    
    train_count = int(len(y) * train_ratio)
    train_idx, val_idx = indices[:train_count], indices[train_count:]
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# ReLU 激活函數
def relu(x):
    return np.maximum(0, x)

# ReLU 對 x 的導數
def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax 激活函數
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止指數爆炸
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)  # 固定隨機種子，確保結果一致
    W1 = np.random.randn(input_size, hidden_size) * 0.01  # 隱藏層權重
    b1 = np.zeros((1, hidden_size))  # 隱藏層偏置
    W2 = np.random.randn(hidden_size, output_size) * 0.01  # 輸出層權重
    b2 = np.zeros((1, output_size))  # 輸出層偏置
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  # 計算隱藏層加權輸入
    A1 = relu(Z1)  # ReLU 激活函數
    Z2 = np.dot(A1, W2) + b2  # 計算輸出層加權輸入
    A2 = softmax(Z2)  # Softmax 輸出機率
    return Z1, A1, Z2, A2


def compute_loss(y_true, y_pred):
    m = y_true.shape[0]  # 样本数
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # 避免 log(0)
    return loss


def backward_propagation(X, y_true, Z1, A1, A2, W2):
    m = X.shape[0]

    # 計算梯度
    dZ2 = A2 - y_true  # 輸出層誤差
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)  # 反向傳遞到隱藏層
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    print("dZ1 = ", dZ1)
    print("dZ2 = ", dZ2)

    return dW1, db1, dW2, db2


def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr=0.01):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


def train_neural_network(X_train, y_train, hidden_size=10, epochs=1000, lr=0.01):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]  # One-hot encoding 的大小
    
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # 前向傳播
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        
        # 計算損失
        loss = compute_loss(y_train, A2)

        # 反向傳播
        dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Z1, A1, A2, W2)

        # 更新權重
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        # 每 100 次迭代輸出一次損失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    print("A1 = ", A1)
    print("A2 = ", A2)
    print("W1 = ", W1)
    print("b1 = ", b1)
    print("W2 = ", W2)
    print("b2 = ", b2)

    return W1, b1, W2, b2

def eval(predict, answer):
    correct_count = sum(p == a for p, a in zip(predict, answer))
    accuracy = correct_count / len(answer)
    return accuracy

# PCA 降維並分割訓練/驗證資料
source_directory = "Data/Data_train"
X, y = load_images_as_vectors(source_directory)

# 將 y 轉換為 one-hot encoding
y_one_hot = one_hot_encode(y)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train, y_train, X_val, y_val = split_data(X_pca, y_one_hot)

# 假設 y_train 已經轉換為 One-Hot Encoding
W1, b1, W2, b2 = train_neural_network(X_train, y_train, hidden_size=10, epochs=1, lr=0.007)

# 測試模型
_, _, _, predictions = forward_propagation(X_val, W1, b1, W2, b2)
predicted_labels = np.argmax(predictions, axis=1)
answer_labels = np.argmax(y_val, axis=1)

accuracy = eval(predicted_labels, answer_labels)

print("accuracy = ", accuracy)



