import pandas as pd

df = pd.read_csv("wine.csv")

test_data = df.groupby(df.columns[0], group_keys = False).apply(lambda x: x.sample(n=20, random_state=42))
train_data = df.drop(test_data.index)

print("Training Data Size:", train_data.shape)
print("Testing Data Size:", test_data.shape)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

label_counts = train_data.iloc[:, 0].value_counts()
prior = label_counts / train_data.shape[0]

print(prior)