import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("possum.csv")
# print(df.head())
# print(df.info())

df = df.dropna()
# print(df.info())

X = df.drop(["case", "site", "Pop", "sex", "age"], axis=1)
y = df["age"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

model = DecisionTreeRegressor(random_state=44)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)
print(y_test)