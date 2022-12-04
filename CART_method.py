import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("tar2_sinais_vitais_treino_com_label.txt")
df = df.dropna()

X = df.drop(["id", "pSist", "pDiast", "grav", "risco"], axis=1)
y = df["grav"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

model = DecisionTreeRegressor(random_state=44, max_depth=8)

model.fit(X_train, y_train)
predictions_test = model.predict(X_test)
predictions_train = model.predict(X_train)
rsme_train = mean_squared_error(y_train, predictions_train, squared=False)
rsme_test = mean_squared_error(y_test, predictions_test, squared=False)

print("REGRESSION USING REGRESSION TREE\n")
print('\t*************** Evaluation on Training Data ***************')
print('\tRSME: ', rsme_train)
# Look at classification report to evaluate the model
print('\t--------------------------------------------------------')
print("")

print('\t*************** Evaluation on Test Data ***************')
print('\tRSME: ', rsme_test)
# Look at classification report to evaluate the model
print('\t--------------------------------------------------------')

    
