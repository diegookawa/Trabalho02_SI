import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

def main():
    df = pd.read_csv("tar2_sinais_vitais_treino_com_label.txt")
    df = df.dropna()

    X_regression = df.drop(["id", "pSist", "pDiast", "grav", "risco"], axis=1)
    y_regression = df["grav"]

    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=44)

    model_regression = DecisionTreeRegressor(random_state=44, max_depth=8)

    X_classification = df.drop(["id", "pSist", "pDiast", "risco"], axis=1)
    y_classification = df["risco"]

    X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
        X_classification, y_classification, test_size=0.2, random_state=44)

    model_classification = tree.DecisionTreeClassifier(criterion='gini', 
                                            splitter='best', 
                                            max_depth=8,
                                            class_weight=None,
                                            min_samples_leaf=200, 
                                            random_state=44, 
                                    )

    model_classification.fit(X_train_classification, y_train_classification)
    predictions_train_classification = model_classification.predict(X_train_classification)
    predictions_test_classification = model_classification.predict(X_test_classification)

    model_regression.fit(X_train, y_train)
    predictions_train_regression = model_regression.predict(X_train)
    predictions_test_regression = model_regression.predict(X_test)
    rsme_train = mean_squared_error(y_train, predictions_train_regression, squared=False)
    rsme_test = mean_squared_error(y_test, predictions_test_regression, squared=False)

    printMetricsClassification(y_test_classification, predictions_test_classification, y_train_classification, predictions_train_classification)
    printMetricsRegression(rsme_train, rsme_test)
    np.random.seed(8)
    clf_classification = MLPClassifier(
                                        hidden_layer_sizes=(256,128,64),max_iter=50000, learning_rate_init=0.001).fit(X_train_classification, y_train_classification)
    predictions_train_classification = clf_classification.predict(X_train_classification)
    predictions_test_classification = clf_classification.predict(X_test_classification)
    np.random.seed(8)
    clf_regression = MLPRegressor(
                                    hidden_layer_sizes=(256,128,64),max_iter=50000, learning_rate_init=0.001, activation="relu",tol=0.0001, n_iter_no_change=50).fit(X_train, y_train)
    predictions_train_regression = clf_regression.predict(X_train)
    predictions_test_regression = clf_regression.predict(X_test)
    rsme_train = mean_squared_error(y_train, predictions_train_regression, squared=False)
    rsme_test = mean_squared_error(y_test, predictions_test_regression, squared=False)

    printMetricsClassification(y_test_classification, predictions_test_classification, y_train_classification, predictions_train_classification)
    printMetricsRegression(rsme_train, rsme_test)
    

def printMetricsRegression(rsme_train, rsme_test):
    print("REGRESSION USING REGRESSION TREE\n")
    print('*************** Evaluation on Training Data ***************')
    print('RSME: ', rsme_train)
    print('--------------------------------------------------------')
    print("")
    print('*************** Evaluation on Test Data ***************')
    print('RSME: ', rsme_test)
    print('--------------------------------------------------------')

def printMetricsClassification(y_test_classification, predictions_test_classification, y_train_classification, predictions_train_classification):
    print("CLASSIFICATION USING CLASSIFICATION TREE\n")
    print('*************** Evaluation on Training Data ***************')
    print(classification_report(y_train_classification, predictions_train_classification, zero_division=0))
    print(f"Confusion matrix:\n {confusion_matrix(y_train_classification, predictions_train_classification)}")
    print('--------------------------------------------------------')
    print("")
    print('*************** Evaluation on Test Data ***************')
    print(classification_report(y_test_classification, predictions_test_classification, zero_division=0))
    print(f"Confusion matrix:\n {confusion_matrix(y_test_classification, predictions_test_classification)}")
    print('--------------------------------------------------------')
    
if __name__ == "__main__":
    main()
