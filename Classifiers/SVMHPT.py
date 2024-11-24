import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plotter
import os

from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

os.system('cls')

def Partition(dataframe):
    Features = dataframe.drop(columns=['Verdict_negative', 'Verdict_neutral', 'Verdict_positive']).values
    Output = dataframe[['Verdict_negative', 'Verdict_neutral', 'Verdict_positive']].values
    Output = np.argmax(Output, axis=1)
    return Features, Output

def PlotConfusionMatrix(y_true, y_pred, labels):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    plotter.figure(figsize=(8, 6))
    plotter.imshow(cm, interpolation='nearest', cmap='Blues')
    plotter.colorbar()
    tick_marks = np.arange(len(labels))
    plotter.xticks(tick_marks, labels, rotation=45)
    plotter.yticks(tick_marks, labels)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plotter.text(j, i, cm[i, j], horizontalalignment="center", color="black")

    plotter.xlabel('Predicted Label')
    plotter.ylabel('True Label')
    plotter.title('Confusion Matrix')
    plotter.tight_layout()
    plotter.show()

def VisualizeFalsePosNeg(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    labels = [f"Class {i}" for i in range(len(fp))]
    
    plotter.figure(figsize=(10, 5))
    bar_width = 0.4
    index = np.arange(len(labels))

    plotter.bar(index, fp, bar_width, color='red', label='False Positives')
    plotter.bar(index + bar_width, fn, bar_width, color='blue', label='False Negatives')

    plotter.xlabel('Classes')
    plotter.ylabel('Count')
    plotter.xticks(index + bar_width / 2, labels)
    plotter.title('False Positives and False Negatives by Class')
    plotter.legend()
    plotter.tight_layout()
    plotter.show()

FileName = input("Enter the name of the DataSet: ")
print()
Categorials = input("Enter the name of All Categorial Variables & The Target Variable: ").split()
print()

Dataset = pd.read_csv(FileName)
Dataset = pd.get_dummies(Dataset, columns=Categorials)

Features, Output = Partition(Dataset)
X_train, X_temp, y_train, y_temp = train_test_split(Features, Output, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

os.system('cls')

print("The Dataset: -\n")
print(Dataset.head(), '\n')

grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']  
}

svm = SVC(probability=True)
grid_search = GridSearchCV(svm, grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
print("Best Estimator:", grid_search.best_estimator_)
print()

print("Creating an SVM Model: -\n")
C = float(input("Enter the C parameter for SVM: "))
kernel = input("Enter the kernel type (linear, poly, rbf, sigmoid): ")
gamma = input("Enter the gamma (scale, auto): ")
print()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

SVMModel = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
SVMModel.fit(X_train, y_train)

Predictions = SVMModel.predict(X_test)
Predictions_proba = SVMModel.predict_proba(X_test)

print("\nPredictions: -\n")
Report = sklearn.metrics.classification_report(y_test, Predictions)
print("\nClassification Report using an SVM Model: -\n")
print(Report)

Accuracy = sklearn.metrics.accuracy_score(y_test, Predictions)
f1 = sklearn.metrics.f1_score(y_test, Predictions, average='weighted')

print("Specifications For The Model: \n")
print(f"C parameter: {C}")
print(f"Kernel type: {kernel}")
print("\nAccuracy Score For This SVM Model is: ", Accuracy * 100, "%", '\n')
print("F1 Score For This SVM Model is: ", f1, '\n')

print("Visualizing Confusion Matrix:")
PlotConfusionMatrix(y_test, Predictions, labels=[f"Class {i}" for i in range(len(np.unique(y_test)))])

print("Visualizing False Positives and False Negatives:")
VisualizeFalsePosNeg(y_test, Predictions)

print("Do You Wish To Save This Model?")
input = input("Enter 'Yes' or 'No': ")

if input == 'Yes':
    import joblib
    joblib.dump(SVMModel, "SVMModel.pkl")
    print("\nModel Saved Successfully!")
else:
    print("\nModel Not Saved!")
