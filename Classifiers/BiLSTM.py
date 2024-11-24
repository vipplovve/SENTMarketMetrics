import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing, metrics
import tensorflow as tf

os.system('cls')

def Partition(dataframe):
    Features = dataframe.drop(columns=['Verdict_negative', 'Verdict_neutral', 'Verdict_positive']).values
    Output = dataframe[['Verdict_negative', 'Verdict_neutral', 'Verdict_positive']].values
    Scaler = sklearn.preprocessing.StandardScaler()
    Features = Scaler.fit_transform(Features)
    return Features, Output

def ReshapeForLSTM(features):
    return features.reshape(features.shape[0], 1, features.shape[1])

def LossPlot(History):
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(History.history['loss'], label='Training Loss')
    Axis.plot(History.history['val_loss'], label='Validation Loss')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('Loss')
    Axis.grid(True)
    plt.legend()
    plt.show()

def PlotErrorsPerEpoch(error_log):
    plt.figure(figsize=(10, 6))
    for label, values in error_log.items():
        if label != "epoch":
            plt.plot(error_log["epoch"], values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Error Count")
    plt.title("False Positives and Negatives per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

FileName = input("Enter the name of the DataSet: ")
Categorials = input("Enter the name of All Categorical Variables & The Target Variable: ").split()
Dataset = pd.read_csv(FileName)
Dataset = pd.get_dummies(Dataset, columns=Categorials)

Train, Valid, Test = np.split(Dataset.sample(frac=1), [int(.6*len(Dataset)), int(.8*len(Dataset))])

TrainF, TrainO = Partition(Train)
ValidF, ValidO = Partition(Valid)
TestF, TestO = Partition(Test)

TrainF = ReshapeForLSTM(TrainF)
ValidF = ReshapeForLSTM(ValidF)
TestF = ReshapeForLSTM(TestF)

neurons = int(input("Enter the number of neurons for LSTM layers: "))
dropout_probability = float(input("Enter the dropout probability (between 0.2 and 0.5): "))
epochs = int(input("Enter the number of epochs for training: "))

inputs = tf.keras.Input(shape=(TrainF.shape[1], TrainF.shape[2]))
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons, return_sequences=True))(inputs)
x = tf.keras.layers.Dropout(dropout_probability)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons))(x)
x = tf.keras.layers.Dropout(dropout_probability)(x)
x = tf.keras.layers.Dense(neurons, activation='relu')(x)
x = tf.keras.layers.Dropout(dropout_probability)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

error_log = {"epoch": [], "False Positives (Negative)": [], "False Positives (Neutral)": [], 
             "False Positives (Positive)": [], "False Negatives (Negative)": [], 
             "False Negatives (Neutral)": [], "False Negatives (Positive)": [],
             "Training Accuracy": [], "Validation Accuracy": []}

for epoch in range(epochs):
    print(f"\nTraining Epoch {epoch + 1}/{epochs}")
    history = model.fit(TrainF, TrainO, validation_data=(ValidF, ValidO), epochs=1, verbose=0)

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    val_predictions = model.predict(ValidF)
    val_pred_labels = np.argmax(val_predictions, axis=1)
    val_true_labels = np.argmax(ValidO, axis=1)

    cm = metrics.confusion_matrix(val_true_labels, val_pred_labels, labels=[0, 1, 2])
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)

    error_log["epoch"].append(epoch + 1)
    error_log["False Positives (Negative)"].append(fp[0])
    error_log["False Positives (Neutral)"].append(fp[1])
    error_log["False Positives (Positive)"].append(fp[2])
    error_log["False Negatives (Negative)"].append(fn[0])
    error_log["False Negatives (Neutral)"].append(fn[1])
    error_log["False Negatives (Positive)"].append(fn[2])
    error_log["Training Accuracy"].append(train_acc)
    error_log["Validation Accuracy"].append(val_acc)

    print(f"False Positives: Negative={fp[0]}, Neutral={fp[1]}, Positive={fp[2]}")
    print(f"False Negatives: Negative={fn[0]}, Neutral={fn[1]}, Positive={fn[2]}")

error_log_df = pd.DataFrame(error_log)
error_log_df.to_csv("False.csv", index=False)
print("\nError log saved to 'ErrorsBiLSTM.csv'.")

PlotErrorsPerEpoch(error_log)

plt.figure(figsize=(10, 6))
plt.plot(error_log["epoch"], error_log["Training Accuracy"], label="Training Accuracy", color="green")
plt.plot(error_log["epoch"], error_log["Validation Accuracy"], label="Validation Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

Predictions = model.predict(TestF)
Predictions = np.argmax(Predictions, axis=1)
TestO_labels = np.argmax(TestO, axis=1)

cm = metrics.confusion_matrix(TestO_labels, Predictions, labels=[0, 1, 2])
labels = ["Negative", "Neutral", "Positive"]
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

Report = metrics.classification_report(TestO_labels, Predictions, target_names=labels)
print("\nClassification Report using the BiLSTM Model: -\n", Report)

Accuracy = metrics.accuracy_score(TestO_labels, Predictions)
print("\nAccuracy Score For This BiLSTM Model is: ", Accuracy * 100, "%")

save_model = input("\nDo You Wish To Save This Model? Enter 'Yes' or 'No': ")
if save_model.lower() == 'yes':
    model.save('Manual_BiLSTM_Model.keras')
    print("\nModel Saved Successfully!")
else:
    print("\nModel Not Saved!")
