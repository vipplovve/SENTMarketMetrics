import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plotter
import os
from sklearn import preprocessing, metrics
import tensorflow as tf
import keras_tuner as kt 

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
    Figure, Axis = plotter.subplots(1, 1)
    Axis.plot(History.history['loss'], label='Training Loss')
    Axis.plot(History.history['val_loss'], label='Validation Loss')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('Loss')
    Axis.grid(True)
    plotter.legend()
    plotter.show()

def AccuracyPlot(History):
    Figure, Axis = plotter.subplots(1, 1)
    Axis.plot(History.history['accuracy'], label='Training Accuracy')
    Axis.plot(History.history['val_accuracy'], label='Validation Accuracy')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('Accuracy')
    Axis.grid(True)
    plotter.legend()
    plotter.show()

def PlotConfusionMatrix(y_true, y_pred, labels):
    cm = metrics.confusion_matrix(y_true, y_pred)
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
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            plotter.text(j, i, f'{cm[i, j]}', ha="center", va="center", color="black", fontsize=12)
    
    plotter.tight_layout()
    plotter.show()

def VisualizeFalsePosNeg(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
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

def F1ScorePlot(History):
    Figure, Axis = plotter.subplots(1, 1)
    f1_train = 2 * (np.array(History.history['accuracy']) * (1 - np.array(History.history['loss']))) / \
               (np.array(History.history['accuracy']) + (1 - np.array(History.history['loss'])))
    f1_val = 2 * (np.array(History.history['val_accuracy']) * (1 - np.array(History.history['val_loss']))) / \
             (np.array(History.history['val_accuracy']) + (1 - np.array(History.history['val_loss'])))
    Axis.plot(f1_train, label='Training F1 Score')
    Axis.plot(f1_val, label='Validation F1 Score')
    Axis.set_xlabel('Epoch')
    Axis.set_ylabel('F1 Score')
    Axis.grid(True)
    plotter.legend()
    plotter.show()

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

def build_model(hp):
    neurons = hp.Int('neurons', min_value=32, max_value=256, step=32)
    dropout_probability = hp.Float('dropout_probability', min_value=0.2, max_value=0.5, step=0.1)
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
    return model

tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=128, factor=3, directory='tuner_dir', project_name='bi_lstm_tuning')
tuner.search(TrainF, TrainO, epochs=64, validation_data=(ValidF, ValidO))

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.hypermodel.build(best_hyperparameters)
History = best_model.fit(TrainF, TrainO, epochs=64, validation_data=(ValidF, ValidO))

LossPlot(History)
AccuracyPlot(History)
F1ScorePlot(History)

Predictions = best_model.predict(TestF)
Predictions = np.argmax(Predictions, axis=1)
TestO_labels = np.argmax(TestO, axis=1)

PlotConfusionMatrix(TestO_labels, Predictions, labels=[f"Class {i}" for i in range(len(np.unique(TestO_labels)))])
VisualizeFalsePosNeg(TestO_labels, Predictions)

Report = metrics.classification_report(TestO_labels, Predictions)
print("\nClassification Report using the Tuned BiLSTM Model: -\n", Report)

Accuracy = metrics.accuracy_score(TestO_labels, Predictions)
print("\nAccuracy Score For This Tuned BiLSTM Model is: ", Accuracy * 100, "%")

save_model = input("\nDo You Wish To Save This Model? Enter 'Yes' or 'No': ")
if save_model.lower() == 'yes':
    best_model.save('Tuned_BiLSTM_Model.keras')
    print("\nModel Saved Successfully!")
else:
    print("\nModel Not Saved!")
