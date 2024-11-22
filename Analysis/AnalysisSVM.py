import numpy as np
import pandas as pd
import joblib
import os

os.system('cls')

FileName = input("\nEnter Path of the SVM Model (e.g., model.pkl): ")
Model = joblib.load(FileName)

TextFile = input("\nEnter Path of the Text File: ")

RawData = []
with open(TextFile, 'r') as file:
    lines = file.readlines()
    for line in lines:
        RawData.append(line.strip())

Json = [{"Text": sentence} for sentence in RawData]
Dataframe = pd.DataFrame(Json)
Dataframe.to_csv('Prompt.csv', index=False)

def LoadEncoder(File):
    Embeddings = {}
    with open(File, 'r', encoding="utf-8") as F:
        for line in F:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            Embeddings[word] = vector
    return Embeddings

def SentenceEmbeddings(sentence, Embeddings, Dimensions):
    words = sentence.split()
    Matrix = [Embeddings[word] for word in words if word in Embeddings]
    if len(Matrix) > 0:
        SentenceEmbedding = np.mean(Matrix, axis=0)
    else:
        SentenceEmbedding = np.zeros((Dimensions,))
    return SentenceEmbedding

EPath = input("\nEnter Path of the Word Embeddings File (e.g., glove.txt): ")
print("\nLoading Encoders...This might take some time...\n")
Encoder = LoadEncoder(EPath)

Dataframe = pd.read_csv('Prompt.csv')
Data = Dataframe['Text']

print('\nData Encoded Successfully!')

FinalDataframe = []
for sentence in Data:
    Sentence = SentenceEmbeddings(sentence, Encoder, 300)
    FinalDataframe.append(Sentence)

FinalDataframe = np.array(FinalDataframe)

for X, sentence in enumerate(Data):
    print(f"\nSentence #{X + 1}: -\n")
    Probability = Model.predict_proba(FinalDataframe[X:X+1])
    print(sentence, '\n')
    print(Probability)
    print("\nThe Probability of the Data being: -\n")
    Probability = np.round(Probability, 5) * 100
    print("Negative For The Finance Market: ", Probability[0][0], "%\n")
    print("Neutral For The Finance Market: ", Probability[0][1], "%\n")
    print("Positive For The Finance Market: ", Probability[0][2], "%\n")
