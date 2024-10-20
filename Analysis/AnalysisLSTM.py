import numpy;
import pandas;
import tensorflow;
import os;

os.system('cls');

FileName = input("\nEnter Path of the Model: ");

Model = tensorflow.keras.models.load_model(FileName);

TextFile = input("\nEnter Path of the Text File: ");

RawData = [];

with open(TextFile, 'r') as file:
    
    lines = file.readlines();

    for line in lines:

        RawData.append(line);

Json = [];

for sentence in RawData:
    
    Json.append({"Text": sentence});

Dataframe = pandas.DataFrame(Json);

Dataframe.to_csv('Prompt.csv', index = False);

def ReshapeForLSTM(features):

    num_samples, num_features = features.shape;
    timesteps = 1;
    return features.reshape(num_samples, timesteps, num_features);

def LoadEncoder(File):
    
    Embeddings = {};
    
    with open(File, 'r', encoding="utf-8") as F:
        
        for line in F:
            
            values = line.split();
            word = values[0];
            vector = numpy.asarray(values[1:], dtype='float32');
            Embeddings[word] = vector;
            
    return Embeddings;

def SentenceEmbeddings(sentence, Embeddings, Dimensions):
    
    words = sentence.split();
    
    Matrix = [];
    
    for word in words:
        
        if word in Embeddings:
            
            Matrix.append(Embeddings[word]);
    
    if len(Matrix) > 0: 
        SentenceEmbedding = numpy.mean(Matrix, axis=0);
    else:
        SentenceEmbedding = numpy.zeros((Dimensions,));
    
    return SentenceEmbedding;

EPath = input("\nEnter Path of the Encoder: ");

print("\nLoading Encoders...This would take some Time...(About 100 Seconds)...\n");

Encoder = LoadEncoder(EPath);

Dataframe = pandas.read_csv('Prompt.csv');

Data = Dataframe['Text'];

print('\nData Encoded Successfully!');

FinalDataframe = [];

for sentence in Data:
    
    Sentence = SentenceEmbeddings(sentence, Encoder, 300);
    
    FinalDataframe.append(Sentence);   
    
FinalDataframe = numpy.array(FinalDataframe);

FinalDataframe = ReshapeForLSTM(FinalDataframe);

for X, sentence in enumerate(Data):
    
    print(f"\nSentence #{X + 1}: -\n");
    
    Probability = Model.predict(FinalDataframe[X:X+1]);
     
    print(sentence);
    
    print("\nThe Probability of the Data being: -\n");

    Probability = numpy.round(Probability, 5) * 100;
    
    print("Negative For The Finance Market: ", Probability[0][0], "%\n");
    print("Neutral For The Finance Market: ", Probability[0][1], "%\n");
    print("Positive For The Finance Market: ", Probability[0][2], "%\n");
            
    if(Probability[0][0] - Probability[0][1] > 5 and Probability[0][0] - Probability[0][2] > 5):
        print("The Data is Negative for the Finance Market!\n");
    elif(Probability[0][1] - Probability[0][0] > 5 and Probability[0][1] - Probability[0][2] > 5):
        print("The Data is Neutral for the Finance Market!\n");
    elif(Probability[0][2] - Probability[0][0] > 5 and Probability[0][2] - Probability[0][1] > 5):
        print("The Data is Positive for the Finance Market!\n");
    elif(Probability[0][0] - Probability[0][1] < 5 and Probability[0][0] - Probability[0][2] > 5):
        print("The Data is Mixed (Negative + Neutral) for the Finance Market!\n");
    elif(Probability[0][0] - Probability[0][1] > 5 and Probability[0][0] - Probability[0][2] < 5):
        print("The Data is Mixed (Negative + Positive) for the Finance Market!\n");
    else:
        print("The Data is Mixed (Neutral + Positive) for the Finance Market!\n");
