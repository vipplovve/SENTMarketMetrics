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

GPath = input("\nEnter Path of the GloVe Encoder: ");

FPath = input("\nEnter Path of the FastText Encoder: ");

print("\nLoading GloVe & FastText Encoders...This would take some Time...(About 100 Seconds)...\n");

GloVe = LoadEncoder(GPath);

FastText = LoadEncoder(FPath);

Dataframe = pandas.read_csv('Prompt.csv');

Data = Dataframe['Text'];

print('\nData Encoded Successfully!');

X = 1;

for sentence in Data:
    
    print(f"\nSentence #{X}: -\n");
    
    X += 1;
    
    print(sentence, '\n');
    
    SentenceEmbeddingG = numpy.array([SentenceEmbeddings(sentence, GloVe, 300)]);
    SentenceEmbeddingF = numpy.array([SentenceEmbeddings(sentence, FastText, 300)]);

    FinalDataframeG = pandas.DataFrame(SentenceEmbeddingG);
    FinalDataframeF = pandas.DataFrame(SentenceEmbeddingF);
        
    Probability1 = Model.predict(FinalDataframeG);
    Probability2 = Model.predict(FinalDataframeF);

    print("\nAs Per GloVe and FastText Encoders, The Probability of the Data being: -\n");

    chances = (Probability1 + Probability2) / 2;

    probability = numpy.round(chances, 5) * 100;

    print("Negative For The Finance Market: ", probability[0][0], "%\n");
    print("Neutral For The Finance Market: ", probability[0][1], "%\n");
    print("Positive For The Finance Market: ", probability[0][2], "%\n");
            
    if(probability[0][0] - probability[0][1] > 0.05 and probability[0][0] - probability[0][2] > 0.05):
        print("The Data is Negative for the Finance Market!\n");
    elif(probability[0][1] - probability[0][0] > 0.05 and probability[0][1] - probability[0][2] > 0.05):
        print("The Data is Neutral for the Finance Market!\n");
    elif(probability[0][2] - probability[0][0] > 0.05 and probability[0][2] - probability[0][1] > 0.05):
        print("The Data is Positive for the Finance Market!\n");
    elif(probability[0][0] - probability[0][1] < 0.05 and probability[0][0] - probability[0][2] > 0.05):
        print("The Data is Mixed (Negative + Neutral) for the Finance Market!\n");
    elif(probability[0][0] - probability[0][1] > 0.05 and probability[0][0] - probability[0][2] < 0.05):
        print("The Data is Mixed (Negative + Positive) for the Finance Market!\n");
    else:
        print("The Data is Mixed (Neutral + Positive) for the Finance Market!\n");
