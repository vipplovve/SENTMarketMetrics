import pandas;
import numpy;

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
    
    sentence = str(sentence);
    
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

GloVe = LoadEncoder('GloVe6B\\glove.6B.300d.txt');
FastText = LoadEncoder('FastText300D\\wiki-news-300d-1M-subword.vec');

Dataframe = pandas.read_csv('..\\Datasets\\PreProcessedNews.csv');

Data = Dataframe['Headline'];

SentenceEmbeddingG = numpy.array([SentenceEmbeddings(sentence, GloVe, 300) for sentence in Data]);
SentenceEmbeddingF = numpy.array([SentenceEmbeddings(sentence, FastText, 300) for sentence in Data]);

FinalDataframeG = pandas.DataFrame(SentenceEmbeddingG);
FinalDataframeF = pandas.DataFrame(SentenceEmbeddingF);

FinalDataframeG['Category'] = Dataframe['Category'];
FinalDataframeF['Category'] = Dataframe['Category'];

FinalDataframeG.to_csv('..\\Datasets\\EncodedData\\NewsEncodedGloVe.csv', index=False);
FinalDataframeF.to_csv('..\\Datasets\\EncodedData\\NewsEncodedFastTexr.csv', index=False);

print('Data Encoded Successfully!');

print('Data Saved Successfully!');
