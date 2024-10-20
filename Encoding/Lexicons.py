import pysentiment2;
import pandas;

LoughranMc = pysentiment2.LM();
Harvard = pysentiment2.HIV4();

def Lexiconize(text, lexicon):
    
    tokens = lexicon.tokenize(text);  
    score = lexicon.get_score(tokens);  
    return score['Polarity'];  

Dataframe = pandas.read_csv('..\\Datasets\\PreProcessedData.csv');

LoughranMcScores = [];
HarvardScores = [];

for text in Dataframe['Text']:
    
    LMScore = Lexiconize(text, LoughranMc);
    LoughranMcScores.append(LMScore);
    
    HScore = Lexiconize(text, Harvard);
    HarvardScores.append(HScore);

Dataframe['LMScore'] = LoughranMcScores;
Dataframe['HScore'] = HarvardScores;

LexiconData = Dataframe[['LMScore', 'HScore', 'Verdict']];

LexiconData.to_csv('..\\Datasets\\EncodedData\\LexiconEncodedData.csv', index=False);

print("Features and labels have been saved to LexiconEncodedData.csv successfully.");
