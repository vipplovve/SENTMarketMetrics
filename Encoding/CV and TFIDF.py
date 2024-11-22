import pandas;
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer;

Dataframe = pandas.read_csv('..\\Datasets\\PreProcessedData.csv');

Data = Dataframe['Text'];

CV = CountVectorizer();
TFIDF = TfidfVectorizer();

CVVectors = CV.fit_transform(Data);
TFIDFVectors = TFIDF.fit_transform(Data);

CVDataframe = pandas.DataFrame(CVVectors.toarray(), columns=CV.get_feature_names_out());
TFIDFDataframe = pandas.DataFrame(TFIDFVectors.toarray(), columns=TFIDF.get_feature_names_out());

CVDataframe = pandas.concat([CVDataframe, Dataframe['Verdict']], axis=1);
TFIDFDataframe = pandas.concat([TFIDFDataframe, Dataframe['Verdict']], axis=1);

CVDataframe.to_csv('..\\Datasets\\EncodedData\\CVEncodedData.csv', index=False);
TFIDFDataframe.to_csv('..\\Datasets\\EncodedData\\TFIDFEncodedData.csv', index=False);

print("Features and labels have been saved to CVEncodedData.csv and TFIDFEncodedData.csv successfully.");
