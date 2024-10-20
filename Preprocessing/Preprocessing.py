import pandas;
import nltk;
from nltk.corpus import stopwords;
from nltk.tokenize import word_tokenize;
from nltk.stem import WordNetLemmatizer;

nltk.download('stopwords');
nltk.download('punkt');
nltk.download('wordnet');
nltk.download('punkt_tab');

Dataframe = pandas.read_csv('C:\\Users\\viplo\\Desktop\\stuff\\SENTMarketMetrics\\Datasets\\NewsClassification.csv');

def remove_punctuation(text):
    punctuations = "?:!.,;"
    temp = ''
    
    # Convert text to string in case it's not
    text = str(text)

    for char in text:
        if char not in punctuations:
            temp += char

    return temp


Dataframe['Headline'] = Dataframe['Headline'].str.lower().apply(remove_punctuation);

stop_words = set(stopwords.words('english'));

def remove_stopwords(sentence):
    words = word_tokenize(sentence);
    return ' '.join([word for word in words if word not in stop_words]);

Dataframe['Headline'] = Dataframe['Headline'].apply(remove_stopwords);

lemmatizer = WordNetLemmatizer();

def lemmatize_words(sentence):
    words = word_tokenize(sentence);
    return ' '.join([lemmatizer.lemmatize(word, 'v') for word in words]);

Dataframe['Headline'] = Dataframe['Headline'].apply(lemmatize_words);

Dataframe.to_csv('..\\Datasets\\PreProcessedNews.csv', index=False);

print("\nData Preprocessed & Saved in CSV Format Successfully.");