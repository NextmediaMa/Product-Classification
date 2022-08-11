## for processing
import re
import nltk
import string

lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords


def utils_preprocess_text(text, lst_stopwords=None):
    ## lower casing
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Lemmatisation
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text


# remove punctutation

PUNCT_TO_REMOVE = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def preprocess(df):
    #df['title'] = df['title'].str.replace('\d+', '')
    df["title"] = df["title"].apply(lambda x: utils_preprocess_text(x, lst_stopwords=lst_stopwords))
    #df["title"] = df["title"].apply(lambda text: remove_punctuation(text))
    #df['title'] = df['title'].apply(remove_urls)
    return df
