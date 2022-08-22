# for processing
import re
import nltk
import string
import pandas as pd
from bs4 import BeautifulSoup

lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords

PUNCT_TO_REMOVE = string.punctuation


def data_preprocessing(text, lst_stopwords=None):
    # lower casing
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # Tokenize
    lst_text = text.split()
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Lemmatisation
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)

    # delete punctuation
    text = text.translate(str.maketrans('  ', '  ', PUNCT_TO_REMOVE))

    # delete urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r' ', text)

    # delete numbers
    text = ''.join([i for i in text if not i.isdigit()])

    return text


# Delete arabic products using regex


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# function that get dataset and for each product see if it is in english keep them, sinon delete the row
def just_english_products(df):
    English_products = []
    for text in df["title"]:
        if ((is_english(text)) == True):
            English_products.append(text)
    df = pd.DataFrame(English_products, columns=['title'])
    return df


def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    text = ' '.join(soup.stripped_strings)
    return text


def deletehtmlwords(text):
    htmlwords = ['nbsp', 'lt', 'gt', 'amp', 'quot', 'apos', 'cent', 'pound', 'yen', 'euro', 'copy', 'reg']
    textwords = text.split()

    resultwords = [word for word in textwords if word.lower() not in htmlwords]
    text = ' '.join(resultwords)

    return text


def preprocess(df):
    # remove html tags
    df['title'] = df['title'].str.replace(r'<[^<>]*>', ' ', regex=True)
    # df['title'] = df['title'].str.replace('\d+', '')
    df["title"] = df["title"].apply(lambda x: data_preprocessing(x, lst_stopwords=lst_stopwords))
    df["title"] = df["title"].apply(lambda x: deletehtmlwords(x))
#   removing arabic products should be in the command
#    df = just_english_products(df)
    return df
