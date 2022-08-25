import pickle
import pandas as pd
from api.utils.preprocess import preprocess
import json
from api.utils.functions import combineColumns
import numpy as np
from tensorflow.python.keras.models import load_model


categories = ['Arts Crafts and Sewing',
              'Beauty',
              'Clothing Shoes and Jewelry',
              'Electronics',
              'Grocery and Gourmet Food',
              'Others',
              'Toys and Games']


def predict(json_data, predict_proba=False):
    df = pd.DataFrame(json_data, columns=['title', 'description'])
    if df['title'].isnull().sum() > 0:
        return {'message': 'title is obligatory'}, 400
    df_combined = combineColumns(df)
    df_processed = preprocess(df_combined)
    vectorize = pickle.load(open('TF-IDF.pkl', 'rb'))
    data = vectorize.transform(df_processed["title"]).toarray()
    print(data.shape)
    df_processed = pd.DataFrame(data)
    model = load_model('ANN-N.h5')
    prediction = model.predict(df_processed)
    result = []
    for i in range(len(prediction)):
        result.append(categories[np.argmax(prediction[i])])
    if predict_proba:
        proba = np.round(prediction[0], 3)
        cat_proba = []
        for i in range(len(categories)):
            cat_proba.append({categories[i]: str(proba[i])})
        print(result[0], cat_proba)
        return {'prediction': result[0], 'proba': cat_proba}
    else:
        json_result = []
        for i in range(len(result)):
            json_result.append({'title': df['title'][i], 'category': result[i]})
        print(json_result)
        return json.dumps(json_result)


def getCategories():
    return categories
