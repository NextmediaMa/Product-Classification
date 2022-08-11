import pickle
import pandas as pd
from api.utils.preprocess import preprocess
import json
from api.utils.functions import combineColumns
import numpy as np


def predict(json_data, predict_proba=False):
    df = pd.DataFrame(json_data, columns=['title', 'description'])
    if df['title'].isnull().sum() > 0:
        return {'message': 'title is obligatory'}, 400
    df_combined = combineColumns(df)
    df_processed = preprocess(df_combined)
    # df = df.reset_index(drop=True)
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    data = vectorizer.transform(df_processed["title"]).toarray()
    df_processed = pd.DataFrame(data)
    model = pickle.load(open('MLP88_08.pkl', 'rb'))
    prediction = model.predict(df_processed)
    result = []
    for i in range(len(prediction)):
        result.append(categories[prediction[i]])
    if predict_proba:
        pred_proba = model.predict_proba(df_processed)
        proba = np.round(pred_proba[0], 3)
        return json.dumps(result[0]), proba
    else:
        # return json.dumps(result, cls=NumpyArrayEncoder)
        json_result = []
        for i in range(len(result)):
            json_result.append({'title': df['title'][i], 'category': result[i]})
        return json.dumps(json_result)


categories = ['Automotive',
              'Pet Supplies',
              'Sports & Outdoors',
              'Beauty',
              'Health & Personal Care',
              'Arts, Crafts & Sewing',
              'Cell Phones & Accessories',
              'Toys & Games',
              'Baby Products',
              'Clothing, Shoes & Jewelry',
              'Appliances',
              'Musical Instruments',
              'Electronics',
              'Tools & Home Improvement',
              'Industrial & Scientific',
              'Office Products',
              'Grocery & Gourmet Food',
              'Patio, Lawn & Garden']
