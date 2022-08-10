import pickle
import pandas as pd
from api.utils.preprocess import preprocess
import json
from api.utils.numpyArrayEncoder import NumpyArrayEncoder
import numpy as np


def predict(json_data, predict_proba=False):
    df = pd.DataFrame(json_data, columns=['Title'])
    df = preprocess(df)
    # df = df.reset_index(drop=True)
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    data = vectorizer.transform(df["Title"]).toarray()
    df = pd.DataFrame(data)
    model = pickle.load(open('MLP88_08.pkl', 'rb'))
    prediction = model.predict(df)
    result = []
    print(prediction)
    for i in range(len(prediction)):
        result.append(categories[prediction[i]])
    if predict_proba:
        pred_proba = model.predict_proba(df)
        proba = np.round(pred_proba[0], 3)
        return json.dumps(result[0], cls=NumpyArrayEncoder), proba
    else:
        return json.dumps(result, cls=NumpyArrayEncoder)


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
