from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
import numpy as np
import nltk
import json
from preprocess import preprocess
from json import JSONEncoder

app = Flask('ame')

CORS(app)

api = Api(app)

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


class Test(Resource):
    def get(self):
        return 'hellooooosooosoosooqofoo'


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


@app.route('/categories', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')

    if content_type == 'application/json':
        return predict(request.json)
    else:
        return 'Content-Type not supported!'


@app.route('/prediction', methods=['GET'])
def proba_pred():
    prediction, proba = predict([request.args.get('title')], True)
    return {'prediction': prediction, 'proba': str(proba)}


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


api.add_resource(Test, '/api')
# api.add_resource(Prediction, '/prediction/<string:product>')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# from app import app
