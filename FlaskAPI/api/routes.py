from flask import request
from api.controller import predict
from api import app


@app.route('/')
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
