from flask import request, abort
from api.controller import predict
from api import app


@app.route('/categories', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        return predict(request.json)
    else:
        return 'Content-Type not supported!'


@app.route('/prediction', methods=['GET'])
def proba_pred():
    if request.args.get('title') is None:
        return "title is obligatory", 400
    prediction, proba = predict([{'title': request.args.get('title'), 'description': request.args.get('description')}],
                                True)
    return {'prediction': prediction, 'proba': str(proba)}
