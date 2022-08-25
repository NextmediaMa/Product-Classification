from flask import request
from api.controller import predict, getCategories
import warnings
from api import app

warnings.filterwarnings("ignore")


@app.route('/categorise', methods=['POST'])
def categoriseMultipleProducts():

    content_type = request.headers.get('Content-Type')

    if content_type == 'application/json':
        return predict(request.json)
    else:
        return 'Content-Type not supported!', 400


@app.route('/categorise/probability', methods=['POST'])
def categoriseProductWithProbability():

    content_type = request.headers.get('Content-Type')

    if content_type == 'application/json':
        return predict(request.json, True)
    else:
        return 'Content-Type not supported!', 400


@app.route('/categories/list', methods=['GET'])
def getAllCategories():
    return getCategories()
