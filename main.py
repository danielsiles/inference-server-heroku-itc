import json
import os

from flask import Flask, request, jsonify
from joblib import load
import pickle

app = Flask(__name__)

features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
                "thal"]


def load_model():
    return pickle.loads(load("./lr.model"))


def load_preprocessing():
    return pickle.loads(load("./scaler.preprocessing"))


@app.route('/')
def hello_world():
    return 'Test route'


@app.route('/predict_single')
def predict_single():
    lrm = load_model()
    prep = load_preprocessing()
    params = request.args
    print(request.args)
    X = []
    for feature in features:
        X.append(float(params[feature]))

    X_prep = prep.transform([X])
    y_pred = lrm.predict(X_prep)
    return {"input": str(dict(zip(features, X))), "output": str(y_pred[0])}


@app.route('/predict_multiple')
def predict_multiple():
    lrm = load_model()
    prep = load_preprocessing()
    body = request.data
    X = []
    data = json.loads(body)
    for x in data["input"]:
        X.append(list(x.values()))
    X_prep = prep.transform(X)
    y_pred = lrm.predict(X_prep)
    return {"output": json.loads(str(y_pred).replace(" ", ","))}


def main():
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()


if __name__ == '__main__':
    main()
