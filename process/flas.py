import numpy as np

from flask_cors import CORS
from flask import Flask
from flask import Flask, jsonify
from proc import proc
import sys

from jsonToCSV import JsonToCsvConverter
import pandas as pd
import numpy as np
import datetime
import requests
import json

sys.path.insert(0, "../Autoformer-main/")


# from exp.exp_main import Exp_Main
# from utils.tools import dotdict


app = Flask(__name__)

CORS(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    print("hello")
    getJsonData()
    jsonToCSV()
    preprocess()
    pro = proc()
    pro.run()
    data = "dater32432a"
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    return response


def getJsonData():
    # API key
    API_KEY = "5c2dcc46834a46c1837a12eaadfe275a"

    current_date = datetime.date.today()
    print(current_date)

    # Request parameters
    lat = -33.86
    lon = 151.209
    start_date = current_date - datetime.timedelta(days=4)
    end_date = current_date - datetime.timedelta(days=1)
    tz = "local"

    # API request URL
    url = f"https://api.weatherbit.io/v2.0/history/subhourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz={tz}&key={API_KEY}"

    # Send API request and get response
    response = requests.get(url)

    # Check response status code
    if response.status_code == 200:
        # Extract data from response
        data = response.json()

        with open("data.json", "w") as outfile:
            json.dump(data, outfile)

    else:
        # Print error message
        print(f"Error {response.status_code}: {response.text}")


def jsonToCSV():
    converter = JsonToCsvConverter()
    input_filenames = ["data.json"]  # Add more filenames if needed
    output_filename = "data.csv"
    converter.convert_to_csv(input_filenames, output_filename)


def preprocess():
    df = pd.read_csv("data.csv", low_memory=False)
    one_hot_encoded = pd.get_dummies(df["weather.description"], dtype=float)
    # print(one_hot_encoded)
    df = pd.concat([df, one_hot_encoded], axis=1)
    # df.head()
    print(df.columns)
    df = df.drop(
        [
            "timestamp_utc",
            "weather.code",
            "weather.description",
            "weather.icon",
            "revision_status",
            "ts",
            "pod",
        ],
        axis=1,
    )
    df = df.rename(columns={"timestamp_local": "date"})

    print(len(df.columns))
    df.to_csv("data.csv", index=False)
