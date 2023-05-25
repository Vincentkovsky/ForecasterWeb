import numpy as np

from flask_cors import CORS
from flask import Flask, jsonify
import datetime
import requests
import json
import pandas as pd
from run import run


app = Flask(__name__)

CORS(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    startTime = datetime.datetime.now()
    jsonData = getJsonData()
    jsonTime = datetime.datetime.now()
    df = jsonToCSV(jsonData)
    df = preprocess(df)
    processTime = datetime.datetime.now()
    results = run(df)
    results = processResults(results)
    # print(results)
    resultsSummary = getResultsSummary(results)
    results = results.to_json(orient="records")
    endTime = datetime.datetime.now()
    print("Time taken: ", endTime - startTime)
    print("Get Json time: ", jsonTime - startTime)
    print("Process Json time: ", processTime - jsonTime)
    print("Run predict model time: ", endTime - processTime)

    response = json.dumps(resultsSummary)

    return response


def getResultsSummary(df: pd.DataFrame):
    highTemp = df["app_temp"].max().round().astype(int).item()
    lowTemp = df["app_temp"].min().round().astype(int).item()
    avgTemp = df["app_temp"].mean().round().astype(int).item()
    windSpeed = df["wind_gust_spd"].mean().round().astype(int).item()
    pressure = df["pres"].mean().round().astype(int).item()
    humidity = df["rh"].mean().round().astype(int).item()
    return {
        "highTemp": highTemp,
        "lowTemp": lowTemp,
        "avgTemp": avgTemp,
        "windSpeed": windSpeed,
        "pressure": pressure,
        "humidity": humidity,
    }


def processResults(results):
    df = pd.DataFrame(results)
    dfcol = colsList.copy()
    dfcol.remove("date")
    df.rename(columns={i: col for i, col in enumerate(dfcol)}, inplace=True)

    start_date = datetime.date.today()
    end_date = start_date + pd.DateOffset(days=1)
    time_range = pd.date_range(
        start=start_date, end=end_date, freq="15min", closed="left"
    )
    # 将时间范围赋值给新的列
    df["timestamp"] = time_range
    df.to_csv("predictionResults.csv", index=False)

    return df


def getJsonData():
    # API key
    API_KEY = "5c2dcc46834a46c1837a12eaadfe275a"

    current_date = datetime.date.today()
    print(current_date)

    # Request parameters
    lat = -33.86
    lon = 151.209
    start_date = current_date - datetime.timedelta(days=3)
    end_date = current_date
    tz = "local"

    # API request URL
    url = f"https://api.weatherbit.io/v2.0/history/subhourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz={tz}&key={API_KEY}"

    # Send API request and get response
    response = requests.get(url)

    # Check response status code
    if response.status_code == 200:
        # Extract data from response
        data = response.json()

        # with open("data.json", "w") as outfile:
        #     json.dump(data, outfile)

    else:
        # Print error message
        print(f"Error {response.status_code}: {response.text}")

    return data


def jsonToCSV(jsonData):
    converter = JsonToCsvConverter()
    return converter.convert_to_csv(jsonData, "data.csv")


def preprocess(df: pd.DataFrame):
    one_hot_encoded = pd.get_dummies(df["weather.description"], dtype=float)
    # print(one_hot_encoded)
    df = pd.concat([df, one_hot_encoded], axis=1)
    # df.head()
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

    for col in colsList:
        if col not in df.columns:
            df.insert(len(df.columns), col, pd.Series(0.0, index=df.index))

    dfcol = list(df.columns[:-11]) + sorted(df.columns[-11:])

    df = df.reindex(dfcol, axis=1)

    # df.to_csv("data.csv", index=False)
    return df


class JsonToCsvConverter:
    def __init__(self):
        pass

    @staticmethod
    def create_dataframe(data: list) -> pd.DataFrame:
        # Declare an empty dataframe to append records
        dataframe = pd.DataFrame()

        # Looping through each record
        for d in data:
            # Normalize the column levels
            record = pd.json_normalize(d)
            record = record.reindex(sorted(record.columns), axis=1)

            # Append it to the dataframe
            dataframe = pd.concat([dataframe, record], ignore_index=True)

        return dataframe

    def convert_to_csv(self, jsonData: json, output_filename: str):
        dataframe = None

        data = jsonData
        data_list = data["data"]

        dataframe = self.create_dataframe(data_list)

        return dataframe


colsList = [
    "app_temp",
    "azimuth",
    "clouds",
    "dewpt",
    "dhi",
    "dni",
    "elev_angle",
    "ghi",
    "precip_rate",
    "pres",
    "rh",
    "slp",
    "snow_rate",
    "solar_rad",
    "temp",
    "date",
    "uv",
    "vis",
    "wind_dir",
    "wind_gust_spd",
    "wind_spd",
    "Broken clouds",
    "Clear Sky",
    "Few clouds",
    "Fog",
    "Haze",
    "Heavy rain",
    "Light rain",
    "Moderate rain",
    "Overcast clouds",
    "Scattered clouds",
    "Thunderstorm with heavy rain",
]
