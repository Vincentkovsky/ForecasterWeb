import json

import pandas as pd


class JsonToCsvConverter:
    def __init__(self):
        pass

    @staticmethod
    def read_json(filename: str) -> dict:
        try:
            with open(filename, "r") as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            raise Exception(f"File '{filename}' not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON in file '{filename}'.")

        return data

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

    def convert_to_csv(self, input_filenames: list, output_filename: str):
        dataframe = None

        for filename in input_filenames:
            data = self.read_json(filename)
            data_list = data["data"]

            if dataframe is None:
                dataframe = self.create_dataframe(data_list)
            else:
                dataframe = pd.concat(
                    [dataframe, self.create_dataframe(data_list)], ignore_index=True
                )

        dataframe.to_csv(output_filename, index=False)
