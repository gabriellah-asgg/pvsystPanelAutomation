import pandas as pd
import pickle


class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def process_dataframe(self):
        # read in and process dataframe to display
        display_df = pd.read_excel(self.filepath, index_col=(0, 1), skiprows=1)
        display_df.rename(columns=lambda column: column.strip(), inplace=True)
        display_df.drop(labels='Unnamed: 10', axis=1, inplace=True)
        display_df.drop(index='Totals', inplace=True)
        display_df.fillna(0, inplace=True)
        display_df = display_df[(display_df.index.get_level_values('Sub-Section').str.strip() != 'Totals') & (
                display_df.index.get_level_values('Sub-Section').str.strip() != 'Grand Totals')]

        # create dataframe to use for model
        model_df = display_df.reset_index(drop=True)
        model_df = model_df.iloc[:, :-1]

        # make sure all inputs are numeric
        for index, row in model_df.iterrows():
            model_df.iloc[index] = pd.to_numeric(row, errors='coerce')

        # get rid of any missing rows
        model_df = model_df[model_df['MWh'] != 0]
        return display_df, model_df
