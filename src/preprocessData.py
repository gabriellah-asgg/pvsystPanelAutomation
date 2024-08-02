import pandas as pd


class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def process_dataframe(self):
        # read in and process dataframe to display
        display_df = pd.read_excel(self.filepath, index_col=(0, 1), skiprows=1)
        display_df.rename(columns=lambda column: column.strip(), inplace=True)
        display_df.drop(labels='Unnamed: 10', axis=1, inplace=True)
        display_df.index = pd.MultiIndex.from_tuples(
            [(i.strip(), j.strip()) for i, j in display_df.index],
            names=display_df.index.names
        )
        # remove extra total rows
        display_df.fillna(0, inplace=True)
        totals_to_remove = (display_df.index.get_level_values('Sub-Section').str.strip() == 'Totals') & (
                    display_df['# of PV Panels'] == 0)
        display_df = display_df[~totals_to_remove]
        grand_totals_to_remove = (display_df.index.get_level_values('Sub-Section').str.strip() == 'Grand Totals') & (
                    display_df['# of PV Panels'] == 0)
        display_df = display_df[~grand_totals_to_remove]

        # create dataframe to use for model

        # remove totals
        totals_to_remove = (display_df.index.get_level_values('Sub-Section').str.strip() == 'Totals')
        model_df = display_df[~totals_to_remove]

        # remove grand totals
        grand_totals_to_remove = (model_df.index.get_level_values('Sub-Section').str.strip() == 'Grand Totals')
        model_df = model_df[~grand_totals_to_remove]
        model_df = model_df.drop(columns=['Notes/Comments'])

        # remove section indexes
        model_df.reset_index(drop=True, inplace=True)

        # make sure all inputs are numeric
        for index, row in model_df.iterrows():
            model_df.iloc[index] = pd.to_numeric(row, errors='coerce')

        # get rid of any missing rows
        model_df = model_df[model_df['MWh'] != 0]
        return display_df, model_df
