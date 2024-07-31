import pickle

import numpy as np
import pandas as pd

from preprocessData import Preprocessor


def get_user_input():
    # get user input of which panels to calculate
    subsection = input("Enter Sub-Section of panels:\n")
    section = subsection[0]

    # get number of panels to add
    num_panels = input("Enter Number of panels:\n")

    return section, subsection, num_panels


# file path
file_path = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\2024_07_08_PVsyst\Compiled Results.xlsx'

# preprocess data
data_processor = Preprocessor(file_path)
display_df, model_df = Preprocessor.process_dataframe(data_processor)

# load in best prediction model
best_model = pickle.load(open('../res/multi_nn.pkl', 'rb'))

# load in pretrained scaler
scaler = pickle.load(open('../res/scaler.pkl', 'rb'))

# get empty rows to predict
empty_rows = display_df[display_df['MWh'] == 0]
for index, row in empty_rows.iterrows():
    num_panels = row.iloc[0]
    section = index[0]
    subsection = index[1]

    # calculate area
    area = row.iloc[1]
    # round this value because not all come out as ints
    dc_system_size = round(row.iloc[2], 0)

    # get prediction
    features = pd.DataFrame(
        {'# of PV Panels': [int(num_panels)], 'Area (m2)': [area], 'DC System Size (kW)': [dc_system_size]},
        index=pd.MultiIndex.from_tuples([(section, subsection)], names=['Section', 'Sub-Section']))
    features = scaler.transform(features)
    predictions = best_model.predict(features)
    mwh = round(predictions[0][0], 0)
    kwh = mwh * 1000
    mwh_before_loss = round(predictions[0][1], 0)
    conversion_loss_diff = mwh - mwh_before_loss
    conversion_loss = (mwh_before_loss - mwh) / mwh_before_loss

    new_row = pd.DataFrame(
        {'# of PV Panels': [int(num_panels)], 'Area (m2)': [area], 'DC System Size (kW)': [dc_system_size], 'kWh': kwh,
         'MWh': mwh, 'MWh Before Inverter Loss': mwh_before_loss,
         'Conversion Loss Difference (MWh)': conversion_loss_diff,
         'Conversion Loss': conversion_loss, 'Notes/Comments': ""},
        index=pd.MultiIndex.from_tuples([(section, subsection)], names=['Section', 'Sub-Section']))

    # remove old row
    display_df.drop(axis=0, index=(section, subsection), inplace=True)

    # insert new row
    display_df = pd.concat([display_df, new_row])

display_df.sort_index(inplace=True)
display_df['Conversion Loss'] = [str(round((loss * 100), 2)) + "%" for loss in display_df['Conversion Loss']]
print(display_df)
