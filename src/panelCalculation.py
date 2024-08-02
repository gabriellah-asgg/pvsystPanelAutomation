import pickle


import pandas as pd

from preprocessData import Preprocessor


def get_user_input():
    # get user input of which panels to calculate
    subsection = input("Enter Sub-Section of panels:\n")
    section = subsection[0]

    # get number of panels to add
    num_panels = input("Enter Number of panels:\n")

    return section, subsection, num_panels


def populate_totals(export_dataframe):
    sections = export_dataframe.index.levels[0]
    for current_section in sections:
        section_values = export_dataframe.loc[current_section]
        section_values = section_values[~(section_values.index.get_level_values('Sub-Section').str.strip() == 'Totals')]
        totals = {}
        for column in export_dataframe.columns:
            if column != 'Notes/Comments':
                # ensure all inputs are numeric
                section_values[column] = pd.to_numeric(section_values[column], errors='coerce')
                section_values[column].fillna(0, inplace=True)
                col_sum = section_values[column].sum()
                totals[column] = col_sum
            else:
                totals[column] = ""
        new_total_row = pd.DataFrame(totals, index=[(current_section, 'Totals')])
        export_dataframe.loc[(current_section, 'Totals')] = new_total_row
    return export_dataframe


def calculate_grand_totals(export_dataframe):
    # extract total columns to sum
    totals = display_df[display_df.index.get_level_values('Sub-Section') == 'Totals']
    grand_totals = {}
    for column in export_dataframe.columns:
        if column != 'Notes/Comments':
            col_sum = totals[column].sum()
            grand_totals[column] = col_sum
        # if column is notes/comments append empty string
        else:
            grand_totals['Notes/Comments'] = ""
    new_grand_total_row = pd.DataFrame(grand_totals, index=[('Grand Totals', 'Grand Totals')])
    # add in grand totals
    export_dataframe = pd.concat([export_dataframe, new_grand_total_row])
    return export_dataframe


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
empty_rows = empty_rows[(empty_rows.index.get_level_values('Sub-Section').str.strip() != 'Totals') & (
        empty_rows.index.get_level_values('Sub-Section').str.strip() != 'Grand Totals')]
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
    new_row = [int(num_panels), area, dc_system_size, kwh, mwh, mwh_before_loss, conversion_loss_diff, conversion_loss,
               row['Notes/Comments']]

    # insert new row
    display_df.loc[(section, subsection)] = new_row

# populate totals
populate_totals(display_df)


# remove grand totals
display_df = display_df[~(display_df.index.get_level_values('Sub-Section') == 'Grand Totals')]

# calculate new grand totals and add in
display_df = calculate_grand_totals(display_df)

# process percentages
display_df['Conversion Loss'] = [str((round((loss * 100), 2))) + "%" for loss in display_df['Conversion Loss']]

# create new sheet with estimated values for dataframe
new_filepath = file_path.split('.xlsx')
new_filepath = new_filepath[0] + '_Estimate.xlsx'
display_df.to_excel(new_filepath)
