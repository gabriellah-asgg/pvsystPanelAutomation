import pickle

from preprocessData import Preprocessor
# file path
file_path = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\2024_07_08_PVsyst\Compiled Results.xlsx'

# preprocess data
data_processor = Preprocessor(file_path)
display_df, model_df = Preprocessor.process_dataframe(data_processor)

# load in best prediction model
best_model = pickle.load(open('../res/multi_nn.pkl', 'rb'))

# get user input of which panels to calculate
subsection = input("Enter Sub-Section of panels:\n")
section = subsection[0]

# get number of panels to add
num_panels = input("Enter Number of panels:\n")

# calculate area
area = int(num_panels) * 2
dc_system_size = round(area * .2457, 0)

# get prediction
features = [num_panels, area, dc_system_size]
predictions = best_model.predict(features)
mwh = predictions[0]
kwh = mwh * 1000
mwh_before_loss = predictions[1]
conversion_loss_diff = mwh - mwh_before_loss
conversion_loss = (mwh_before_loss - mwh) / mwh_before_loss

new_row = [section, subsection, num_panels, area, dc_system_size, kwh,mwh, mwh_before_loss, conversion_loss_diff,  conversion_loss, "" ]
