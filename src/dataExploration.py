import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# random state
rand = 42

# path to file
file_path = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\2024_07_08_PVsyst\Compiled Results.xlsx'


# read excel file into pandas dataframe
display_df = pd.read_excel(file_path, index_col=(0, 1), skiprows=1)
display_df.rename(columns=lambda column: column.strip(), inplace=True)
display_df.drop(labels='Unnamed: 10', axis=1, inplace=True)
display_df.drop(index='Totals', inplace=True)
display_df.fillna(0, inplace=True)
display_df = display_df[(display_df.index.get_level_values('Sub-Section').str.strip() != 'Totals') & (
        display_df.index.get_level_values('Sub-Section').str.strip() != 'Grand Totals')]
print(display_df.head())
print(display_df.info())

# create dataframe to use for model
model_df = display_df.reset_index(drop=True)
model_df = model_df.iloc[:, :-1]

for index, row in model_df.iterrows():
    model_df.iloc[index] = pd.to_numeric(row, errors='coerce')

# get rid of any missing rows
model_df = model_df[model_df['MWh'] != 0]
print(model_df.info())

# create heatmap
corr_matrix = model_df.corr()
plt.figure()
sns.heatmap(corr_matrix)

# create a histogram
for column in model_df.columns:
    plt.figure()
    plt.hist(model_df[column])
    plt.title(column)

    # create boxplots
    plt.figure()
    plt.boxplot(model_df[column])
    plt.title(column)
# data distribution is bimodal for most
# outliers in each feature but not too many

plt.show()
