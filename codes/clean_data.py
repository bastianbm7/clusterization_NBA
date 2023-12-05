import pandas as pd

# -------------------------
# Read data
df = pd.read_csv('data\\Player Per Game.csv')


# -------------------------
# Filter by last decade season
df = df[df['season'] >= 2013]

# -------------------------
# Select variables to use in clusterization
#          Age
#          experience
#          variables per game

vars_to_select = [6, 7] + list(range(13, len(df.columns)))
df_filter = df.iloc[:, vars_to_select]

# Delete variables containing 'percent'
vars_to_delete = [col for col in df_filter.columns if 'percent' in col]
df_filter = df_filter.drop(columns=vars_to_delete)


# -------------------------
# Check quality of data

# Type of data in each variable
df_filter.info()

# Quantity of na values
pd.isna(df_filter).sum()

# Quantity of duplicated rows
df_filter.duplicated().sum()

# Data is clean

# -------------------------
# Export data
df_filter.to_csv('data\\PPG_data.csv', index=False)