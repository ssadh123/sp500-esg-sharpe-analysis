import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
import seaborn as sns
import statsmodels.formula.api as smf

##Complete data set for SnP 500 ESG risk ratings 

file = "./SP 500 ESG Risk Ratings.csv"
df2 = pd.read_csv(file)

df2

## Data for SnP 500 ESG Risk Rating 

risk_ratings = pd.read_csv("SP 500 ESG Risk Ratings.csv")

# drop rows where ‘Environment Risk Score’ is NaN
risk_ratings_cleaned = risk_ratings.dropna(subset=['Environment Risk Score'])

# create subset of dataset with only columns we care about, name of the company and ESG, Environment, Governance and Social Risk Scores
final_risk_rate = risk_ratings_cleaned[['Name', 'Total ESG Risk score', 'Environment Risk Score', 'Governance Risk Score', 'Social Risk Score']]

final_risk_rate


vix = pd.read_csv("CBOE VIX.csv")
vix

#VIX > 30 Data and data wragling 
vix = pd.read_csv("CBOE VIX.csv")

# VIX at beginning of month
vix = vix[['Date', 'Open']]

# filter where VIX is greater than 30 to see which periods we will focus on
filtered_vix = vix[vix['Open'] > 30]

filtered_vix


file_path = "./Sharpe Ratios for S&P 500 Co.csv" 
df = pd.read_csv(file_path)

#drop rows with no relevant info
df = df.dropna(how='all')
df = df.drop([513, 514])

#drop Metrics and unnamed columns, since it's the same for every row and has no relevant info
df = df.drop('Metrics', axis=1)
df = df.drop('Unnamed: 40', axis=1)
df = df.drop('Unnamed: 41', axis=1)

df

#Sharpe ratio fo SnP 500 

file_path = "./Sharpe Ratios for S&P 500 Co.csv" 
df = pd.read_csv(file_path)
df
#Sharpe Ratio for Period where VIX > 30 to test for connections 

# Note: column names are referring to the month and year (all data is from the first of the month)
selected_columns = ["Symbol", "Name", "Oct 22", "May 22", "Feb 21", "Nov 20", "July 20", "May 20"]
filtered_df = df[selected_columns]
filtered_df = filtered_df.dropna(how='all')

display(filtered_df)

print(filtered_df.head())

# Load datasets
vix_df = pd.read_csv("./CBOE VIX.csv")
sharpe_ratios_df = pd.read_csv("./Sharpe Ratios for S&P 500 Co.csv")
risk_ratings_df = pd.read_csv("./SP 500 ESG Risk Ratings.csv")

# Cleaning null values
sharpe_ratios_df = sharpe_ratios_df.loc[:, ~sharpe_ratios_df.columns.str.contains('^Unnamed')]

sharpe_ratios_df.dropna(how='all', inplace=True)
risk_ratings_df.dropna(how='all', inplace=True)

# Convert to datetime values
vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.to_period('M')
vix_df = vix_df[['Date', 'Close']]

sharpe_ratios_long = sharpe_ratios_df.melt(
    id_vars=['Symbol', 'Name', 'Metrics'], 
    var_name='Date', 
    value_name='Sharpe Ratio'
)
sharpe_ratios_long['Date'] = pd.to_datetime(sharpe_ratios_long['Date'], format='%b %y', errors='coerce').dt.to_period('M')
sharpe_ratios_long = sharpe_ratios_long.dropna(subset=['Date'])

# Merge Sharpe Ratios with ESG Risk Ratings
merged_data = sharpe_ratios_long.merge(risk_ratings_df, on="Symbol", how="left")

# Merge with VIX data
final_timeseries = merged_data.merge(vix_df, on="Date", how="left")

cols = ["Symbol", "Name_x", "Metrics", "Date", "Sharpe Ratio", "Sector", "Close", "Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", "Social Risk Score"]
final_timeseries = final_timeseries[cols]

final_timeseries

# Agregated all companies time series data
aggregated_timeseries = final_timeseries.groupby('Date').agg({
    'Sharpe Ratio': 'mean',
    'Close': 'mean',
    'Total ESG Risk score': 'mean',
    'Environment Risk Score': 'mean',
    'Governance Risk Score': 'mean',
    'Social Risk Score': 'mean'
}).reset_index()
aggregated_timeseries.dtypes

# Convert numeric columns to float for proper plotting
aggregated_timeseries['Date'] = aggregated_timeseries['Date'].astype(str)
final_timeseries['Date'] = final_timeseries['Date'].astype(str)
aggregated_timeseries


# 1. READ THE S&P 500 ESG DATA (KEEPING ALL ROWS)

esg_file = "SP 500 ESG Risk Ratings.csv"
risk_ratings = pd.read_csv(esg_file)

# 2. READ THE SHARPE RATIO DATA (KEEPING ALL ROWS)
sharpe_file = "Sharpe Ratios for S&P 500 Co.csv"
df_sharpe = pd.read_csv(sharpe_file)

# 3. MERGE THE TWO DATASETS
merged_df = pd.merge(
    risk_ratings, 
    df_sharpe, 
    on="Symbol",
    how="inner"    
)

print("Merged dataset preview:")
display(merged_df.head())

# 4. ENCODE SECTOR TO INCLUDE IT IN THE CORRELATION MATRIX

if "Sector" in merged_df.columns:
    merged_df["Sector"] = merged_df["Sector"].astype(str)
    sectors = merged_df["Sector"].unique()
    sec_dict = {sec: i for i, sec in enumerate(sectors)}
    reverse_sec_dict = {i: sec for sec, i in sec_dict.items()}
    merged_df["Sector_Encoded"] = merged_df["Sector"].map(sec_dict)
else:
    print("No 'Sector' column found to encode.")

# 5. CREATE AN AVERAGE SHARPE RATIO COLUMN

sharpe_cols = ["Sept 22", "April 22", "Feb 22", "Jan 21", "Oct 20", "June 20"]
existing_sharpe_cols = [c for c in sharpe_cols if c in merged_df.columns]

if existing_sharpe_cols:
    merged_df["Avg_Sharpe"] = merged_df[existing_sharpe_cols].mean(axis=1, skipna=True)
else:
    print("No matching Sharpe columns found in the merged dataset.")

# 6. BUILD A CORRELATION MATRIX

num_cols = []
for col in [
    "Environment Risk Score",
    "Social Risk Score",
    "Governance Risk Score",
    "Total ESG Risk Score",
    "Avg_Sharpe",
    "Sector_Encoded"
]:
    if col in merged_df.columns:
        num_cols.append(col)

# Drop rows with missing data in these columns
df_for_corr = merged_df[num_cols].dropna()
corr_matrix = df_for_corr.corr()

print("\nCorrelation Matrix:")
display(corr_matrix)

# 7. VISUALIZE THE CORRELATION MATRIX WITH ANNOTATIONS INSIDE THE CELLS
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix for ESG Scores, Sector, and Sharpe Ratios")
plt.xticks(rotation=45, ha="right")
plt.show()



## Scatter plot of the environment risk score with the AVG_Sharpe 
plt.scatter(x=merged_df['Environment Risk Score'], y=merged_df['Avg_Sharpe'])

plt.xlabel('Environmental Risk Score')
plt.ylabel('Average Sharpe Score')
plt.title('Relationship between Environmental Risk Score and Sharpe Ratio')

plt.show

plt.scatter(x=merged_df['Social Risk Score'], y=merged_df['Governance Risk Score'])

plt.xlabel('Social Risk Score')
plt.ylabel('Governance Risk Score')
plt.title('Relationship between Social Risk Score and Governance Risk Score')

plt.show()

columns_to_check = ["Social Risk Score", "Governance Risk Score", "Environment Risk Score", "Avg_Sharpe"]

# Plot histograms for each column
fig, axes = plt.subplots(nrows=len(columns_to_check), figsize=(8, len(columns_to_check) * 4))

for i, col in enumerate(columns_to_check):
    sns.histplot(df_for_corr[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

df_for_corr.describe()

print(merged_df[["Sector"]].head())

sec = 'Sector_Encoded'
target_col = 'Avg_Sharpe'
cats = df_for_corr[sec].unique()
corr_matrices = {}

for cat in cats:
    cat_df = df_for_corr[df_for_corr[sec] == cat]
    cat_df = cat_df.drop(columns = [sec])
    
    corr_matrix = cat_df.corr()
    if target_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_col].drop(target_col, errors='ignore')
        
        if np.any(np.abs(target_corrs) > 0.3):
            corr_matrices[cat] = corr_matrix

for cat, mat in corr_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)

    plt.title(f"Correlation Matrix for {reverse_sec_dict[cat]} Sector companies ESG Scores and Sharpe Ratios\n (Correlation with Sharpe > 0.3)")
    plt.xticks(rotation=45, ha="right")
    plt.show()

#timeseries data wraggling in Timesweies.ipynb
print( final_timeseries.describe())

plt.figure(figsize=(12, 6))
sns.boxplot(data=final_timeseries, x="Sector", y="Sharpe Ratio")
plt.ylim(0, 1.5)
plt.xticks(rotation=45, ha="right")
plt.title("Sharpe Ratio Distribution by Sector")
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=final_timeseries, x="Date", y="Sharpe Ratio", hue="Sector", estimator="mean")
plt.ylim(0,1.2)
plt.xticks(rotation=45)
plt.title("Average Sharpe Ratio Over Time by Sector")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(final_timeseries.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap for Final Time Series Data")
plt.show()

sector_list = final_timeseries["Sector"].unique()
filtered_sectors = []
corr_matrices = {}

for sector in sector_list:
    if pd.isna(sector):
        continue
    sector_data = final_timeseries[final_timeseries["Sector"] == sector]

    selected_cols = ["Sharpe Ratio", "Close", "Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", "Social Risk Score"]
    corr_matrix = sector_data[selected_cols].corr(numeric_only=True)
    
    if "Sharpe Ratio" in corr_matrix.columns:
        target_corrs = corr_matrix["Sharpe Ratio"].drop("Sharpe Ratio", errors='ignore')
        if np.any(np.abs(target_corrs) > 0.3):
            filtered_sectors.append(sector)
            corr_matrices[sector] = corr_matrix

fig, axes = plt.subplots(nrows=len(filtered_sectors), figsize=(8, len(filtered_sectors) * 5))

if len(filtered_sectors) == 1:
    axes = [axes]

for i, sector in enumerate(filtered_sectors):
    sns.heatmap(corr_matrices[sector], annot=True, cmap="coolwarm", fmt=".2f", ax=axes[i], vmin=-1, vmax=1)
    axes[i].set_title(f"Correlation Heatmap for {sector} Sector (|corr| > 0.3 with Sharpe Ratio)")

plt.tight_layout()
plt.show()

print(aggregated_timeseries.describe())

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

sns.lineplot(data=aggregated_timeseries, x="Date", y="Sharpe Ratio", ax=axes[0])
axes[0].set_title("Average Sharpe Ratio Over Time")

# VIX Close Trend
sns.lineplot(data=aggregated_timeseries, x="Date", y="Close", ax=axes[1])
axes[1].set_title("Average VIX Close Over Time")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Make a copy of dataset
df = final_timeseries.copy()

# Drop rows with missing values 
df = df.dropna(subset=["Sharpe Ratio", "Close", "Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", "Social Risk Score"])

# Fit a linear regression with interaction: 
# Sharpe Ratio ~ VIX + ESG + (VIX * ESG)
model = smf.ols("Q('Sharpe Ratio') ~ Q('Close') * Q('Total ESG Risk score')", data=df).fit()

# Print the summary to see coefficients, p-values and R-squared
print(model.summary())

df = final_timeseries.copy()

# 1. Create bins for VIX. Split into four quartiles (Low, Med-Low, Med-High, High)
df['VIX_bin'] = pd.qcut(df['Close'], q=4, labels=['Low Vol','Med-Low Vol','Med-High Vol','High Vol'])

# 2. Group by Sector and the new VIX_bin
grouped = df.groupby(['Sector', 'VIX_bin'])

for (sector, vol_bin), group_data in grouped:
    # Compute correlation between Sharpe Ratio and ESG in this subset
    corr_value = group_data[['Sharpe Ratio', 'Total ESG Risk score']].corr().iloc[0,1]
    
    print(f"Sector: {sector} | Volatility Bin: {vol_bin} | Correlation (Sharpe vs ESG): {corr_value:.2f}")

# Create a grid of values for VIX (Close) and pick a few representative ESG scores
vix_range = np.linspace(df['Close'].min(), df['Close'].max(), 50)

# Use three ESG levels: 25th, 50th, and 75th percentiles
esg_levels = [
    df['Total ESG Risk score'].quantile(0.25),
    df['Total ESG Risk score'].quantile(0.50),
    df['Total ESG Risk score'].quantile(0.75)
]

plot_data = []
for esg in esg_levels:
    for vix in vix_range:
        plot_data.append({
            'Close': vix,
            'Total ESG Risk score': esg
        })

plot_df = pd.DataFrame(plot_data)

# Get the model's predicted Sharpe Ratios for each (VIX, ESG) pair
plot_df['predicted_sharpe'] = model.predict(plot_df)

# Plot lines for each ESG level
plt.figure(figsize=(8, 6))
for esg in esg_levels:
    subset = plot_df[plot_df['Total ESG Risk score'] == esg]
    plt.plot(subset['Close'], subset['predicted_sharpe'], label=f'ESG={esg:.2f}')

plt.xlabel('VIX (Close)')
plt.ylabel('Predicted Sharpe Ratio')
plt.title('Predicted Sharpe Ratio by ESG Score and VIX by our model')
plt.legend(title='ESG Level')
plt.show()

model_sector = smf.ols("Q('Sharpe Ratio') ~ C(Sector) + Q('Close') * Q('Total ESG Risk score')", data=df).fit()
print(model_sector.summary())

# Segment companies by ESG quartiles
df['ESG_quartile'] = pd.qcut(df['Total ESG Risk score'], 4, labels=['Q1','Q2','Q3','Q4'])

# Group by ESG quartile and maybe further by VIX_bin
grouped_quartiles = df.groupby(['Sector', 'ESG_quartile', 'VIX_bin'])

summary = grouped_quartiles['Sharpe Ratio'].mean().reset_index()

print(summary.head(10))