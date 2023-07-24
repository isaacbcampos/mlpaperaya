#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

TRANSLATED_VARIABLES = {
    'sex': 'sex',
    'group': 'group',
    'rbc': 'red blood cells',
    'hemoglobin': 'hemoglobin',
    'hematocrit': 'hematocrit',
    'vcm': 'vcm',
    'hcm': 'hcm',
    'chcm': 'chcm',
    'leukocytes': 'leukocytes',
    'neutrophils': 'neutrophils (%)',
    'neutrophils2': 'neutrophils (absolute)',
    'segmented': 'segmented neutrophils (%)',
    'segmented2': 'segmented neutrophils (absolute)',
    'eosinophils': 'eosinophils (%)',
    'eosinophils2': 'eosinophils (absolute)',
    'lymphocyte': 'lymphocyte (%)',
    'lymphocyte2': 'lymphocyte (absolute)',
    'Monocytes': 'monocytes (%)',
    'Monocytes2': 'monocytes (absolute)',
    'platelets': 'platelets',
    'glucose': 'glucose',
    'sodium': 'sodium',
    'potassium': 'potassium',
    'hdlcholesterol': 'hdl cholesterol',
    'ldlcholesterol': 'ldl cholesterol',
    'triglycerides': 'triglycerides',
    'totalcholesterol': 'total cholesterol',
    'pcr': 'crp',
    'ureia': 'urea',
    'cortiplasm': 'plasmatic cortisol',
    'il6': 'il6',
    'bdnf': 'bdnf',
    'aucsalivar': 'awakening salivary cortisol',
    'creatinin': 'creatinin',
    'tgo': 'ast',
    'tgp': 'alt',
}

# Load the data
data = pd.read_csv('h4dt.csv', sep=';')

# Convert columns to numeric format
for col in ['creatinin', 'rbc', 'hemoglobin', 'hematocrit', 'vcm', 'hcm', 'chcm', 'potassium', 'pcr', 'cortiplasm', 'il6']:
    data[col] = data[col].str.replace(',', '.').astype(float)

# Filter the data for ayahuasca treatment
data = data[data['treatment'] == 1]

# Drop categorical columns
data = data.drop(columns=['group', 'sex', 'id', 'age'])

# Translate variable names
data.rename(columns=TRANSLATED_VARIABLES, inplace=True)

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['deltamadrs'])
y = data['deltamadrs']

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=1)

# Fit the model
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
features_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
features_df['selected'] = features_df['feature'].isin(['bdnf', 'plasmatic cortisol'])

# Order the features by importance in descending order
features_df.sort_values('importance', ascending=False, inplace=True)



# In[6]:


# Create the figure
fig = plt.figure(figsize=(15, 10), dpi=300)  # Adjust the figure size

# Define the grid layout
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

fontsize = 30
fontsize_letters = fontsize / 2

# Create the bar plot (Part A)
ax0 = plt.subplot(gs[:, 0])
sns.barplot(
    y='feature',
    x='importance',
    data=features_df,
    palette=features_df['selected'].map({True: 'blue', False: 'red'}),
    ax=ax0
)
ax0.set_xlabel('Importances')
ax0.text(-0.1, 1, 'a)', fontsize=fontsize_letters, ha='left', va='top', fontweight='bold')

# Part B: Scatter plot with regression line for 'bdnf'
ax1 = plt.subplot(gs[0, 1])
sns.scatterplot(data=data, x='bdnf', y='deltamadrs', ax=ax1)
X_plot = data['bdnf'].values.reshape(-1, 1)
y_plot = data['deltamadrs'].values.reshape(-1, 1)
reg = LinearRegression().fit(X_plot, y_plot)
y_pred = reg.predict(X_plot)
sns.lineplot(x=data['bdnf'], y=y_pred.flatten(), color='red', ax=ax1)
corr, pvalue = pearsonr(data['bdnf'], data['deltamadrs'])
ax1.text(0.05, 0.95, f'Correlation: {corr:.2f}\np-value: {pvalue:.2f}', transform=ax1.transAxes, verticalalignment='top')
ax1.text(-0.2, 1, 'b)', fontsize=fontsize_letters, ha='left', va='top', fontweight='bold')
ax1.set_xlabel('Delta (Δ) BDNF')
ax1.set_ylabel('Delta (Δ) MADRS')

# Part C: Scatter plot with regression line for 'plasmatic cortisol'
ax2 = plt.subplot(gs[1, 1])
sns.scatterplot(data=data, x='plasmatic cortisol', y='deltamadrs', ax=ax2)
X_plot = data['plasmatic cortisol'].values.reshape(-1, 1)
y_plot = data['deltamadrs'].values.reshape(-1, 1)
reg = LinearRegression().fit(X_plot, y_plot)
y_pred = reg.predict(X_plot)
sns.lineplot(x=data['plasmatic cortisol'], y=y_pred.flatten(), color='blue', ax=ax2)
corr, pvalue = pearsonr(data['plasmatic cortisol'], data['deltamadrs'])
ax2.text(0.05, 0.95, f'Correlation: {corr:.2f}\np-value: {pvalue:.2f}', transform=ax2.transAxes, verticalalignment='top')
ax2.text(-0.2, 1, 'c)', fontsize=fontsize_letters, ha='left', va='top', fontweight='bold')
ax2.set_xlabel('Delta (Δ) Plasmatic Cortisol')
ax2.set_ylabel('Delta (Δ) MADRS')

plt.tight_layout()

# Save the figure as a high-resolution PNG image
plt.savefig('RFRegression_figure.png', dpi=300)
plt.savefig('RFRegression_figure.jpg', format='jpg', dpi=300)


# In[16]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

# Load the data
data = pd.read_csv('h4dt.csv', sep=';')

# Convert columns to numeric format
for col in ['creatinin', 'rbc', 'hemoglobin', 'hematocrit', 'vcm', 'hcm', 'chcm', 'potassium', 'pcr', 'cortiplasm', 'il6']:
    data[col] = data[col].str.replace(',', '.').astype(float)

# Filter the data for ayahuasca treatment
data = data[data['treatment'] == 1]

# Drop categorical columns
data = data.drop(columns=['group', 'sex', 'id', 'age'])

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['deltamadrs'])
y = data['deltamadrs']

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=1)

# Initialize RFECV with 2 folds
rfecv = RFECV(estimator=model, step=1, cv=2, scoring='r2')

# Fit RFECV
rfecv.fit(X, y)

# Get the selected features and score
selected_features_rfecv = X.columns[rfecv.support_]
score_rfecv = rfecv.score(X, y)

# Print the selected features and the score
print("Selected features:", selected_features_rfecv)
print("R^2 score:", score_rfecv)

# Select only 'bdnf' and 'cortiplasm'
X_bdnf_cortiplasm = X[['bdnf', 'cortiplasm']]

# Initialize RFECV with 2 folds
rfecv_bdnf_cortiplasm_2 = RFECV(estimator=model, step=1, cv=2, scoring='r2')

# Fit RFECV
rfecv_bdnf_cortiplasm_2.fit(X_bdnf_cortiplasm, y)

# Get the selected features and score
selected_features_rfecv_bdnf_cortiplasm_2 = X_bdnf_cortiplasm.columns[rfecv_bdnf_cortiplasm_2.support_]
score_rfecv_bdnf_cortiplasm_2 = rfecv_bdnf_cortiplasm_2.score(X_bdnf_cortiplasm, y)

# Print the selected features and the score
print("Selected features:", selected_features_rfecv_bdnf_cortiplasm_2)
print("R^2 score:", score_rfecv_bdnf_cortiplasm_2)

# Initialize RFECV with 2 fold using all features
model_all_features = RandomForestRegressor(n_estimators=100, random_state=1)

# Perform cross-validation with 2 folds and calculate R2 scores
cv_scores = cross_val_score(model_all_features, X, y, cv=KFold(n_splits=2), scoring='r2')

# Calculate the mean R2 score
mean_r2_score = np.mean(cv_scores)

# Print the mean R2 score
print("Mean R2 score using all features:", mean_r2_score)



# In[10]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate model performance metrics for model with all variables
cv = KFold(n_splits=2, shuffle=True, random_state=1)
y_pred_all = cross_val_predict(model, X, y, cv=cv)
r2_all = rfecv.score(X, y)
rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
mae_all = mean_absolute_error(y, y_pred_all)

# Calculate model performance metrics for model with only 'bdnf' and 'cortiplasm'
y_pred_selected = cross_val_predict(model, X_bdnf_cortiplasm, y, cv=cv)
r2_selected = rfecv_bdnf_cortiplasm_2.score(X_bdnf_cortiplasm, y)
rmse_selected = np.sqrt(mean_squared_error(y, y_pred_selected))
mae_selected = mean_absolute_error(y, y_pred_selected)

r2_all, rmse_all, mae_all, r2_selected, rmse_selected, mae_selected


# In[12]:


# Create the table
ax0 = plt.subplot(gs[:, 0])
ax0.axis('off')
columns = ['R-squared', 'RMSE', 'MAE']
rows = ['All variables', 'BDNF and Cortiplasm']
cell_text = []
cell_text.append([f'{r2_all:.2f}', f'{rmse_all:.2f}', f'{mae_all:.2f}'])
cell_text.append([f'{r2_selected:.2f}', f'{rmse_selected:.2f}', f'{mae_selected:.2f}'])
table = ax0.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc = 'center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)


# In[ ]:




