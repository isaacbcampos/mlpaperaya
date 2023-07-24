# -*- coding: iso-8859-1 -*-


'''
    Scripts for Brute Force search on all variables for predicting MADRS D2
    Author: Inácio Gomes Medeiros
    E-mail: inaciogmedeiros@gmail.com
    Date: 2023-05-16

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

print(f'Scripts for Brute Force search on all variables for predicting MADRS D2')

print(f'Importing libraries...', end=' ', flush=True)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from itertools import product, combinations

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from tqdm import tqdm
from joblib import delayed, Parallel

print('Done!', flush=True)

pd.options.display.float_format = '{:,.2f}'.format

MODE_PATIENT = True

DELTAS = 'data/h4dt.csv'

if MODE_PATIENT:
    MADRS = 'data/madrs.2.csv'
else:
    MADRS = 'data/madrs.3.csv'

def build_signature(df):
    df['Signature'] = df.apply(lambda row: f'{int(row["Sujeito"])}|{int(row["Controle"])}|{row["Grupo"]}', axis=1)
    return df

def is_a_number(value):
    return all(ch in '0123456789.,-' for ch in str(value))

def format_id(value):
    return ''.join(v for v in value if v in '0123456789')

def format_number(value):
    if is_a_number(value):
        return value.replace(',', '.') if type(value) == str else value
    return np.nan

@np.vectorize
def div(a, b):
    return a if b == 0 else a / b

# Reads and pre-process data
# ----------------------------------------------------------------------------
print(f'Loading and pre-processing data...', end=' ', flush=True)

deltas = pd.read_csv(DELTAS, sep=';').rename({'treatment': 'Grupo', 'id': 'Sujeito'}, axis=1)
deltas['Sujeito'] = deltas['Sujeito'].map(format_id)
deltas['Grupo'] = deltas['Grupo'].map(lambda v: 'Ayahuasca' if v == 1 else 'Placebo')
deltas['Controle'] = 0
deltas = build_signature(deltas)

madrs = pd.read_csv(MADRS).dropna()

for col in deltas.columns[4:-3]:
    if deltas.dtypes[col] == object:
        deltas[col] = deltas[col].map(format_number).astype(np.float64)

cols_to_exclude = [col for col in deltas.columns if col[-1] == '1' and col != 'D-1']
deltas = deltas[[col for col in deltas.columns if col not in cols_to_exclude]]

madrs['Grupo'] = madrs['Grupo'].map(lambda v: 'Ayahuasca' if v == 'J' else 'Placebo')
madrs = build_signature(madrs)

if MODE_PATIENT:
    madrs['D7_Basal'] = (div(madrs['D7'], madrs['D-1']) - 1) * 100

madrs['D2_Basal'] = (div(madrs['D2 (Alta)'], madrs['D-1']) - 1) * 100

def join(base_df, madrs_df):
    madrs_df = madrs_df[madrs_df['Signature'].isin(base_df['Signature'])]
    base_df = base_df.sort_values(by='Signature')
    madrs_df = madrs_df.sort_values(by='Signature')
    cols_to_use = madrs_df.columns.difference(base_df.columns)
    base_df = base_df.merge(madrs_df[['Signature', *cols_to_use]], on='Signature')
    return base_df

deltas = join(deltas.copy(), madrs.copy())

translated_variables = {
    'sex': 'sex',
    'group': 'group',
    'rbc': 'red blood cells',
    'hemoglobin': 'hemoglobin',
    'hematocrit': 'hematocrit',
    'vcm': 'vcm',
    'hcm': 'hcm',
    'chcm': 'chcm',
    'leukocytes': 'leukocytes',
    'neutrophils': 'neutrophils',
    'segmented': 'segmented neutrophils',
    'eosinophils': 'eosinophils',
    'lymphocyte': 'lymphocyte',
    'Monocytes': 'monocytes',
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

deltas = deltas.rename(columns=translated_variables)

print('Done!', flush=True)

# Performs Brute-Force search
# ----------------------------------------------------------------------------
print(f'Performing Brute-Force search...', flush=True)

def make_folds(df, splits=10):
    scores = []
    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(df):
        train = df.loc[train_index]
        test = df.loc[test_index]
        yield train, test

def cv(df, features, class_label, splits=5):
    folds = make_folds(df.reset_index(drop=True), splits=splits)

    for fold_train, fold_test in folds:
        regressor = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        regressor = regressor.fit(fold_train[features], fold_train[class_label])
        yield regressor, fold_train, fold_test

def process(features_to_use):
    d1 = {'features': ','.join(features_to_use)}

    d2 = {
        grupo: np.array([
            regressor.score(fold_test[features_to_use], fold_test[f'D2_Basal'])
            for regressor, fold_train, fold_test in cv(df_grupo, features_to_use, f'D2_Basal', splits=5)
        ]).mean()
        for grupo, df_grupo in deltas.groupby('Grupo')
    }
    
    return {**d1, **d2}

features = [
    'creatinin', 'ast', 'alt',
    'red blood cells', 'hemoglobin', 'hematocrit', 'vcm', 'hcm', 'chcm',
    'leukocytes', 'neutrophils', 'neutrophils2', 'segmented neutrophils',
    'segmented2', 'eosinophils', 'eosinophils2', 'lymphocyte',
    'lymphocyte2', 'monocytes', 'Monocytes2', 'platelets', 'glucose',
    'sodium', 'potassium', 'hdl cholesterol', 'ldl cholesterol',
    'triglycerides', 'total cholesterol', 'crp', 'urea',
    'plasmatic cortisol', 'il6', 'bdnf', 'awakening salivary cortisol',
]

r = []
results = pd.DataFrame()
i = 0

for perms in tqdm(range(1, len(features))):
    for features_mask in combinations(features, r=perms):
        r.append(delayed(process)(list(features_mask)))

        i += 1

        if ((i % 30) == 0) and (i > 0):
            r = pd.DataFrame(Parallel(n_jobs=-1)(r))
            results = pd.concat([results, r]).sort_values(by='Ayahuasca', ascending=False).head(10)
            del r
            r = []

            
if len(r) > 0:
    r = pd.DataFrame(Parallel(n_jobs=-1)(r))
    results = pd.concat([results, r]).sort_values(by='Ayahuasca', ascending=False).head(10)
    del r
    r = []
    
print(f'Done!', end=' ', flush=True)
    
# Saving results
# ----------------------------------------------------------------------------
print(f'Saving results...', end=' ', flush=True)

results.sort_values(by='Ayahuasca', ascending=False).to_csv(f'tables/brute_force_madrs_D2_all_variables.csv', index=False)

print(f'Done!', flush=True)
print(f'Script has completed successfully! Please check \'tables\' folder to see the results', flush=True)
