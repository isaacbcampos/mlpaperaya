# -*- coding: iso-8859-1 -*-


'''
    Scripts for analyzing MADRS brute-force search results
    Author: Inácio Gomes Medeiros
    E-mail: inaciogmedeiros@gmail.com
    Date: 2023-05-17

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

print('Scripts for analyzing MADRS brute-force search results')
print(f'Importing libraries...', end=' ', flush=True)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import Counter
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

print(f'Done!', flush=True)

pd.options.display.float_format = '{:,.2f}'.format

IMG_DIR = 'figures'
FONTSIZE = 15

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

print(f'Done!', flush=True)


# Helper functions for analyzing MADRS brute-force search results
# ----------------------------------------------------------------------------


def make_folds(df, splits=10):
    scores = []
    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(df):
        train = df.loc[train_index]
        test = df.loc[test_index]
        yield train, test

        
def cv(df, features, class_label, splits=5):
    folds = make_folds(df.reset_index(drop=True), splits=5)

    for fold_train, fold_test in folds:
        regressor = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        regressor = regressor.fit(fold_train[features], fold_train[class_label])
        yield regressor, fold_train, fold_test

        
def process(features_to_use, class_label='D2_Basal'):    
    d1 = {'features': ','.join(features_to_use)}

    d2 = {
        grupo: np.array([
            regressor.score(fold_test[features_to_use], fold_test[class_label])
            for regressor, fold_train, fold_test in cv(df_grupo, features_to_use, class_label, splits=5)
        ]).mean()
        for grupo, df_grupo in deltas.groupby('Grupo')
    }
    
    return {**d1, **d2}


@dataclass
class FeaturesResults:
    feat_list: list
    class_label: str
    coefficients: pd.DataFrame
    scores: pd.Series

    
def build_features_results(feat_list, class_label):
    coefs = []
    scores = []
    for regressor, fold_train, fold_test in cv(deltas[deltas['Grupo'] == 'Ayahuasca'], feat_list, class_label, splits=5):
        coefs.append({feature: coef for feature, coef in zip(feat_list, regressor.steps[1][1].coef_)})
        scores.append(regressor.score(fold_test[feat_list], fold_test[class_label]))

    coefs = pd.DataFrame(coefs).transpose()
    scores = pd.Series(scores)
    
    return FeaturesResults(feat_list=feat_list, class_label=class_label, coefficients=None, scores=scores)


def evaluate_results(results_file_name, base_output_file_name):
    class_label = 'D2_Basal'

    print(f'\tLoading {results_file_name}....', end=' ', flush=True)
    
    best_results = pd.read_csv(results_file_name)
    
    print(f'Done!', flush=True)

    all_features = [
        'creatinin',
        'ast',
        'alt',
        'red blood cells',
        'hemoglobin',
        'hematocrit',
        'leukocytes',
        'neutrophils',
        'neutrophils2',
        'segmented neutrophils',
        'segmented2',
        'eosinophils',
        'eosinophils2',
        'lymphocyte',
        'lymphocyte2',
        'monocytes',
        'Monocytes2',
        'platelets',
        'glucose',
        'sodium',
        'potassium',
        'hdl cholesterol',
        'ldl cholesterol',
        'triglycerides',
        'total cholesterol',
        'crp',
        'urea',
        'plasmatic cortisol',
        'il6',
        'bdnf',
        'awakening salivary cortisol'
    ]
    
    print(f'\tCalculating features occurrences in best features sets...', end=' ', flush=True)
    
    best_features = [feature_set.split(',') for feature_set in best_results['features'].values]
    best_features_names = [f'Best Feature Set {i+1}' for i in range(len(best_features))]
    best_results['N'] = best_results['features'].map(lambda v: len(v.split(',')))

    features_occurrences = pd.Series(Counter([e for fs in best_features for e in fs])).sort_values()
    features_to_display = features_occurrences[features_occurrences >= 2].index.values
    
    columns_names = ['All features', *best_features_names]

    features_applications_results = [
        build_features_results(feat_list, class_label)
        for feat_list in (
            all_features,
            *best_features
        )
    ]
    
    unique_features = Counter([e for fs in best_features for e in fs])

    feature_sets_df = pd.DataFrame([
        {
            feature: 'X' if feature in feature_set else ' '
            for feature in sorted(unique_features.keys())
        }    
        for feature_set in best_features
    ], index=columns_names[1:])

    feature_sets_df.to_csv(f'tables/{base_output_file_name}_features_occurrences_in_best_features_sets.csv')
    
    print(f'Done!', flush=True)

    print(f'\tGenerating features occurrences figure...', end=' ', flush=True)
    
    # plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    ax = features_occurrences.plot(kind='barh', color='blue', fontsize=20, ax=ax)
    ax.set_ylabel('Feature', fontsize=20)
    ax.set_xlabel('Number of occurrences\nin best models', fontsize=20)
    for img_format in ('png', 'jpeg'):
        fig.savefig(
            f'{IMG_DIR}/{base_output_file_name}_best_features_frequency.{img_format}',
            format=img_format,
            dpi=300
        )
    
    print(f'Done!', flush=True)
    
    print(f'\tCalculating R² scores for features sets...', end=' ', flush=True)

    scores_data = pd.DataFrame(
        [results.scores for results in features_applications_results],
        index=columns_names
    )

    scores_data.columns = [f'Fold {i+1}' for i in range(5)]
    scores_data['Mean R²'] = scores_data.mean(axis=1)

    scores_data.to_csv(f'tables/{base_output_file_name}_r2_scores.csv')
    
    print(f'Done!', flush=True)


# Analyzing the results
# ----------------------------------------------------------------------------
print('Analyzing results from MADRS D2 Brute-Force search on selected variables from Ayahuasca vs Placebo...', flush=True)
evaluate_results('tables/brute_force_madrs_D2_selected_variables_from_ayahuasca_placebo.csv', 'madrs_D2_selected_variables')
print('Done!', flush=True)

print('Analyzing impacting biomarkers (Red Blood Cells and CRP) from Ayahuasca vs Placebo...', flush=True)
evaluate_results('tables/ayahuasca_impacting_biomarkers.csv', 'ayahuasca_impacting_biomarkers')
print('Done!', flush=True)


print(f'Script has completed successfully! Please check \'figures\' and \'tables\' folders to see the results', flush=True)
