# -*- coding: iso-8859-1 -*-


'''
    Scripts for "Ayahuasca vs Placebo" analysis
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

print(f'Scripts for "Ayahuasca vs Placebo" analysis')
print(f'Importing libraries...', end=' ', flush=True)

import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib import *
from scipy.stats import pearsonr, ttest_ind, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


print(f'Done!', flush=True)

warnings.filterwarnings('ignore')


IMG_DIR = 'figures'
FONTSIZE = 15
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


def convert_to_number(series):
    return series.map(lambda v: str(v).replace(',', '.') if type(v) == str else v).astype(np.float64)


def fix_number(n):
    return str(n).replace(',', '.')


def test_ml_algorithm(df, features, class_label, Algorithm, splits=10):
    scores = []
    trees = []
    for train, test in make_cv(df, class_label, splits=splits):
        dt = Algorithm.fit(train[features], train[class_label])
        scores.append(dt.score(test[features].values, test[class_label]))
        trees.append(dt)
    return pd.Series(scores), trees


# Reads and pre-process data
# ----------------------------------------------------------------------------
print(f'Loading and pre-processing data...', end=' ', flush=True)

deltas_all = pd.read_csv('data/h2dt.csv', sep=';')

deltas = deltas_all.rename({'treatment': 'Grupo'}, axis=1)
deltas = deltas.rename(columns=TRANSLATED_VARIABLES)

deltas['Grupo'] = deltas['Grupo'].map(lambda v: 'Ayahuasca' if v == 1 else 'Placebo')

features = deltas.columns[4:-1]

# for feature in deltas.columns:
#     if '2' == feature[-1]:
#         # features = [f for f in features if f != feature[:-1]]
#         features = [f for f in features if f != feature]

for col in features:
    deltas[col] = deltas[col].map(fix_number).astype(np.float64)

features = [f for f in features if '1' != f[-1] and f not in ['vcm', 'hcm', 'chcm']]
deltas = deltas.reset_index(drop=True)
all_features = features
n_splits = 10

print(f'Done!', flush=True)

# Evaluates machine learning algorithms on data
# ----------------------------------------------------------------------------
print(f'Evaluating machine learning algorithms on data...', end=' ', flush=True)

control_patient_labels = ('PG', 'CG')
control_patient_names = ('patient', 'control')

for label, name in zip(control_patient_labels, control_patient_names):
    algorithms_list = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        SVC(random_state=42),
        LogisticRegression(random_state=42),
        GaussianNB(),
        BernoulliNB(),
    ]

    algorithms_name = [
        'Decision Tree',
        'Random Forest',
        'Support Vector Machine',
        'Logistic Reggression',
        'Gaussian Naive-Bayes',
        'Bernoulli Naive-Bayes',
    ]

    comparison_results = pd.DataFrame({
        name: test_ml_algorithm(
            deltas[deltas['group'] == label].reset_index(drop=True), all_features,
            'Grupo', alg_object,
            splits=5
        )[0]
        for name, alg_object in zip(algorithms_name, algorithms_list)
    }) * 100


    comparison_results.index = [f'Test Fold {i+1}' for i in range(5)]

    comparison_results_t = comparison_results.transpose()
    comparison_results_t['Mean Accuracy'] = comparison_results_t.mean(axis=1)
    comparison_results_t[['Mean Accuracy']].to_csv(f'tables/ayahuasca_vs_placebo_ml_models_comparison_only_{name}.csv')

print(f'Done!', flush=True)

# Pipeline that calculates variable importantes, performs
# Recursive Feature Selection and calculates AUC scores
# ----------------------------------------------------------------------------
def pipeline(df_no_na, base_name, class_label, pos_label, plot_figure=True, splits=5):
    print(f'Analysis for {base_name}', flush=True)
    # Calculates variable importantes
    # ----------------------------------------------------------------------------
    print(f'\t[{base_name}] Calculating variable importances...', end=' ', flush=True)
    scores, trees = cv_scores(df_no_na, all_features, class_label, splits)
    features_scores = pd.DataFrame([pd.Series(T.feature_importances_, index=all_features) for T in trees])
    print(f'Done!', flush=True)
    
    # Recursive Feature Selection
    # ----------------------------------------------------------------------------
    print(f'\t[{base_name}] Performing Recursive Feature Selection...', end=' ', flush=True)
    visualizer = RFECV(RandomForestClassifier(random_state=42), cv=splits)
    visualizer = visualizer.fit(df_no_na[all_features], df_no_na[class_label])
    features_ranking = pd.Series(visualizer.ranking_, index=all_features).sort_values(ascending=True)
    features_ranking_dict = features_ranking.to_dict()
    colors = {feature: 'blue' if ranking == 1 else 'red' for feature, ranking in features_ranking_dict.items()}
    print(f'Done!', flush=True)

    # Plots Figure 3
    # ----------------------------------------------------------------------------
    fig, axs = None, None

    if plot_figure:
        print(f'\t[{base_name}] Plotting Figure...', end=' ', flush=True)

        # plt.style.use('ggplot')
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['font.size'] = 20

        fig, axs = plt.subplots(1, 2, figsize=(24, 12), constrained_layout=True)
        axit = iter(axs.ravel())

        ax = sns.heatmap(features_scores.transpose(), annot=True, cmap='inferno_r', ax=next(axit))
        ax.set_xticklabels([f'Fold {i+1}' for i in range(splits)]);
        ax.set_title('a', fontsize=35, weight='bold')
        ax.title.set_position((-.2, 0.94))  # set position of title "a"
        
        data_to_plot = features_scores.mean(axis=0).sort_values()

        ax = data_to_plot.plot(
            kind='barh', ax=next(axit),
            color=[colors.get(i, 'gray') for i in data_to_plot.index]
        )

        ax.set_xlabel('Importance', color='black')
        # ax.set_ylabel('Variables', color='black')
        ax.set_title('b', fontsize=35, weight='bold')
        ax.title.set_position((-.3, 0.94))

        print(f'Done!', flush=True)
    
    # Calculates AUC scores
    # ----------------------------------------------------------------------------
    print(f'\t[{base_name}] Calculating AUC scores...', end=' ', flush=True)

    selected_features = features_ranking[features_ranking == 1].index

    cv = StratifiedKFold(n_splits=splits)
    classifier = RandomForestClassifier(random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    results = {}

    for feature_set_name, feature_set in zip(['Original', 'Selected'], [all_features, selected_features]):    
        results[feature_set_name] = {'AUCROC': [], 'Specificity': [], 'Sensitivity': [], 'Accuracy': []}
        for i, (train, test) in enumerate(cv.split(df_no_na, df_no_na[class_label])):
            classifier.fit(
                df_no_na.loc[train, feature_set],
                df_no_na.loc[train, class_label]
            )

            predictions = classifier.predict(df_no_na.loc[test, feature_set])

            accuracy = accuracy_score(df_no_na.loc[test, class_label].values, predictions)
            aucroc = roc_auc_score(df_no_na.loc[test, class_label].values, classifier.predict_proba(df_no_na.loc[test, feature_set])[:, 1])

            cm = confusion_matrix(
                df_no_na.loc[test, class_label].values,
                predictions,
                labels=['Ayahuasca', 'Placebo']
            )

            sensitivity, specificity = np.round(cm / cm.sum(axis=1), 2)[[0, 1], [0, 1]]
            results[feature_set_name]['AUCROC'].append(aucroc)
            results[feature_set_name]['Specificity'].append(specificity)
            results[feature_set_name]['Sensitivity'].append(sensitivity)
            results[feature_set_name]['Accuracy'].append(accuracy)

    pd.DataFrame({
        'Original': pd.DataFrame(results['Original']).mean(),
        'Selected': pd.DataFrame(results['Selected']).mean()
    }).to_csv(f'tables/ayahuasca_vs_placebo_{base_name}_auc_scores.csv')
    
    print(f'Done!', flush=True)
    
    return fig, axs


# Execution of the pipeline
# ----------------------------------------------------------------------------
fig, axs = pipeline(deltas[deltas['group'] == 'PG'].reset_index(drop=True), 'Only Patient', 'Grupo', 'Ayahuasca', plot_figure=True, splits=5)
fig.savefig(f'{IMG_DIR}/Figure 3.jpeg', format='jpeg', dpi=300)
fig.savefig(f'{IMG_DIR}/Figure 3.png', format='png', dpi=300)

_ = pipeline(deltas[deltas['group'] == 'CG'].reset_index(drop=True), 'Only Controls', 'Grupo', 'Ayahuasca', plot_figure=False, splits=5)

print(f'Done!', flush=True)
print(f'Script has completed successfully! Please check \'figures\' and \'tables\' folders to see the results', flush=True)
