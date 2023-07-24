# -*- coding: iso-8859-1 -*-


'''
    Scripts for "Control vs Patient" analysis
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

print(f'Scripts for "Control vs Patient" analysis')
print(f'Importing libraries...', end=' ', flush=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lib import *
from scipy.stats import pearsonr, ttest_ind, wilcoxon
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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


def convert_to_number(series):
    return series.map(lambda v: str(v).replace(',', '.') if type(v) == str else v).astype(np.float64)


def test_ml_algorithm(df, features, class_label, Algorithm, alg_name, splits=10):
    data = df.copy()
    
    if alg_name == 'XGBoost':
        classes = data[class_label].unique()
        data[class_label] = data[class_label].map({
            class_name: i for i, class_name in enumerate(classes)
        })
    
    scores = []
    trees = []
    for train, test in make_cv(data, class_label, splits=splits):
        dt = Algorithm.fit(train[features], train[class_label])
        scores.append(dt.score(test[features].values, test[class_label]))
        trees.append(dt)
    return pd.Series(scores), trees


# Reads and pre-process data
# ----------------------------------------------------------------------------
print(f'Loading and pre-processing data...', end=' ', flush=True)

df = pd.read_csv('data/h1dt.csv', sep=';')
df = df.rename(columns=TRANSLATED_VARIABLES)

features = df.columns[4:]
for feature in features:
    if feature[-1] == '2':
        # features = [f for f in features if f != feature[:-1]]
        features = [f for f in features if f != feature]

df = df[['sex', 'group', *features]]

for col in df.columns[4:]:
    if df[col].dtype == object:
        df[col] = convert_to_number(df[col])

df_no_na = df.fillna(df.mean())

all_features = [col for col in df_no_na.columns[4:] if col not in ['vcm', 'hcm', 'chcm']]
class_label = 'group'
splits = 5

print(f'Done!', flush=True)

# Evaluates machine learning algorithms on data
# ----------------------------------------------------------------------------
print(f'Evaluating machine learning algorithms on data...', end=' ', flush=True)

algorithms_list = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    SVC(random_state=42),
    LogisticRegression(random_state=42),
    GaussianNB(),
    BernoulliNB(),
    XGBClassifier(),
]

algorithms_name = [
    'Decision Tree',
    'Random Forest',
    'Support Vector Machine',
    'Logistic Reggression',
    'Gaussian Naive-Bayes',
    'Bernoulli Naive-Bayes',
    'XGBoost',
]

comparison_results = pd.DataFrame({
    name: test_ml_algorithm(
        df_no_na, all_features,
        class_label, alg_object,
        name,
        splits=5
    )[0]
    for name, alg_object in zip(algorithms_name, algorithms_list)
}) * 100


comparison_results.index = [f'Test Fold {i+1}' for i in range(5)]

comparison_results_t = comparison_results.transpose()
comparison_results_t['Mean Accuracy'] = comparison_results_t.mean(axis=1)
comparison_results_t.to_csv(f'tables/control_vs_patient_ml_models_comparison.csv')

print(f'Done!', flush=True)

# Calculates feature importantes
# ----------------------------------------------------------------------------
print(f'Calculating variable importances...', end=' ', flush=True)

scores, trees = cv_scores(df_no_na, all_features, class_label, splits)
features_scores = pd.DataFrame([pd.Series(T.feature_importances_, index=all_features) for T in trees])
features_scores_sorted = features_scores.mean().sort_values(ascending=False)

print(f'Done!', flush=True)

# Performs Recursive Feature Selection
# ----------------------------------------------------------------------------
print(f'Performing Recursive Feature Selection...', end=' ', flush=True)

visualizer = RFECV(RandomForestClassifier(random_state=42), cv=splits)
visualizer = visualizer.fit(df_no_na[all_features], df_no_na[class_label])

features_ranking = pd.Series(visualizer.ranking_, index=all_features).sort_values(ascending=True)
features_ranking_dict = features_ranking.to_dict()
colors = {feature: 'blue' if ranking == 1 else 'red' for feature, ranking in features_ranking_dict.items()}

print(f'Done!', flush=True)

# Plots Supplementary Figure 1
# ----------------------------------------------------------------------------
print(f'Generating Supplementary Figure 1...', end=' ', flush=True)

cumulative_features = features_ranking.index

cumulative_scores = []
cumulative_scores_raw = []
for feature_index in range(len(cumulative_features)):
    scores_cf, trees_cf = cv_scores(
        df_no_na, cumulative_features[:feature_index+1], class_label, splits
    )
    cumulative_scores_raw.append(scores_cf)
    cumulative_scores.append([scores_cf.mean(), scores_cf.std()])

cumulative_scores = pd.DataFrame(cumulative_scores, columns=['Mean', 'Std'])
cumulative_scores['Mean'] = cumulative_scores['Mean'] * 100

csr_df = pd.DataFrame(cumulative_scores_raw).transpose()

# plt.style.use('ggplot')
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['font.size'] = 15;

fig, ax = plt.subplots(1, 1, figsize=(12, 12), constrained_layout=True)

ax = cumulative_scores['Mean'][::-1].plot(ax=ax, kind='barh', grid=False);
ax.set_yticks(np.arange(len(cumulative_features)));
ax.set_yticklabels(cumulative_features[::-1]);
ax.set_ylabel('Features');
ax.set_xlabel(f'{splits}-Fold Cross-Validation mean accuracy (in %)');
ax.set_xlim([0, 100]);
ax.axhline((features_ranking.shape[0] - ((features_ranking == 1).sum() + 1)) + .5, c='blue', lw=3);

# props = dict(
#     color='black',
#     fontsize=14,
#     weight='bold', 
# )

# for (i, patch), feature_name in zip(enumerate(ax.patches), ax.get_yticklabels()):
#     if features_ranking_dict[feature_name.get_text()] == 1:
#         ax.annotate("*", xy=(patch.get_x() + (patch.get_width() * 1.02), patch.get_y() + patch.get_height()/8), **props)

fig.savefig(f'{IMG_DIR}/Supplementary Figure 1.jpeg', format='jpeg', dpi=300)
fig.savefig(f'{IMG_DIR}/Supplementary Figure 1.png', format='png', dpi=300)


print(f'Done!', flush=True)

# Plots Figure 2
# ----------------------------------------------------------------------------
print(f'Generating Figure 2...', end=' ', flush=True)

# plt.style.use('ggplot')
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.color'] = 'black'

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

ax.set_xlabel('Importance', color='black');
ax.set_ylabel('Variables', color='black');
ax.set_title('b', fontsize=35, weight='bold')
ax.title.set_position((-.3, 0.94))

fig.savefig(f'{IMG_DIR}/Figure 2.jpeg', format='jpeg', dpi=300)
fig.savefig(f'{IMG_DIR}/Figure 2.png', format='png', dpi=300)

print(f'Done!', flush=True)


# Calculates mean AUC scores
# ----------------------------------------------------------------------------
print(f'Calculating AUC scores...', end=' ', flush=True)

selected_features = features_ranking[features_ranking == 1].index

cv = StratifiedKFold(n_splits=splits)
classifier = RandomForestClassifier(random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

results = {}

for feature_set_name, feature_set in zip(['Original', 'Selected'], [all_features, selected_features]):    
    results[feature_set_name] = {'Sensitivity': [], 'Specificity': [], 'Accuracy': []}
    for i, (train, test) in enumerate(cv.split(df_no_na, df_no_na[class_label])):
        classifier.fit(
            df_no_na.loc[train, feature_set],
            df_no_na.loc[train, class_label]
        )
        
        predictions = classifier.predict(df_no_na.loc[test, feature_set])

        accuracy = accuracy_score(df_no_na.loc[test, class_label].values, predictions)
        
        cm = confusion_matrix(
            df_no_na.loc[test, class_label].values,
            predictions,
            labels=['CG', 'PG']
        )
        
        sensitivity, specificity = np.round(cm / cm.sum(axis=1), 2)[[0, 1], [0, 1]]
        results[feature_set_name]['Accuracy'].append(accuracy)
        results[feature_set_name]['Sensitivity'].append(sensitivity)
        results[feature_set_name]['Specificity'].append(specificity)

mean_auc_scores = pd.DataFrame({'Original': pd.DataFrame(results['Original']).mean(), 'Selected': pd.DataFrame(results['Selected']).mean()})
mean_auc_scores.to_csv(f'tables/control_vs_patient_mean_auc_scores.csv')

print(f'Done!', flush=True)

print(f'Script has completed successfully! Please check \'figures\' and \'tables\' folders to see the results', flush=True)
