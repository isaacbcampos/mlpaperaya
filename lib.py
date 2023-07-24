import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


IMG_DIR = 'figures'
FONTSIZE = 15


class LinearRegressionPipeline():
    def __init__(self):
        self.clf = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    
    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
    
    def score(self, X, y):
        return self.clf.score(X, y)
    
    def get_params(self, deep=True):
        return self.clf.get_params()

    @property
    def coef(self):
        return self.clf[1].feature_importances_


def make_classification_cv(df, class_label, splits=10):
    scores = []
    kf = StratifiedKFold(n_splits=splits)
    for train_index, test_index in kf.split(df, df[class_label]):
        train = df.loc[train_index]
        test = df.loc[test_index]
        yield train, test

        
def make_reggression_cv(df, splits=10):
    scores = []
    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(df):
        train = df.loc[train_index]
        test = df.loc[test_index]
        yield train, test


def make_cv(df, class_label='', splits=10, classification=True):
    if classification:
        return make_classification_cv(df, class_label, splits)
    return make_reggression_cv(df, splits)


def cv_scores_reggression(df, features, class_label, splits=10):
    scores = []
    lrs = []
    corrs = []
    pcorrs = []
    pvalues = []

    for train, test in make_cv(df, splits=splits, classification=False):
        lr = LinearRegression().fit(train[features], train[class_label])
        lrs.append(lr)
        
        scores.append(lr.score(test[features], test[class_label]))
        corrs.append(pd.Series(lr.coef_, index=features))
    
    return pd.Series(scores), pd.DataFrame(corrs).fillna(0), pcorrs, lrs


def cv_scores_classification(df, features, class_label, splits=10):
    scores = []
    trees = []
    for train, test in make_cv(df, class_label, splits=splits):
        dt = RandomForestClassifier(random_state=42).fit(train[features], train[class_label])
        scores.append(dt.score(test[features], test[class_label]))
        trees.append(dt)
    return pd.Series(scores), trees


def cv_scores(df, features, class_label, splits=10, classification=True):
    if classification:
        return cv_scores_classification(df, features, class_label, splits)
    return cv_scores_reggression(df, features, class_label, splits)


def cv_pipeline(deltas, all_features, deltas_title, target, splits=10):
    scores, corrs, pcorrs, lrs = cv_scores(deltas, all_features, target, splits=splits, classification=False)
    best_lrs = lrs[scores.sort_values().index[0]]
    coefs = pd.Series(best_lrs.coef_, index=all_features)
   
    rows = pd.DataFrame([(row - row.mean()) / row.std() for _, row in corrs.iterrows()])

    fig, ax = plt.subplots(1, 2, figsize=(24, 12), constrained_layout=True)
    ax[0] = sns.heatmap(corrs.transpose(), cmap='inferno_r', annot=True, ax=ax[0])
    ax[0].set_xticklabels([f'Fold {i+1}' for i in range(splits)]);
    ax[1] = sns.heatmap(rows.transpose(), cmap='inferno_r', annot=True, ax=ax[1])
    ax[1].set_xticklabels([f'Fold {i+1}' for i in range(splits)]);
    fig.savefig(f'{IMG_DIR}/madrs_{deltas_title}_{target}_features_correlations_with_target.svg', format='svg', dpi=100)

    selected_features = corrs.mean().sort_values(key=lambda v: abs(v), ascending=False) # .head(10)
    selected_features.to_csv(f'{IMG_DIR}/madrs_{deltas_title}_{target}_selected_features_corrs.csv')

    fig, ax = plt.subplots(1, 1, figsize=(18, 4), constrained_layout=True)

    ax = selected_features.plot(figsize=(18, 4), rot=45, kind='bar', grid=True)
    ax.set_xlabel('Features');
    ax.set_ylabel('Mean coefficient');
    ax.set_xticks(np.arange(len(selected_features.index)));
    ax.set_xticklabels(selected_features.index);

    for fmt_ in ['svg', 'png']:
        fig.savefig(f'{IMG_DIR}/madrs_{deltas_title}_{target}_z_score.{fmt_}', format=fmt_, dpi=100)

    visualizer = RFECV(LinearRegression(), cv=splits)
    visualizer = visualizer.fit(deltas[all_features], deltas[target])

    selected_data = pd.DataFrame(
        [visualizer.grid_scores_, visualizer.support_],
        columns=all_features,
        index=['Scores', 'Ranking']
    ).transpose().sort_values(by=['Ranking', 'Scores'], ascending=False)
  

    cumulative_features = selected_data.index
    fig, ax = plt.subplots(figsize=(24, 4), constrained_layout=True)
    ax = selected_data['Scores'].plot(kind='bar', grid=True, rot=45, ax=ax)
    ax.axvline(selected_data[selected_data['Ranking']].shape[0] - 0.5, c='red');
    ax.set_xlabel('Features');
    ax.set_ylabel('Grid Score');
    ax.set_ylim([-100, 0]);
    for fmt_ in ['svg', 'png']:
        fig.savefig(f'{IMG_DIR}/{deltas_title}_{target}_ginis.{fmt_}', format=fmt_, dpi=100)
    
    cumulative_scores = []
    cumulative_scores_raw = []
    for feature_index in range(len(cumulative_features)):
        scores_cf, trees_cf, _, _ = cv_scores(
            deltas, cumulative_features[:feature_index+1], target, splits,
            classification=False
        )
        cumulative_scores_raw.append(scores_cf)
        cumulative_scores.append([scores_cf.mean(), scores_cf.std()])

    cumulative_scores = pd.DataFrame(cumulative_scores, columns=['Mean', 'Std'], index=cumulative_features)
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 4), constrained_layout=True)

    ax = cumulative_scores['Mean'].plot(ax=ax, grid=True)
    ax.set_xticks(np.arange(len(cumulative_features)));
    ax.set_xticklabels(cumulative_features, rotation=45);
    ax.set_xlabel('Features');
    ax.set_ylabel('10-Fold Cross Validation\nMean R²');
    ax.axvline(selected_data[selected_data['Ranking'] == 1].shape[0]-1, c='red');

    for fmt_ in ['svg', 'png']:
        fig.savefig(f'{IMG_DIR}/{deltas_title}_{target}_features_performances.{fmt_}', format=fmt_, dpi=100)
    
    scores_selected, trees_selected, _, _ = cv_scores(
        deltas, selected_data[selected_data['Ranking']].index,
        target, splits, classification=False
    )

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax = (pd.DataFrame([scores, scores_selected], index=['Original', 'Selected']).transpose()).boxplot(ax=ax, fontsize=FONTSIZE)
    ax.set_title(f'{deltas_title}\np-valor: {ttest_ind(scores, scores_selected).pvalue:.5f}', fontsize=FONTSIZE);
    for fmt in ['svg', 'png']:
        fig.savefig(f'{IMG_DIR}/{deltas_title}_{target}_features_scores.{fmt}', format=fmt, dpi=300)