import os

from common import append_dict, partition
import utilities.datagrid as dg

import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from joblib import dump # Similar to pickle, optimized for objects with large internal numpy arrays

import os
from common import append_dict, partition, df_reorder_columns
import utilities.datagrid as dg
import numpy as np, pandas as pd, datetime
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from joblib import dump


def run_batch(datasets,
              data_config,
              regressors,
              regressor_param_grid,
              cv=3,
              ext_dataset=None,
              eval_type='regressor',
              cnd=None,
              threshold=None,
              show_best_runs=0,
              ):
    '''
        'run_batch' is an updated version of the legacy 'execute' function designed to use internal DataGrids for
    multi-threaded grid searches throughout the pipeline. The code has been streamlined to accept input with better
    readability and generality. As a result of the parallel-batch nature of this function there are no for-loops and
    the trade off we have for speed is a decrease in verbosity.
    TODO: Add progress indicators and a verbosity option to either run_batch or nary_product.

    :param datasets: List of DataSet type objects with identical feature names.
    :param data_config: Dictionary containing information on how the datasets should be extracted and scaled.
        :key target_cnames: List of column names to be used as targets.
        :key feature_options: {group_name: {option: [features]}}
            Dictionary of feature groups, each containing 'option: [features]' key-value pairs.
        :key transform_options: {feature: {category: [transformers]}}
            Dictionary of categorical features to split the run over, for each possible category there is a list of
            transformers. Ideal for sex and ethnicity.
        :key scalar_config: Dictionary containing 'feature: scalar' key-value pairs for sklearn formatted scalars.
    :param regressors:
    :param regressor_param_grid:
    :param cv: Number of folds in cross-validation.
    :param ext_dataset:
    :param eval_type:
    :param cnd:
    :param threshold:
    :param show_best_runs:
    :return:
    '''
    if eval_type == 'classifier':
        cv_method = stratifiedCV
        score_method = classifierScore
    else:
        cv_method = regressorCV
        score_method = regressorScore

    # create DataGrids
    dataset_dg = dg.singleton_datagrid(datasets, 'DataSet')
    target_dg = dg.singleton_datagrid(data_config['target_cnames'], 'Target')
    trnsfrm_dg = dg.list_product(lambda x, y: x + y, dg.option_datagrid_list(data_config['transform_options']))
    cname_dg = dg.list_product(lambda x, y: x + y, dg.option_datagrid_list(data_config['feature_options']))
    cname_dg = cname_dg[(cname_dg['__data__'].map(len) != 0)]  # Remove rows with no input columns

    # extract data to MLPData
    print('[STATUS] Extracting Data')
    data_dg = dg.nary_product((lambda dataset, target, trnsfrm, cnames:
        dataset.extract_data(cnames, target, scaler_config=(data_config['scalar_config']), data_transformers=trnsfrm)),
       dataset_dg, target_dg, trnsfrm_dg, cname_dg, multicore=True)

    print('[STATUS] Training Estimators')
    if regressor_param_grid != {}:
        # perform hyper-parameter grid search
        assert callable(regressors), 'regressor_param_grid cannot be used on multi-regressor runs.'
        hyperparam_dg = dg.compose_dictgrid(regressor_param_grid)
        regressor_dg = dg.nary_product(lambda params: regressors(**params), hyperparam_dg)
        results_dg = dg.nary_product((lambda data, reg: cv_method(data, reg, cnd, threshold, cv=cv)),
                                     data_dg, regressor_dg, multicore=True)
    elif isinstance(regressors, list):
        # train each regressor on the extracted data
        regressor_dg = dg.singleton_datagrid(regressors, 'Model')
        results_dg = dg.nary_product((lambda data, reg: cv_method(data, reg, cnd, threshold, cv=cv)),
                                     data_dg, regressor_dg, multicore=True)
    else:
        # train single regressor on extracted data
        results_dg = dg.nary_product((lambda data: cv_method(data, regressors, cnd, threshold, cv=cv)),
                                     data_dg, multicore=True)
    # unpack final results
    results_df = dg.unpack_dictgrid(results_dg)

    if ext_dataset is not None:
        print('[STATUS] Extracting External Data')
        ext_dataset_dg = dg.singleton_datagrid(ext_dataset, 'ExtDataSet')
        ext_data_dg = dg.nary_product((lambda ext_dataset, target, trnsfrm, cnames:
            ext_dataset.extract_data(cnames, target, scaler_config=(data_config['scalar_config']), data_transformers=trnsfrm)),
          ext_dataset_dg, target_dg, trnsfrm_dg, cname_dg, multicore=True)

        # ensure that ext_data_dg has the same rows as results_df
        if regressor_param_grid != {}:
            ext_data_dg = dg.binary_product(lambda data, reg: data, ext_data_dg, hyperparam_dg)
        elif isinstance(regressors, list):
            ext_data_dg = dg.binary_product(lambda data, reg: data, ext_data_dg, regressor_dg)

        print('[STATUS] Training Estimators on External Data')
        results_df['__data__'] = results_df['estimator']
        results_df = dg.inplace_eval((lambda reg, data:
                                      {'dataset': type(ext_dataset).__name__, **score_method(data, reg, cnd, cv=cv)}),
                                     results_df, ext_data_dg, multicore=True)
        results_df['__data__'] = results_df['__data__'].map(lambda row: {'ext_' + key:val for key, val in row.items()})
        results_df = dg.unpack_dictgrid(results_df)
        results_df = df_reorder_columns(send_back=['estimator'], df=results_df)

    if show_best_runs != 0:
        #results_df['FPR0'] = results_df['FPR'].map(lambda x: 1 - x if x != 0 else x)
        block_size = int(len(results_dg) / (len(dataset_dg) * len(target_dg) * len(trnsfrm_dg)))
        run_blocks = partition(results_df, block_size)
        run_blocks = [df.nlargest(show_best_runs, columns=['test_r2']) for df in run_blocks]
        best_runs = pd.concat(run_blocks)
        print('[STATUS] Best Runs:\n', best_runs)
        # save = str(input('[EXPORT] Save evaluators to disk (y/[n])? '))

    return results_df


def threshold_scan(thresholds,
                    datasets,
                    data_config,
                    regressors,
                    regressor_param_grid,
                    cv=3,
                    ext_dataset=None,
                    cnd=0,
                    show_best_runs=0
                    ):
    if isinstance(thresholds, int):
        thresholds = np.linspace(0, 1, thresholds)
    results = pd.DataFrame()
    for threshold in thresholds:
        print('[STATUS] Current threshold: ', threshold)
        batch_results = run_batch(
                    datasets,
                    data_config,
                    regressors,
                    regressor_param_grid,
                    cv=cv,
                    ext_dataset=ext_dataset,
                    eval_type='classifier',
                    cnd=cnd,
                    threshold=threshold,
                    show_best_runs=show_best_runs
        )
        batch_results.insert(0, 'threshold', threshold)
        results = pd.concat([results, batch_results])
    return results


def ext_train_save(model,
                   dataset,
                   feature_cnames,
                   target_cname,
                   eval_type='classifier',
                   cnd=None,
                   model_name=None
                   ):
    if eval_type == 'classifier':
        cv_method = stratifiedCV
        score_method = classifierScore
    else:
        cv_method = regressorCV
        score_method = regressorScore
    if model_name is None:
        model_name = type(model).__name__
    timestamp = int(datetime.datetime.now().strftime('%Y%m%d%H%M')[2:])
    save_name = f"models/{target_cname}_{model_name}_{timestamp}"
    cv_score = cv_method(dataset.extract_data(feature_cnames, target_cname), model,
      cnd=cnd, total_train=True)
    dump(cv_score['estimator'], f"{save_name}.joblib")
    del cv_score['estimator']
    if os.path.exists('models/models.txt'):
        f = open('models/models.txt', 'a+')
    else:
        f = open('models/models.txt', 'w+')
    f.write(f"Run {timestamp}: {model_name} on {target_cname} with {type(dataset).__name__}\n"
            f"Location: {save_name}.joblib\n"
            f"Features: {feature_cnames}\n"
            f"Score Info: {dict(cv_score)}\n\n")
    f.close()

'''
Cross Validation and Scoring Functions
'''

def regressorCV(data, reg, cnd=None, threshold=None, cv=5, total_train=False):
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()
    cv_scores = cross_validate(reg, X, y, scoring=('r2', ), cv=cv, return_train_score=True,
      return_estimator=True)
    estimator = cv_scores['estimator'][np.argmax(cv_scores['test_r2'])]
    del cv_scores['estimator']
    del cv_scores['fit_time']
    del cv_scores['score_time']
    cv_scores = {key:sum(value) / cv for key, value in cv_scores.items()}
    if total_train:
        reg0 = reg.fit(X, y)
        cv_scores = {**cv_scores, **(regressorScore(data, reg0, prefix='total_train_'))}
        estimator = reg0
    cv_scores['estimator'] = estimator
    return {**{'subjects':len(y)}, **cv_scores}


def regressorScore(data, reg, cnd=None, threshold=None, cv=5, prefix=''):
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()
    columns = dict()
    columns['subjects'] = len(y)
    columns['r2'] = r2_score(y, reg.predict(X))
    if prefix != '':
        columns = {prefix + key:value for key, value in columns.items()}
    return columns


def stratifiedCV(data, clf, cnd=None, threshold=None, cv=5, total_train=False):
    X = data.x_scaled.values
    if len(X) == 0:
        return {'subjects': 0}
    # print('-------------------------------------------------------------')
    # print(X)
    # print(data.y.values)
    y = data.y.values.astype('int32').ravel()

    cv_scores = pd.DataFrame()
    for train_idx, test_idx in StratifiedKFold(n_splits=cv).split(X, y):
        cv_cols = dict()
        X_train, X_test, y_train, y_test = (X[train_idx], X[test_idx], y[train_idx], y[test_idx])
        clf0 = clf if threshold is None else threshold_clf(clf, cnd, threshold)
        clf0.fit(X_train, y_train)
        y_pred = clf0.predict(X_test)
        mtrx = confusion_matrix(y_test, y_pred)
        cv_cols['Confusion Matrix'] = mtrx
        cv_cols['train_acc'] = accuracy_score(y_train, clf0.predict(X_train))
        cv_cols['test_acc'] = accuracy_score(y_test, y_pred)
        if cnd is not None:
            cv_cols = {**cv_cols, **(cnd_score(mtrx, cnd, verbose=True))}
        cv_cols['estimator'] = clf0
        cv_scores = append_dict(cv_scores, cv_cols)

    mean_cv_scores = cv_scores.mean(axis=0)
    for cname in cv_scores.columns:
        if 'Matrix' in cname:
            mean_cv_scores[cname] = np.sum(cv_scores[cname].values) / cv

    mean_cv_scores['estimator'] = cv_scores['estimator'].values[np.argmax(cv_scores['test_acc'].values)]
    for cname in ('tp', 'fn', 'fp', 'tn'):
        mean_cv_scores[cname] = int(mean_cv_scores[cname] * cv)

    if total_train:
        clf0 = clf
        print(clf0)
        print(type(clf0))
        mean_cv_scores = {**mean_cv_scores, **(classifierScore(data, clf0, cnd=cnd, prefix='total_train_'))}
        mean_cv_scores['estimator'] = clf0
    return {**{'subjects': len(y)}, **mean_cv_scores}


def stratifiedCV_ExtValdation(data, clf, cnd=None, threshold=None, cv=5, total_train=False):
    X = data.x_scaled.values
    y = data.y.values.astype('int32').ravel()

    X_cv, X_ext, y_cv, y_ext = train_test_split(X, y, test_size=0.15, random_state=42)  # ext is 'test'

    cv_scores = pd.DataFrame()
    for train_idx, test_idx in StratifiedKFold(n_splits=cv).split(X_cv, y_cv):
        cv_cols = dict()
        X_train, X_test, y_train, y_test = (X_cv[train_idx], X_cv[test_idx], y_cv[train_idx], y_cv[test_idx])
        clf0 = clf if threshold is None else threshold_clf(clf, cnd, threshold)
        clf0.fit(X_train, y_train)
        y_pred = clf0.predict(X_test)
        mtrx = confusion_matrix(y_test, y_pred)
        cv_cols['Confusion Matrix'] = mtrx
        cv_cols['train_acc'] = accuracy_score(y_train, clf0.predict(X_train))
        cv_cols['test_acc'] = accuracy_score(y_test, y_pred)
        if cnd is not None:
            cv_cols = {**cv_cols, **(cnd_score(mtrx, cnd, verbose=True))}
        #cv_cols['estimator'] = clf0
        cv_scores = append_dict(cv_scores, cv_cols)

    mean_cv_scores = cv_scores.mean(axis=0)
    for cname in cv_scores.columns:
        if 'Matrix' in cname:
            mean_cv_scores[cname] = np.sum(cv_scores[cname].values) / cv

    #mean_cv_scores['estimator'] = cv_scores['estimator'].values[np.argmax(cv_scores['test_acc'].values)]
    for cname in ('tp', 'fn', 'fp', 'tn'):
        mean_cv_scores[cname] = int(mean_cv_scores[cname] * cv)

    clf0 = clf
    clf0.fit(X_cv, y_cv)
    ext_pred = clf0.predict(X_ext)
    ext_scores = dict()
    ext_scores['acc'] = accuracy_score(y_ext, ext_pred)
    ext_scores = {**ext_scores, **cnd_score(confusion_matrix(y_ext, ext_pred), cnd, verbose=True)}
    ext_scores = {'ext_' + key: value for key, value in ext_scores.items()}

    if total_train:
        clf0 = clf
        mean_cv_scores = {**mean_cv_scores, **(classifierScore(data, clf0, cnd=cnd, prefix='total_train_'))}
        mean_cv_scores['estimator'] = clf0
    return {**{'subjects': len(y)}, **mean_cv_scores, **ext_scores, **{'estimator': clf0}}


def stratifiedShuffleCV(data, clf, cnd=None, threshold=None, cv=5, nruns=100):
    X = data.x_scaled.values
    y = data.y.values.astype('int32').ravel()
    cv_scores = pd.DataFrame()
    for train_idx, test_idx in RepeatedStratifiedKFold(n_splits=cv, n_repeats=(int(nruns / cv))).split(X, y):
        cv_cols = dict()
        X_train, X_test, y_train, y_test = (X[train_idx], X[test_idx], y[train_idx], y[test_idx])
        clf0 = clf if threshold is None else threshold_clf(clf, cnd, threshold)
        clf0.fit(X_train, y_train)
        y_pred = clf0.predict(X_test)
        mtrx = confusion_matrix(y_test, y_pred)
        cv_cols['Confusion Matrix'] = mtrx
        cv_cols['train_acc'] = accuracy_score(y_train, clf0.predict(X_train))
        cv_cols['test_acc'] = accuracy_score(y_test, y_pred)
        if cnd is not None:
            cv_cols = {**cv_cols, **(cnd_score(mtrx, cnd, verbose=True))}
        cv_cols['estimator'] = clf0
        cv_scores = append_dict(cv_scores, cv_cols)

    mean_cv_scores = cv_scores.mean(axis=0)
    for cname in cv_scores.columns:
        if 'Matrix' in cname:
            mean_cv_scores[cname] = np.sum(cv_scores[cname].values) / nruns

    mean_cv_scores['estimator'] = cv_scores['estimator'].values[np.argmax(cv_scores['test_acc'].values)]
    for cname in ('tp', 'fn', 'fp', 'tn'):
        mean_cv_scores[cname] = int(mean_cv_scores[cname] * (nruns - 1))

    return {**{'subjects': len(y)}, **mean_cv_scores}


def classifierScore(data, clf, cnd=None, threshold=None, cv=5, prefix=''):
    X = data.x_scaled.values
    y = data.y.values.astype('int64').ravel()
    y_pred = clf.predict(X) if threshold is None else \
        threshold_clf(clf, cnd, threshold).predict(X)
    mtrx = confusion_matrix(y, y_pred)
    columns = dict()
    columns['acc'] = accuracy_score(y, y_pred)
    if cnd is not None:
        columns = {**columns, **(cnd_score(mtrx, cnd))}
    if prefix != '':
        columns = {prefix + key:value for key, value in columns.items()}
    return columns


def cnd_score(mtrx, cnd, verbose=False):
    columns = dict()
    tp = mtrx[cnd][cnd]
    fn = np.sum(mtrx[cnd, :]) - tp
    fp = np.sum(mtrx[:, cnd]) - tp
    tn = np.sum(mtrx) - (tp + fn + fp)

    if verbose:
        columns['tn'] = tn
        columns['fn'] = fn
        columns['fp'] = fp
        columns['tp'] = tp

    columns[f"Condition {cnd} Matrix"] = np.asarray([[tp, fp], [fn, tn]])
    if fn + tp != 0:
        columns['TPR'] = tp / (fn + tp)
        columns['FNR'] = fn / (fn + tp)
    else:
        columns['TPR'] = np.nan
        columns['FNR'] = np.nan
    if fp + tn != 0:
        columns['FPR'] = fp / (fp + tn)
        columns['TNR'] = tn / (fp + tn)
    else:
        columns['FPR'] = np.nan
        columns['TNR'] = np.nan
    if tp + fp != 0:
        columns['precision'] = tp / (tp + fp)
    else:
        columns['precision'] = np.nan
    if columns['TPR'] != np.nan:
        pass
    if columns['precision'] != np.nan:
        if columns['TPR'] + columns['precision'] != 0:
            columns['F1'] = 2 * columns['TPR'] * columns['precision'] / (columns['TPR'] + columns['precision'])
        else:
            columns['F1'] = np.nan

    return columns

'''
Custom Estimators For Classification
'''
# Classes for custom sklearn estimator
class Estimator:
    def __init__(self, **kwargs):
        self.params = {**kwargs}

    def get_params(self, **kwargs):
        return self.params


def apply_threshold(proba_list, cnd, threshold):
    if proba_list[cnd] >= threshold:
        return cnd
    else:
        proba_list[cnd] = 0
        return np.argmax(proba_list)


class threshold_clf(Estimator):
    def __init__(self, clf, cnd, threshold=.5):
        super().__init__(clf=clf, cnd=cnd, threshold=threshold)
        self.clf = clf
        self.fit_clf = None
        self.cnd = cnd
        self.threshold = threshold
        # changeClassTypeName(threshold_clf,
        #                    f'{type(clf).__name__}_t({threshold})')

    def fit(self, X, y, **kwargs):
        if self.fit_clf is None:
            self.fit_clf = self.clf.fit(X, y, **kwargs) # Needs revision
        else:
            self.fit_clf = self.clf.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self.fit_clf is None:
            raise Exception('Estimator not fit')
        probas = self.fit_clf.predict_proba(X, **kwargs)
        preds = [apply_threshold(proba, self.cnd, self.threshold) for proba in probas]
        return preds

    def update_threshold(self, threshold):
        self.threshold = threshold


'''
Legacy Runner Functions
'''