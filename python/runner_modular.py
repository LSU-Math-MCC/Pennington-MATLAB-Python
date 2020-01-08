import os, datetime, numpy as np, pandas as pd
from joblib import delayed

import utilities.datagrid as dg
from common import append_dict, partition, df_reorder_columns, save_df, df_append, adjusted_r2_score
from utilities.paramutils import iter_params
from utilities.plotutils import hist_df, param_grid_plot, graph_predicted_vs_actual

from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, StratifiedKFold, cross_validate, RepeatedStratifiedKFold, ParameterGrid, GridSearchCV
from joblib import dump  # Similar to pickle, optimized for objects with large internal numpy arrays


def run_batch(datasets,
              data_config,
              regressors,
              regressor_param_grid={},
              cv=3,
              ext_dataset=None,
              eval_type='regressor',
              cnd=None,
              threshold=None,
              show_best_runs=0
              ):
    '''
        'run_batch' is an updated version of the legacy 'execute' function designed to use internal DataGrids for
    multi-threaded grid searches throughout the pipeline. The code has been streamlined to accept input with better
    readability and generality. As a result of the parallel-batch nature of this function there are no for-loops and
    the trade off we have for speed is a decrease in verbosity.
    TODO: Add progress indicators and a verbosity option to either run_batch or eval_product.

    :param datasets: List of DataSet type objects with identical feature names.
    :param data_config: Dictionary containing information on how the datasets should be extracted and scaled.
        :key target_cnames: List of column names to be used as targets.
        :key feature_options: {group_name: {option: [features]}}
            Dictionary of feature groups, each containing 'option: [features]' key-value pairs.
        :key transform_options: {feature: {category: [transformers]}}
            Dictionary of categorical features to split the run over, for each possible category there is a list of
            data transformers. Ideal for sex and ethnicity.
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
    # setup cross-validation and scoring method, different for regression and classification
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

    # extract data to ExtractedData type object
    print('[STATUS] Extracting Data')
    data_dg = dg.nary_product((lambda dataset, target, trnsfrm, cnames:
        dataset.extract_data(cnames, target, scaler_config=(data_config['scalar_config']), data_transformers=trnsfrm)),
       dataset_dg, target_dg, trnsfrm_dg, cname_dg, multicore=True)

    # train, test and score models. There are three possibilities for input.
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

        # unpack final results
        results_df = dg.unpack_dictgrid(results_df)
        results_df = df_reorder_columns(send_back=['estimator'], df=results_df)

    if show_best_runs != 0:
        # results_df['FPR0'] = results_df['FPR'].map(lambda x: 1 - x if x != 0 else x)  # used to sort by FPR for classification runs
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
    '''
        Given a list of threshold values and a classifier with predict_proba, run a run_batch grid search and score the
    classifier at each threshold. All arguments except :thresholds are passed to run_batch.
    '''
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
                   model_name=None,
                   save_location = 'models'
                   ):
    '''
    Extract data, train a model on the data, then save the model to ./models/

    :param model: an sklearn estimator
    :param dataset: a DataSet object from datasets.py
    :param feature_cnames: list of column names to be used as features
    :param target_cname: column name of target
    :param eval_type: 'classifier' or 'regressor'
    :param cnd: condition index to be considered as 'positive' for classification
    :param model_name: name to save the model as (autogenerated by default)
    :param save_location: location to save the model (relative where this function is called from)
    '''
    # TODO: Save to models.csv instead
    if eval_type == 'classifier':
        cv_method = stratifiedCV
        score_method = classifierScore
    else:
        cv_method = regressorCV
        score_method = regressorScore
    if model_name is None:
        model_name = type(model).__name__
    timestamp = int(datetime.datetime.now().strftime('%Y%m%d%H%M')[2:])  # timestamp used in save name
    save_name = f"{save_location}/{target_cname}_{model_name}_{timestamp}"  # location and save name of extracted model
    # train and score the model
    cv_score = cv_method(dataset.extract_data(feature_cnames, target_cname), model, cnd=cnd, total_train=True)
    # export the model
    dump(cv_score['estimator'], f"{save_name}.joblib")
    del cv_score['estimator']
    # append score, input features and the model's name to models.txt for future reference so that the model may be used
    if os.path.exists('models/models.txt'):
        f = open(f'{save_location}/models.txt', 'a+')
    else:
        f = open(f'{save_location}/models.txt', 'w+')
    f.write(f"Run {timestamp}: {model_name} on {target_cname} with {type(dataset).__name__}\n"
            f"Location: {save_name}.joblib\n"
            f"Features: {feature_cnames}\n"
            f"Score Info: {dict(cv_score)}\n\n")
    f.close()


'''
Cross Validation and Scoring Functions
    Any function ended in 'Score' expects to score a pre-trained model for the :reg parameter.
'''


# Train a regressor and score it using cross-validation
def regressorCV(data, reg, cnd=None, threshold=None, cv=5, total_train=False):
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()
    cv_scores = cross_validate(reg, X, y, scoring=('r2', ), cv=cv,
                               return_train_score=True, return_estimator=True)
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


# Score a TRAINED regressor given standard scoring input
def regressorScore(data, reg, cnd=None, threshold=None, cv=5, prefix=''):
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()
    columns = dict()
    columns['subjects'] = len(y)
    columns['r2'] = r2_score(y, reg.predict(X))
    if prefix != '':
        columns = {prefix + key: value for key, value in columns.items()}
    return columns


#
def stratifiedCV(data, clf, cnd=None, threshold=None, cv=5, total_train=False):
    X = data.x_scaled.values
    if len(X) == 0:
        return {'subjects': 0}
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


#
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


#
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


#
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


#
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


# '''
# Legacy Runner Functions
# '''
#
#
# def execute(datasets,
#             loader_params,
#             loader_param_grid,
#             regressors,
#             regressor_param_grid,
#             targets=None,
#             ext_dataset=None,
#             save_name=None,
#             cv=5,
#             multicore=True,
#             write_validation=True,
#             save_df_report=True,
#             show_hist=False,
#             extract_data_callback=None,
#             include_regressor_and_mlpdata=False,
#             # Classifier option in execute function
#             eval_type='regressor',
#             cnd=None,  # Condition (index) to test for, e.g. for GLU_risk 'Healthy' := 1
#             # Export (save to ./models/) best n evaluators for each sex and target option
#             export_best=0,
#             # Save report as spreadsheet and save scatter plot png
#             report=None,
#             save_png=False,
#             show_png=False
#             ):
#     '''
#         The original runner function accepts an instance of the DataSet class (:datasets),
#     extracts groups of features (:loader_param_grid), applies specified scalars (:loader_params) and
#     then performs regression or classification (:eval_type) on :targets using user-specified models (:regressors)
#     and their potential hyper-parameters (:regressor_param_grid).
#         The function may take :targets, :datasets and :regressors as lists. For each combination of these options, a
#     grid search is performed by 'run_grid_search' on :loader_param_grid and :regressor_param_grid keys.
#     '''
#     # Conflicting input check
#     write_validation = write_validation and save_name is not None
#     save_df_report = save_df_report and save_name is not None
#     save_png = save_png and report is not None
#
#     # Warnings for current issues and bugs
#     if regressor_param_grid != {} and isinstance(regressors, list):
#         warnings.warn('Hyperparameter search is not supported for multi-regressor runs, removing regressor_param_grid.')
#         regressor_param_grid = {}
#     if regressor_param_grid != {} and eval_type == 'classifier' and not callable(regressors):
#         warnings.warn('Classifier object is not callable, removing regressor_param_grid.')
#         regressor_param_grid = {}
#     if isinstance(datasets, list):
#         warnings.warn('Multi-dataset run detected, please insure feature column names have been standardized.')
#
#     # Support for old code
#     regressors = regressors if isinstance(regressors, list) else [regressors]
#     targets = targets if isinstance(targets, list) else [targets]
#     datasets = datasets if isinstance(datasets, list) else [datasets]
#     if eval_type == 'classifier':
#         for target in targets:
#             loader_params['scaler_config'][target] = None  # Don't standardize classifier columns
#
#     full_df = pd.DataFrame()
#     best_df = pd.DataFrame() # Needed to keep track of best runs if export_best > 0
#     for target in targets:
#         if targets != [None]:
#             loader_params['label_cname'] = target
#         else:
#             target = loader_params['label_cname']
#
#         for dataset in datasets:
#             print(f"[EXTRAC] Extracting {type(dataset).__name__} for {eval_type} on {loader_params['label_cname']}")
#             # extracts data and applies transformations from loader_params
#             # param_map contains the arguments for extract data
#             # each param_map is the same except for 'feature_cnames'
#             param_maps, column_maps = zip(*iter_params(loader_param_grid, noniterative_params=loader_params))
#             # extracted_data_list - data.x ('data') and data.y ('target') extracted from dataset for each 'segment'
#             extracted_data_list = resolve_delayed(
#                 delayed(dataset.extract_data)(**param_map) for param_map in param_maps)
#
#             if callable(extract_data_callback):
#                 extract_data_callback(extracted_data_list)
#
#             for regressor in regressors:
#                 print(f"[STATUS] Running {type(regressor).__name__}")
#                 grid_search_results = []
#                 if write_validation:
#                     f = open(f"reports/{save_name}-validation.txt", "w")
#                 for data, param_map, column_map in zip(extracted_data_list, param_maps, column_maps):
#                     if show_hist:
#                         hist_df(pd.concat([data.x, data.y], axis=1, sort=False), bins=15)
#                         # hist_df(pd.concat([data.x_scaled, data.y_scaled], axis=1), bins=15)
#                     if write_validation:
#                         feature_cnames = param_map['feature_cnames']
#                         cols = list(data.x.columns.values)
#                         f.write(
#                             f"column_map: {column_map}\nfeature_cnames: {feature_cnames}\n"
#                             f"data.x.columns: {data.x.columns}\nparam_map: {param_map}\n\n\n")
#                         f.flush()
#                     if eval_type == 'regressor':
#                         grid_search_results.append(
#                             delayed(run_grid_search)(data, regressor, regressor_param_grid, ext_dataset=ext_dataset, cv=cv))
#                     elif eval_type == 'classifier':
#                         grid_search_results.append(
#                             delayed(run_grid_search_clf)(data, regressor, regressor_param_grid, cnd=cnd, cv=cv))
#                 if write_validation:
#                     f.close()
#
#                 grid_search_results = resolve_delayed(grid_search_results)  # Actually training the regressors here
#                 master_df = None
#                 for (df, best), column_map, data in zip(grid_search_results, column_maps, extracted_data_list):
#                     # __accuracy(best, data)
#                     if include_regressor_and_mlpdata:
#                         df["best_regressor"] = best
#                         df["mlp_data"] = data
#                     # adds column_map.items() information (tree branch) to row
#                     master_df = df_append(master_df, df, column_map)
#                 if save_df_report:
#                     save_df(master_df, save_name)
#
#                 master_df = df_reorder_columns(master_df, send_back=list(grid_search_results[0][0].columns))
#                 # Prepended columns
#                 master_df.insert(0, 'dataset', type(dataset).__name__)
#                 master_df.insert(0, 'target', target)
#                 master_df.insert(0, 'regressor', type(regressor).__name__)
#
#                 if ext_dataset is not None:
#                     external_data_list = resolve_delayed(
#                         delayed(ext_dataset.extract_data)(**param_map) for param_map in param_maps)
#
#                     ext_scores = []
#                     for data, est in zip(external_data_list, master_df['estimator']):
#                         X = data.x_scaled.values
#                         y = data.y_scaled.values.ravel()
#                         ext_scores += [sum(cross_validate(est, X, y, cv=cv)['test_score'])/cv]
#
#                     master_df.insert(len(master_df.columns) - 2, 'external_r2', ext_scores)
#
#                 full_df = pd.concat([full_df, master_df], sort=False)
#
#             # Exports best n evaluators for each sex option
#             # Possible issue may occur when executing through many datasets
#             # I DO NOT RECOMMEND USING THIS FOR CLASSIFICATION
#             if export_best > 0:
#                 column_data = loader_param_grid["feature_cnames"]["__data__"]
#                 #print(len(column_data), len(extracted_data_list))
#                 sex_options = np.unique(full_df['SEX'])
#                 print(f'[EXPORT] Fitting {len(sex_options)*export_best} '
#                       f'best {target} {eval_type}s on {type(dataset).__name__}')
#                 for sex in sex_options:
#                     df = full_df.loc[(full_df['SEX'] == sex) & (full_df['target'] == target)]
#                     df = df.nlargest(export_best, columns=['mean_test_r2']) if eval_type == 'regressor' \
#                         else df.nlargest(export_best, columns=['TPR'])
#                     for idx, row in df.iterrows():
#                         best_run_dict = dict()
#                         if eval_type == 'regressor':
#                             scores = ['mean_train_r2', 'mean_test_r2']
#                         elif eval_type == 'classifier' and cnd is None:
#                             scores = ['mean_train_acc', 'mean_test_acc', 'Confusion Matrix']
#                         else:
#                             scores = ['mean_train_acc', 'mean_test_acc', 'Confusion Matrix', 'TPR', 'FNR', 'FPR', 'TNR', 'recall', 'F1']
#
#                         row_data = ['regressor', 'target', 'dataset', 'SEX'] + \
#                                    list(loader_param_grid["feature_cnames"].columns[:-1]) + ['n_samples'] + scores
#                         for cname in row_data:
#                             best_run_dict[cname] = row[cname]
#
#                         # Use relative index for extracted_data_list containing parameter (column_data) and SEX columns
#                         grid_len = loader_param_grid["feature_cnames"].shape[0] # Number of rows in parameter grid
#                         data_idx = idx % (len(sex_options) * grid_len)
#                         X = extracted_data_list[data_idx].x_scaled.values
#                         y = extracted_data_list[data_idx].y_scaled.values.ravel() if eval_type == 'regressor' \
#                             else extracted_data_list[data_idx].y.values.astype('int64').ravel()
#                         reg = row['estimator'].fit(X, y)
#                         if eval_type == 'regressor':
#                             best_run_dict['total_train_r2'] = r2_score(y, reg.predict(X))
#                         else:
#                             best_run_dict['total_train_acc'] = accuracy_score(y, reg.predict(X))
#                         best_run_dict['estimator'] = reg
#
#                         best_run_dict['param_grid'] = {'SEX': sex,
#                                                        **loader_param_grid["feature_cnames"].iloc[idx % grid_len, :-1]}
#                         best_run_df = pd.DataFrame(data={key: {0: value} for key, value in best_run_dict.items()})
#                         best_df = pd.concat([best_df, best_run_df])
#         full_df.index = range(len(full_df.index))  # So the run numbers print correctly (reset_index() but actually works)
#
#     if isinstance(report, str):
#         if show_png or save_png:
#             print(f"[REPORT] Writing parameter grid plot to ./reports/{report}.png" if save_png
#                   else "[REPORT] Printing parameter grid plot to console")
#             if eval_type == 'regressor':
#                 mask = (full_df['mean_train_r2'] >= 0) & (full_df['mean_train_r2'] <= 1) & \
#                        (full_df['mean_test_r2'] >= 0) & (full_df['mean_test_r2'] <= 1)
#                 param_grid_plot(full_df.loc[mask],
#                                 [['target', 'dataset', 'regressor'],
#                                  ['SEX'] + list(loader_param_grid['feature_cnames'].keys()[:-1]),
#                                  list(regressor_param_grid.keys())],
#                                 ['mean_train_r2', 'mean_test_r2'],
#                                 save_name=report if save_png else None
#                                 ).show()
#             elif eval_type == 'classifier':
#                 if cnd is None:
#                     mask = (full_df['mean_test_acc'] >= 0) & (full_df['mean_test_acc'] <= 1) & \
#                            (full_df['mean_train_acc'] >= 0) & (full_df['mean_train_acc'] <= 1)
#                 else:
#                     mask = (full_df['FPR'] >= 0) & (full_df['FPR'] <= 1) & (full_df['TPR'] >= 0) & (full_df['TPR'] <= 1)
#                 param_grid_plot(full_df.fillna(420).loc[mask],
#                                 [['target', 'dataset', 'regressor'],
#                                  ['SEX'] + list(loader_param_grid['feature_cnames'].keys()[:-1]),
#                                  list(regressor_param_grid.keys())],
#                                 ['mean_train_acc', 'mean_test_acc'] if cnd is None else ['FPR', 'TPR'],
#                                 save_name=report if save_png else None
#                                 ).show()
#         print(f"[REPORT] Writing results to ./reports/{report}.xlsx")
#         full_df.to_excel(f"reports/{report}.xlsx", index=False)
#
#     if export_best > 0:
#         timestamp = int(datetime.datetime.now().strftime('%Y%m%d%H%M')[2:])
#         best_df.index = range(len(best_df.index))
#         print('[EXPORT] Best runs to be exported:')
#         print(best_df)
#         save = str(input('[EXPORT] Save evaluators to disk (y/[n])? '))
#         if save == 'y':
#             run_list = input('[EXPORT] List evaluators to save (space separated): ')
#             for run_idx in run_list.split():
#                 run_data = best_df.iloc[int(run_idx)]
#                 print(run_data)
#                 save_name = f'models/{run_data["target"]}_{run_data["estimator"]}_{timestamp}-{run_idx}'
#                 dump(run_data["estimator"], f"{save_name}.joblib")
#                 if os.path.exists('models/models.txt'):
#                     f = open("models/models.txt", "a+")  # Append to models.txt
#                 else:
#                     f = open("models/models.txt", "w+")  # Write models.txt
#                 f.write(
#                     f'Run {timestamp}-{run_idx}: {run_data["estimator"]} on ({run_data["sex"]}) {run_data["target"]}\n'
#                     f'Input parameters: {run_data["param_grid"]}\n'
#                     f'train/test: {run_data["mean_train_r2"]}/{run_data["mean_test_r2"]}\n'
#                     f'total train: {run_data["total_train_r2"]}\n\n'
#                 )
#                 f.close()
#     return full_df
#
#
# def __accuracy(mlp, data):
#     y_predict = data.predict(mlp)
#
#     df = pd.DataFrame({"Actual": data.y.squeeze(), "Predicted": y_predict.squeeze()}, index=data.y.index)
#     df["diff"] = (df["Actual"] - df["Predicted"]).abs()
#     df = df.sort_values("diff", ascending=False)
#     print(df)
#     graph_predicted_vs_actual(df["Predicted"], df["Actual"])
#
#
# def run_grid_search(data, regressor, parameter_grid, ext_dataset=None, cv=3, display=["mean", "r2", "std"]): # "r2adj",
#     X = data.x_scaled.values
#     y = data.y_scaled.values.ravel()
#
#     # # limit sample size to N
#     # N = 150
#     # #idxs = np.random.permutation(len(X))[:N] # Select randomly
#     # idxs = range(N)
#     # X = X[idxs]
#     # y = y[idxs]
#
#     # Split to external validation set
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
#     clf = GridSearchCV(regressor,
#                        parameter_grid,  # regressor_parameter_grid
#                        n_jobs=-1,
#                        return_train_score=True,
#                        cv=cv if cv is not None else [
#                            (np.asarray(range(0, len(data.x))), np.asarray(range(0, len(data.x))))],
#                        iid=True
#                        )
#     clf.fit(X, y)
#
#     params = clf.cv_results_['params']
#     best = clf.best_estimator_
#     #__accuracy(best, data)
#
#
#     num_splits = sum(1 for key in clf.cv_results_.keys() if key.startswith("split") and key.endswith("_test_score"))
#     num_runs = len(clf.cv_results_['mean_test_score'])
#
#     # https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html
#     test_scores = [np.asarray([clf.cv_results_[f"split{split}_test_score"][run] for split in range(num_splits)])
#                    for run in range(num_runs)]
#     train_scores = [np.asarray([clf.cv_results_[f"split{split}_train_score"][run] for split in range(num_splits)])
#                     for run in range(num_runs)]
#     #train_scores = [[clf.cv_results_[f"split{split}_train_score"][run] for split in range(num_splits)] for run
#     #                in range(num_runs)]
#
#     #print(clf.cv_results_)
#
#     # def adjusted_r2(r2_scores_list):
#     #     return adjusted_r2_score(np.asarray(r2_scores_list), len(data.x.columns), len(data.x))
#
#     # Creating a list of tuples in 'alphabetical order'
#     #columns = OrderedDict()
#     columns = dict()
#
#     # Used to save cross-validation data for paper.
#     with open(f'reports/reg_data.csv', 'a') as f:
#         for i in range(len(test_scores[0])):
#             f.write(f'{type(regressor).__name__}, {test_scores[0][i]}, {train_scores[0][i]}\n')
#     f.close()
#
#     columns["n_samples"] = len(X)
#
#     for param_name in params[0]:
#         columns[param_name] = [param[param_name] for param in params]
#
#     sets = [("_train_", train_scores), ("_test_", test_scores)] if cv is not None else [("_", train_scores)]
#     for score_name, score_fn in [("r2", lambda x: x), ("r2adj", adjusted_r2)]:
#         if score_name not in display:
#             continue
#         for set_name, set_data in sets:
#             for metric in ["mean", "min", "max", "std"]:
#                 if metric not in display:
#                     continue
#                 columns[f"{metric}{set_name}{score_name}"] = [getattr(score_fn(entry), metric)() for entry in set_data]
#
#     if ext_dataset is not None:
#         print()
#         #columns["ext_mean_test_r2"] = r2_score(y_test, best.predict(X_test))
#     columns["mean_fit_time"] = clf.cv_results_['mean_fit_time'] # use for testing solvers
#     columns["estimator"] = best
#     return pd.DataFrame(data=columns), best
#
#
# def run_grid_search_clf(data, classifier, parameter_grid, cnd=None, cv=3, display = ["mean", "r2", "r2adj", "std"]):
#     X = data.x_scaled.values
#     # Classifier needs to have int type target values instead of scaled floats
#     y = data.y.values.astype('int64').ravel()
#
#     # # limit sample size to N
#     # N = 150
#     # idxs = np.random.permutation(len(X))[:N] # Select randomly
#     # X = X[idxs]
#     # y = y[idxs]
#
#     # Split to external validation set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
#
#     grid = ParameterGrid(parameter_grid)
#     df = pd.DataFrame()
#     for kparams in list(grid):
#         columns = dict()
#         clf = classifier if kparams == {} else classifier(**kparams)
#         #clf = exec(type(classifier).__name__)(**{**classifier.__dict__, **kparams})
#
#         clf_cv = cross_validate(clf, X_train, y_train, scoring='accuracy', cv=cv,
#                                 return_estimator=True, return_train_score=True)
#         internal_test_scores = clf_cv['test_score']
#         external_test_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clf_cv['estimator']]
#         best = clf_cv['estimator'][np.argmax(external_test_scores)]
#
#         # Get predictions based on best estimator from GridSearchCV for later calculations
#         y_pred = best.predict(X_test)
#
#         columns['n_samples'] = len(y)
#         #params = clf.cv_results_['params']
#         #columns['Best Params'] = clf.best_params_ # temporary fix for regressor_param_grid enumeration issue
#         for param_name, param_value in kparams.items():
#             columns[param_name] = param_value
#
#         mtrx = confusion_matrix(y_test, y_pred)
#
#         columns['mean_train_acc'] = sum(clf_cv['train_score'])/cv
#         columns['mean_test_acc'] = sum(clf_cv['test_score'])/cv
#         '''
#         Old (direct) calculation, new version gives a cross-validated average over cv runs
#         num_pred = len(y_train)
#         num_correct = np.sum(np.diagonal(confusion_matrix(y_train, clf.predict(X_train))))
#         columns['accuracy_train'] = num_correct / num_pred
#
#         num_pred = len(y_test) # same as np.sum(mtrx)
#         num_correct = np.sum(np.diagonal(mtrx))
#         columns['accuracy_test'] = num_correct / num_pred
#         '''
#
#         columns['Confusion Matrix'] = mtrx
#
#         if cnd is not None:
#             labels = list(np.unique(y_test))
#             cnd_index = labels.index(cnd) # Only needed for str type labels which are not currently implemented
#
#             # METRICS: https://en.wikipedia.org/wiki/Confusion_matrix (2 x 2 Reduction)
#             tp = mtrx[cnd_index][cnd_index]
#             fn = np.sum(mtrx[cnd_index, :]) - tp
#             fp = np.sum(mtrx[:, cnd_index]) - tp
#             tn = np.sum(np.diagonal(mtrx)) - tp
#
#             columns[f'Condition {cnd} Matrix'] = [[tp, fp], [fn, tn]]
#             if (fp + tp) != 0:
#                 columns['TPR'] = tp / (fn + tp) # AKA precision - use as metric when cost of fp is high
#                 columns['FNR'] = fn / (fn + tp) # When cnd is 'healthy', this is the (SIMA) FALSE ALARM rate (detecting health)
#             else:
#                 columns['TPR'] = np.nan#'No Positives'
#                 columns['FNR'] = np.nan#'-'
#             if (fn + tn) != 0:
#                 columns['FPR'] = fp / (fp + tn) # When cnd is 'healthy', this is the (SIMA) MISSED DECTECTION rate *MINIMIZE*
#                 columns['TNR'] = tn / (fp + tn)
#             else:
#                 columns['FPR'] = np.nan#'No Negatives'
#                 columns['TNR'] = np.nan#'-'
#             columns['recall'] = tp / (tp + fn) # use as metric when cost of fn is high
#             if columns['TPR'] != np.nan: #'No Positives'
#                 columns['F1'] = 2*columns['TPR']*columns['recall']/(columns['TPR'] + columns['recall'])
#         columns['estimator'] = best
#
#         columns = {key: {0: value} for key, value in columns.items()}
#         df = pd.concat([df, pd.DataFrame(data=columns)], sort=False)
#     return df, best