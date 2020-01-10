import os
import uuid
import warnings

import numpy as np
import pandas as pd
import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import ParameterGrid
from joblib import dump # Similar to pickle, optimized for objects with large internal numpy arrays


from utilities.common_functions import df_append, adjusted_r2_score, df_reorder_columns
from utilities.paramutils import iter_params
from utilities.plotutils import hist_df, param_grid_plot, graph_predicted_vs_actual

# Creates file in csv file in tmp folder that opens in excel
def open_in_excel(df):
    # uuid4(): Generate a random UUID for tmp file
    fname = str(uuid.uuid4()) + ".csv"
    df.to_csv(f"tmp/{fname}")
    os.system(f"mkdir -p tmp;cd tmp;open {fname}")

# Save data frame to csv file in reports
def save_df(df, save_name):
    df.to_csv(f"reports/{save_name}.csv", index=False)
    df.to_pickle(f"reports/{save_name}.pkl")

def execute(datasets,
            loader_params,
            loader_param_grid,
            regressors,
            regressor_param_grid,
            targets=None,
            ext_dataset=None,
            save_name=None,
            cv=5,
            multicore=True,
            write_validation=True,
            save_df_report=True,
            show_hist=False,
            extract_data_callback=None,
            include_regressor_and_mlpdata=False,
            # Classifier option in execute function
            eval_type='regressor',
            cnd=None, # Condition (index) to test for, e.g. for GLU_risk 'Healthy' := 1
            # Export (save to ./models/) best n evaluators for each sex and target option
            export_best=0,
            # Save report as spreadsheet and save scatter plot png
            report=None,
            save_png=False,
            show_png=False
            ):
    '''
        The original runner function accepts an instance of the DataSet class (:datasets),
    extracts groups of features (:loader_param_grid), applies specified scalars (:loader_params) and
    then performs regression or classification (:eval_type) on :targets using user-specified models (:regressors)
    and their potential hyper-parameters (:regressor_param_grid).
        The function may take :targets, :datasets and :regressors as lists. For each combination of these options, a
    grid search is performed by 'run_grid_search' on :loader_param_grid and :regressor_param_grid keys.
    '''
    # Conflicting input check
    write_validation = write_validation and save_name is not None
    save_df_report = save_df_report and save_name is not None
    save_png = save_png and report is not None

    # Resolve delayed statements with multicore processing
    def resolve_delayed(delayed_iter):
        delayed_list = list(delayed_iter)
        if multicore:
            return list(Parallel(n_jobs=-1)(delayed_list)) # Parallel - n_jobs = -1, use all cpu's
        else:
            return [delayed[0](*delayed[1], **delayed[2]) for delayed in delayed_list]

    # Warnings for current issues and bugs
    if regressor_param_grid != {} and isinstance(regressors, list):
        warnings.warn('Hyperparameter search is not supported for multi-regressor runs, removing regressor_param_grid.')
        regressor_param_grid = {}
    if regressor_param_grid != {} and eval_type == 'classifier' and not callable(regressors):
        warnings.warn('Classifier object is not callable, removing regressor_param_grid.')
        regressor_param_grid = {}
    if isinstance(datasets, list):
        warnings.warn('Multi-dataset run detected, please insure feature column names have been standardized.')

    # Support for old code
    regressors = regressors if isinstance(regressors, list) else [regressors]
    targets = targets if isinstance(targets, list) else [targets]
    datasets = datasets if isinstance(datasets, list) else [datasets]
    if eval_type == 'classifier':
        for target in targets:
            loader_params['scaler_config'][target] = None  # Don't standardize classifier columns

    full_df = pd.DataFrame()
    best_df = pd.DataFrame() # Needed to keep track of best runs if export_best > 0
    for target in targets:
        if targets != [None]:
            loader_params['label_cname'] = target
        else:
            target = loader_params['label_cname']

        for dataset in datasets:
            print(f"[EXTRAC] Extracting {type(dataset).__name__} for {eval_type} on {loader_params['label_cname']}")
            # extracts data and applies transformations from loader_params
            # param_map contains the arguments for extract data
            # each param_map is the same except for 'feature_cnames'
            param_maps, column_maps = zip(*iter_params(loader_param_grid, noniterative_params=loader_params))
            # extracted_data_list - data.x ('data') and data.y ('target') extracted from dataset for each 'segment'
            extracted_data_list = resolve_delayed(
                delayed(dataset.extract_data)(**param_map) for param_map in param_maps)

            if callable(extract_data_callback):
                extract_data_callback(extracted_data_list)

            for regressor in regressors:
                print(f"[STATUS] Running {type(regressor).__name__}")
                grid_search_results = []
                if write_validation:
                    f = open(f"reports/{save_name}-validation.txt", "w")
                for data, param_map, column_map in zip(extracted_data_list, param_maps, column_maps):
                    if show_hist:
                        hist_df(pd.concat([data.x, data.y], axis=1, sort=False), bins=15)
                        # hist_df(pd.concat([data.x_scaled, data.y_scaled], axis=1), bins=15)
                    if write_validation:
                        feature_cnames = param_map['feature_cnames']
                        cols = list(data.x.columns.values)
                        f.write(
                            f"column_map: {column_map}\nfeature_cnames: {feature_cnames}\n"
                            f"data.x.columns: {data.x.columns}\nparam_map: {param_map}\n\n\n")
                        f.flush()
                    if eval_type == 'regressor':
                        grid_search_results.append(
                            delayed(run_grid_search)(data, regressor, regressor_param_grid, ext_dataset=ext_dataset, cv=cv))
                    elif eval_type == 'classifier':
                        grid_search_results.append(
                            delayed(run_grid_search_clf)(data, regressor, regressor_param_grid, cnd=cnd, cv=cv))
                if write_validation:
                    f.close()

                grid_search_results = resolve_delayed(grid_search_results)  # Actually training the regressors here
                master_df = None
                for (df, best), column_map, data in zip(grid_search_results, column_maps, extracted_data_list):
                    # __accuracy(best, data)
                    if include_regressor_and_mlpdata:
                        df["best_regressor"] = best
                        df["mlp_data"] = data
                    # adds column_map.items() information (tree branch) to row
                    master_df = df_append(master_df, df, column_map)
                if save_df_report:
                    save_df(master_df, save_name)

                master_df = df_reorder_columns(master_df, send_back=list(grid_search_results[0][0].columns))
                # Prepended columns
                master_df.insert(0, 'dataset', type(dataset).__name__)
                master_df.insert(0, 'target', target)
                master_df.insert(0, 'regressor', type(regressor).__name__)

                if ext_dataset is not None:
                    external_data_list = resolve_delayed(
                        delayed(ext_dataset.extract_data)(**param_map) for param_map in param_maps)

                    ext_scores = []
                    for data, est in zip(external_data_list, master_df['estimator']):
                        X = data.x_scaled.values
                        y = data.y_scaled.values.ravel()
                        ext_scores += [sum(cross_validate(est, X, y, cv=cv)['test_score'])/cv]

                    master_df.insert(len(master_df.columns) - 2, 'external_r2', ext_scores)

                full_df = pd.concat([full_df, master_df], sort=False)

            # Exports best n evaluators for each sex option
            # Possible issue may occur when executing through many datasets
            # I DO NOT RECOMMEND USING THIS FOR CLASSIFICATION
            if export_best > 0:
                column_data = loader_param_grid["feature_cnames"]["__data__"]
                #print(len(column_data), len(extracted_data_list))
                sex_options = np.unique(full_df['SEX'])
                print(f'[EXPORT] Fitting {len(sex_options)*export_best} '
                      f'best {target} {eval_type}s on {type(dataset).__name__}')
                for sex in sex_options:
                    df = full_df.loc[(full_df['SEX'] == sex) & (full_df['target'] == target)]
                    df = df.nlargest(export_best, columns=['mean_test_r2']) if eval_type == 'regressor' \
                        else df.nlargest(export_best, columns=['TPR'])
                    for idx, row in df.iterrows():
                        best_run_dict = dict()
                        if eval_type == 'regressor':
                            scores = ['mean_train_r2', 'mean_test_r2']
                        elif eval_type == 'classifier' and cnd is None:
                            scores = ['mean_train_acc', 'mean_test_acc', 'Confusion Matrix']
                        else:
                            scores = ['mean_train_acc', 'mean_test_acc', 'Confusion Matrix', 'TPR', 'FNR', 'FPR', 'TNR', 'recall', 'F1']

                        row_data = ['regressor', 'target', 'dataset', 'SEX'] + \
                                   list(loader_param_grid["feature_cnames"].columns[:-1]) + ['n_samples'] + scores
                        for cname in row_data:
                            best_run_dict[cname] = row[cname]

                        # Use relative index for extracted_data_list containing parameter (column_data) and SEX columns
                        grid_len = loader_param_grid["feature_cnames"].shape[0] # Number of rows in parameter grid
                        data_idx = idx % (len(sex_options) * grid_len)
                        X = extracted_data_list[data_idx].x_scaled.values
                        y = extracted_data_list[data_idx].y_scaled.values.ravel() if eval_type == 'regressor' \
                            else extracted_data_list[data_idx].y.values.astype('int64').ravel()
                        reg = row['estimator'].fit(X, y)
                        if eval_type == 'regressor':
                            best_run_dict['total_train_r2'] = r2_score(y, reg.predict(X))
                        else:
                            best_run_dict['total_train_acc'] = accuracy_score(y, reg.predict(X))
                        best_run_dict['estimator'] = reg

                        best_run_dict['param_grid'] = {'SEX': sex,
                                                       **loader_param_grid["feature_cnames"].iloc[idx % grid_len, :-1]}
                        best_run_df = pd.DataFrame(data={key: {0: value} for key, value in best_run_dict.items()})
                        best_df = pd.concat([best_df, best_run_df])
        full_df.index = range(len(full_df.index))  # So the run numbers print correctly (reset_index() but actually works)

    if isinstance(report, str):
        if show_png or save_png:
            print(f"[REPORT] Writing parameter grid plot to ./reports/{report}.png" if save_png
                  else "[REPORT] Printing parameter grid plot to console")
            if eval_type == 'regressor':
                mask = (full_df['mean_train_r2'] >= 0) & (full_df['mean_train_r2'] <= 1) & \
                       (full_df['mean_test_r2'] >= 0) & (full_df['mean_test_r2'] <= 1)
                param_grid_plot(full_df.loc[mask],
                                [['target', 'dataset', 'regressor'],
                                 ['SEX'] + list(loader_param_grid['feature_cnames'].keys()[:-1]),
                                 list(regressor_param_grid.keys())],
                                ['mean_train_r2', 'mean_test_r2'],
                                save_name=report if save_png else None
                                ).show()
            elif eval_type == 'classifier':
                if cnd is None:
                    mask = (full_df['mean_test_acc'] >= 0) & (full_df['mean_test_acc'] <= 1) & \
                           (full_df['mean_train_acc'] >= 0) & (full_df['mean_train_acc'] <= 1)
                else:
                    mask = (full_df['FPR'] >= 0) & (full_df['FPR'] <= 1) & (full_df['TPR'] >= 0) & (full_df['TPR'] <= 1)
                param_grid_plot(full_df.fillna(420).loc[mask],
                                [['target', 'dataset', 'regressor'],
                                 ['SEX'] + list(loader_param_grid['feature_cnames'].keys()[:-1]),
                                 list(regressor_param_grid.keys())],
                                ['mean_train_acc', 'mean_test_acc'] if cnd is None else ['FPR', 'TPR'],
                                save_name=report if save_png else None
                                ).show()
        print(f"[REPORT] Writing results to ./reports/{report}.xlsx")
        full_df.to_excel(f"reports/{report}.xlsx", index=False)

    if export_best > 0:
        timestamp = int(datetime.datetime.now().strftime('%Y%m%d%H%M')[2:])
        best_df.index = range(len(best_df.index))
        print('[EXPORT] Best runs to be exported:')
        print(best_df)
        save = str(input('[EXPORT] Save evaluators to disk (y/[n])? '))
        if save == 'y':
            run_list = input('[EXPORT] List evaluators to save (space separated): ')
            for run_idx in run_list.split():
                run_data = best_df.iloc[int(run_idx)]
                print(run_data)
                save_name = f'models/{run_data["target"]}_{run_data["estimator"]}_{timestamp}-{run_idx}'
                dump(run_data["estimator"], f"{save_name}.joblib")
                if os.path.exists('models/models.txt'):
                    f = open("models/models.txt", "a+")  # Append to models.txt
                else:
                    f = open("models/models.txt", "w+")  # Write models.txt
                f.write(
                    f'Run {timestamp}-{run_idx}: {run_data["estimator"]} on ({run_data["sex"]}) {run_data["target"]}\n'
                    f'Input parameters: {run_data["param_grid"]}\n'
                    f'train/test: {run_data["mean_train_r2"]}/{run_data["mean_test_r2"]}\n'
                    f'total train: {run_data["total_train_r2"]}\n\n'
                )
                f.close()
    return full_df


def __accuracy(mlp, data):
    y_predict = data.predict(mlp)

    df = pd.DataFrame({"Actual": data.y.squeeze(), "Predicted": y_predict.squeeze()}, index=data.y.index)
    df["diff"] = (df["Actual"] - df["Predicted"]).abs()
    df = df.sort_values("diff", ascending=False)
    print(df)
    graph_predicted_vs_actual(df["Predicted"], df["Actual"])


def run_grid_search(data, regressor, parameter_grid, ext_dataset=None, cv=3, display=["mean", "r2", "std"]): # "r2adj",
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()

    '''
    # limit sample size to N
    N = 150
    #idxs = np.random.permutation(len(X))[:N] # Select randomly
    idxs = range(N)
    X = X[idxs]
    y = y[idxs]
    '''

    # Split to external validation set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = GridSearchCV(regressor,
                       parameter_grid,  # regressor_parameter_grid
                       n_jobs=-1,
                       return_train_score=True,
                       cv=cv if cv is not None else [
                           (np.asarray(range(0, len(data.x))), np.asarray(range(0, len(data.x))))],
                       iid=True
                       )
    clf.fit(X, y)

    params = clf.cv_results_['params']
    best = clf.best_estimator_
    #__accuracy(best, data)


    num_splits = sum(1 for key in clf.cv_results_.keys() if key.startswith("split") and key.endswith("_test_score"))
    num_runs = len(clf.cv_results_['mean_test_score'])

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html
    test_scores = [np.asarray([clf.cv_results_[f"split{split}_test_score"][run] for split in range(num_splits)])
                   for run in range(num_runs)]
    train_scores = [np.asarray([clf.cv_results_[f"split{split}_train_score"][run] for split in range(num_splits)])
                    for run in range(num_runs)]
    #train_scores = [[clf.cv_results_[f"split{split}_train_score"][run] for split in range(num_splits)] for run
    #                in range(num_runs)]

    #print(clf.cv_results_)

    def adjusted_r2(r2_scores_list):
        return adjusted_r2_score(np.asarray(r2_scores_list), len(data.x.columns), len(data.x))

    # Creating a list of tuples in 'alphabetical order'
    #columns = OrderedDict()
    columns = dict()

    # Used to save cross-validation data for paper.
    with open(f'reports/reg_data.csv', 'a') as f:
        for i in range(len(test_scores[0])):
            f.write(f'{type(regressor).__name__}, {test_scores[0][i]}, {train_scores[0][i]}\n')
    f.close()

    columns["n_samples"] = len(X)

    for param_name in params[0]:
        columns[param_name] = [param[param_name] for param in params]

    sets = [("_train_", train_scores), ("_test_", test_scores)] if cv is not None else [("_", train_scores)]
    for score_name, score_fn in [("r2", lambda x: x), ("r2adj", adjusted_r2)]:
        if score_name not in display:
            continue
        for set_name, set_data in sets:
            for metric in ["mean", "min", "max", "std"]:
                if metric not in display:
                    continue
                columns[f"{metric}{set_name}{score_name}"] = [getattr(score_fn(entry), metric)() for entry in set_data]

    if ext_dataset is not None:
        print()
        #columns["ext_mean_test_r2"] = r2_score(y_test, best.predict(X_test))
    columns["mean_fit_time"] = clf.cv_results_['mean_fit_time'] # use for testing solvers
    columns["estimator"] = best
    return pd.DataFrame(data=columns), best


def run_grid_search_clf(data, classifier, parameter_grid, cnd=None, cv=3, display = ["mean", "r2", "r2adj", "std"]):
    X = data.x_scaled.values
    # Classifier needs to have int type target values instead of scaled floats
    y = data.y.values.astype('int64').ravel()

    '''
    # limit sample size to N
    N = 150
    #idxs = np.random.permutation(len(X))[:N] # Select randomly
    idxs = range(N)
    X = X[idxs]
    y = y[idxs]
    '''

    # Split to external validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    grid = ParameterGrid(parameter_grid)
    df = pd.DataFrame()
    for kparams in list(grid):
        columns = dict()
        clf = classifier if kparams == {} else classifier(**kparams)
        #clf = exec(type(classifier).__name__)(**{**classifier.__dict__, **kparams})

        clf_cv = cross_validate(clf, X_train, y_train, scoring='accuracy', cv=cv,
                                return_estimator=True, return_train_score=True)
        internal_test_scores = clf_cv['test_score']
        external_test_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clf_cv['estimator']]
        best = clf_cv['estimator'][np.argmax(external_test_scores)]

        # Get predictions based on best estimator from GridSearchCV for later calculations
        y_pred = best.predict(X_test)

        columns['n_samples'] = len(y)
        #params = clf.cv_results_['params']
        #columns['Best Params'] = clf.best_params_ # temporary fix for regressor_param_grid enumeration issue
        for param_name, param_value in kparams.items():
            columns[param_name] = param_value

        mtrx = confusion_matrix(y_test, y_pred)

        columns['mean_train_acc'] = sum(clf_cv['train_score'])/cv
        columns['mean_test_acc'] = sum(clf_cv['test_score'])/cv
        '''
        Old (direct) calculation, new version gives a cross-validated average over cv runs
        num_pred = len(y_train)
        num_correct = np.sum(np.diagonal(confusion_matrix(y_train, clf.predict(X_train))))
        columns['accuracy_train'] = num_correct / num_pred

        num_pred = len(y_test) # same as np.sum(mtrx)
        num_correct = np.sum(np.diagonal(mtrx))
        columns['accuracy_test'] = num_correct / num_pred
        '''

        columns['Confusion Matrix'] = mtrx

        if cnd is not None:
            labels = list(np.unique(y_test))
            cnd_index = labels.index(cnd) # Only needed for str type labels which are not currently implemented

            # METRICS: https://en.wikipedia.org/wiki/Confusion_matrix (2 x 2 Reduction)
            tp = mtrx[cnd_index][cnd_index]
            fn = np.sum(mtrx[cnd_index, :]) - tp
            fp = np.sum(mtrx[:, cnd_index]) - tp
            tn = np.sum(np.diagonal(mtrx)) - tp

            columns[f'Condition {cnd} Matrix'] = [[tp, fp], [fn, tn]]
            if (fp + tp) != 0:
                columns['TPR'] = tp / (fn + tp) # AKA precision - use as metric when cost of fp is high
                columns['FNR'] = fn / (fn + tp) # When cnd is 'healthy', this is the (SIMA) FALSE ALARM rate (detecting health)
            else:
                columns['TPR'] = np.nan#'No Positives'
                columns['FNR'] = np.nan#'-'
            if (fn + tn) != 0:
                columns['FPR'] = fp / (fp + tn) # When cnd is 'healthy', this is the (SIMA) MISSED DECTECTION rate *MINIMIZE*
                columns['TNR'] = tn / (fp + tn)
            else:
                columns['FPR'] = np.nan#'No Negatives'
                columns['TNR'] = np.nan#'-'
            columns['recall'] = tp / (tp + fn) # use as metric when cost of fn is high
            if columns['TPR'] != np.nan: #'No Positives'
                columns['F1'] = 2*columns['TPR']*columns['recall']/(columns['TPR'] + columns['recall'])
        columns['estimator'] = best

        columns = {key: {0: value} for key, value in columns.items()}
        df = pd.concat([df, pd.DataFrame(data=columns)], sort=False)
    return df, best


def gen_layers(start, end, step=1, levels=1):
    layers = []
    for i in range(start, end + 1, step):
        layers.append((i,))
        if levels > 1:
            for j in range(start, end + 1, step):
                layers.append((i, j))
            if levels > 2:
                for k in range(start, end + 1, step):
                    layers.append((i, j, k))
    return layers