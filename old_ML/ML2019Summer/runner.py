import os
import uuid
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix



from archive.plot_playground import graph_predicted_vs_actual
from common import df_append, adjusted_r2_score, df_reorder_columns
from paramutils import iter_params
from plotutils import hist_df


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
            save_name=None,
            cv=5,
            multicore=True,
            write_validation=True,
            save_df_report=True,
            show_hist=False,
            extract_data_callback=None,
            include_regressor_and_mlpdata=False,

            # NEW: Classifier option in execute function - needs revision
            eval_type='regressor',
            cnd=None
            ):
    # conflicting input check
    write_validation = write_validation and save_name is not None
    save_df_report = save_df_report and save_name is not None

    def resolve_delayed(delayed_iter):
        delayed_list = list(delayed_iter)
        if multicore:
            # Parallel - n_jobs = -1, use all cpu's
            return list(Parallel(n_jobs=-1)(delayed_list))
        else:
            return [delayed[0](*delayed[1], **delayed[2]) for delayed in delayed_list]

    regressors = [regressors] if type(regressors).__name__ != 'list' else regressors
    targets = [targets] if type(targets).__name__ != 'list' else targets
    datasets = [datasets] if type(datasets).__name__ != 'list' else datasets
    full_df = pd.DataFrame()

    for regressor in regressors:
        for target in targets:
            if targets != [None]:
                loader_params['label_cname'] = target
            else:
                target = loader_params['label_cname']


            print(f"[STATUS] Running {type(regressor).__name__} on {loader_params['label_cname']}")

            for dataset in datasets:
                print(f"[STATUS] Extracting {type(dataset).__name__}")
                param_maps, column_maps = zip(*iter_params(loader_param_grid, additional_params=loader_params))

                # extracted_data_list - data.x ('data') and data.y ('target') extracted from dataset for each 'segment'
                extracted_data_list = resolve_delayed(
                    delayed(dataset.extract_data)(**param_map) for param_map in param_maps)

                # extracts data and applies transformations from loader_params
                # param_map contains the arguments for extract data
                # each param_map is the same except for 'feature_cnames'
                if callable(extract_data_callback):
                    extract_data_callback(extracted_data_list)

                grid_search_results = []
                if write_validation:
                    f = open(f"reports/{save_name}-validation.txt", "w")
                print(f"[STATUS] Training and validating {type(regressor).__name__} {eval_type}")
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
                            delayed(run_grid_search)(data, regressor, regressor_param_grid, cv=cv))
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
                    #print(column_map)
                    master_df = df_append(master_df, df, column_map)
                if save_df_report:
                    save_df(master_df, save_name)

                master_df = df_reorder_columns(master_df, send_back=list(grid_search_results[0][0].columns))
                # Prepended columns
                master_df.insert(0, 'dataset', type(dataset).__name__)
                master_df.insert(0, 'target', target)
                master_df.insert(0, 'regressor', type(regressor).__name__)
                # Appended columns
                master_df['regressor full'] = regressor

                full_df = pd.concat([full_df, master_df], sort=False)
    return full_df


def __accuracy(mlp, data):
    y_predict = data.predict(mlp)

    df = pd.DataFrame({"Actual": data.y.squeeze(), "Predicted": y_predict.squeeze()}, index=data.y.index)
    df["diff"] = (df["Actual"] - df["Predicted"]).abs()
    df = df.sort_values("diff", ascending=False)
    print(df)
    graph_predicted_vs_actual(df["Predicted"], df["Actual"])


def run_grid_search(data, regressor, parameter_grid, cv=3, display=["mean", "r2", "r2adj", "std"]):
    clf = GridSearchCV(regressor,
                       parameter_grid,  # regressor_parameter_grid
                       n_jobs=-1,  iid=True,
                       return_train_score=True,
                       cv=cv if cv is not None else [
                           (np.asarray(range(0, len(data.x))), np.asarray(range(0, len(data.x))))]
                       )
    clf.fit(data.x_scaled, data.y_scaled.values.ravel())

    # save_df(pd.concat([data.x, data.y], axis=1), "grid")
    # save_df(pd.concat([data.x_scaled, data.y_scaled], axis=1), "grid_scaled")

    params = clf.cv_results_['params']
    best = clf.best_estimator_
    # __accuracy(best, data)


    num_splits = sum(1 for key in clf.cv_results_.keys() if key.startswith("split") and key.endswith("_test_score"))
    num_runs = len(clf.cv_results_['mean_test_score'])

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html
    test_scores = [np.asarray([clf.cv_results_[f"split{split}_test_score"][run] for split in range(num_splits)])
                   for run in range(num_runs)]
    train_scores = [np.asarray([clf.cv_results_[f"split{split}_train_score"][run] for split in range(num_splits)])
                    for run in range(num_runs)]
    # train_scores = [[clf.cv_results_[f"split{split}_train_score"][run] for split in range(num_splits)] for run
    #                 in range(num_runs)]

    #print(clf.cv_results_)

    def adjusted_r2(r2_scores_list):
        return adjusted_r2_score(np.asarray(r2_scores_list), len(data.x.columns), len(data.x))


    # Creating a list of tuples in 'alphabetical order'
    #columns = OrderedDict()
    columns = dict()

    columns["n_samples"] = len(data.x)

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

    columns["mean_fit_time"] = clf.cv_results_['mean_fit_time'] # use for testing solvers
    return pd.DataFrame(data=columns), best


def run_grid_search_clf(data, classifier, parameter_grid, cnd=None, cv=3, display = ["mean", "r2", "r2adj", "std"]):
    # TODO: Print multiple rows for regressor_param_grid entries. May need to do separate gridsearchcv.
    X = data.x_scaled.values
    # Classifier needs to have int type target values instead of scaled floats
    y = data.y.values.astype('int64').ravel()

    # Split to external validation set before GridSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    clf = GridSearchCV(classifier,
                       parameter_grid,
                       n_jobs=-1,
                       return_train_score=True,
                       cv=cv if cv is not None else [
                           (np.asarray(range(0, len(data.x))), np.asarray(range(0, len(data.x))))],
                       scoring='accuracy'
                       , iid=True)
    clf.fit(X_train, y_train)

    # Get predictions based on best estimator from GridSearchCV for later calculations
    best = clf.best_estimator_
    y_pred = best.predict(X_test)

    columns = dict()
    columns['samples'] = len(y)
    params = clf.cv_results_['params']
    #columns['Best Params'] = clf.best_params_ # temporary fix for regressor_param_grid enumeration issue
    for param_name in params[0]:
        columns[param_name] = [param[param_name] for param in params]

    mtrx = confusion_matrix(y_test, y_pred)

    num_pred = len(y_train)
    num_correct = np.sum(np.diagonal(confusion_matrix(y_train, best.predict(X_train))))
    columns['accuracy_train'] = num_correct / num_pred

    num_pred = len(y_test) # same as np.sum(mtrx)
    num_correct = np.sum(np.diagonal(mtrx))
    columns['accuracy_test'] = num_correct / num_pred

    columns['Confusion Matrix'] = mtrx

    if cnd is not None:
        columns['cnd'] = cnd
        labels = list(np.unique(y_test))
        cnd_index = labels.index(cnd) # Only needed for str type labels which are not currently implemented

        # METRICS: https://en.wikipedia.org/wiki/Confusion_matrix (2 x 2 Reduction)
        tp = mtrx[cnd_index][cnd_index]
        fn = np.sum(mtrx[cnd_index, :]) - tp
        fp = np.sum(mtrx[:, cnd_index]) - tp
        tn = np.sum(np.diagonal(mtrx)) - tp

        columns[f'Condition {cnd} Matrix'] = [[tn, fp], [fn, tp]]
        if (fp + tp) != 0:
            columns['TPR'] = tp / (fp + tp) # precision
            columns['FNR'] = fn / (fp + tp)
        else:
            columns['TPR'] = 'No Positives'
            columns['FNR'] = '-'
        if (fn + tn) != 0:
            columns['FPR'] = fp / (fn + tn)
            columns['TNR'] = tn / (fn + tn)
        else:
            columns['FPR'] = 'No Negatives'
            columns['TNR'] = '-'
        columns['recall'] = tp / (tp + fn)
        if columns['TPR'] != 'No Positives':
            columns['F1'] = 2*columns['TPR']*columns['recall'] / (columns['TPR'] + columns['recall'])

    columns = {key: {0: value} for key, value in columns.items()}
    return pd.DataFrame(data=columns), best


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