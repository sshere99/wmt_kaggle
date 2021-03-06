import process_weather as pw
import feature_eng as fe
import score_funcs as sf
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn import grid_search


def main():

    print("Reading in already processed weather file.....")
    weather = pd.read_csv('../wmt_data/weather_processed.csv')
    print("Columns in weather file are {}\n\n\ Dropping Some Columns..".format(weather.columns))

    # Columns not to use for this run

    # Get training data merged with weather data
    train = merge_train_weather(weather)

    # Get test data merged with weather data
    test = merge_test_weather(weather)

    #items = [15, 45, 16, 37, 9]
    #items = [8, 9, 98, 95, 87, 81, 78, 67, 63, 13, 17, 24, 28, 35, 38, 41, 48, 58, 68, 70, 74]
    items = range(37, 112)     # items with worst score [93, 9, 15, 45, 16, 37]
    #items = [1]

    for item_nbr in items:
        train_item = train[train['item_nbr'] == item_nbr]
        test_item = test[test['item_nbr'] == item_nbr]

        print('running for item {0}, score to beat {1:.03f}'.format(item_nbr, best_scores_dict[item_nbr]))

        # Select which strategy to use
        if run_code == 'BASE_GRID':
            run_model_by_item_gridsearch(train_item, test_item, item_nbr)

        elif run_code == 'FEAT_GRID':
            run_feature_gridsearch(train_item, test_item, item_nbr)

        elif run_code == 'BY_STORE':
            all_labels_store = np.array([])
            all_preds_store = np.array([])
            for j in range(45):
                store_num = j+1
                train_item_store = train_item[train_item['store_nbr'] == store_num]
                test_item_store = test_item[test_item['store_nbr'] == store_num]

                Y_dev, X_dev_pred = run_model_by_item_and_store(train_item_store, test_item_store, item_nbr, store_num)
                all_labels_store = np.hstack((all_labels_store,Y_dev))
                all_preds_store = np.hstack((all_preds_store,X_dev_pred))

            item_score = sf.rmsle(all_labels_store, all_preds_store)
            print("RMSLE for item {} = {}".format(item_nbr, item_score))
            if item_score < best_scores_dict[item_nbr]:
                # update_new_best_score(item_nbr, item_score)
                print("FOUND NEW BEST SCORE!")

        else:
            print("***** INVALID RUN CODE ******")

    print("COMPLETED ENTIRE DATA SET")

    return


def merge_train_weather(weather):

    print("Reading in training file.....")
    train = pd.read_csv('../wmt_data/train/train_with_stn.csv')
    train = train.set_index('date', drop = False)
    train.drop('Unnamed: 0', axis=1, inplace=True)
    weather = weather.set_index('date', drop = False)

    print('Merging train with weather.....')
    train_merged = pd.merge(train, weather, left_on=['date', 'station_nbr'], right_on=['date', 'station_nbr'], how='left')

    print('Dropping unneccesary columns from train.....')
    train_merged.drop('date', axis=1, inplace=True)
    train_merged.drop('station_nbr', axis=1, inplace=True)

    print("shape = {}".format(train_merged.shape))
    print('Removing outliers from item 5.....')
    train_merged.drop(train_merged[(train_merged.item_nbr == 5) & (train_merged.units > 800)].index, inplace=True)
    print("shape = {}".format(train_merged.shape))

    return train_merged


def merge_test_weather(weather):

    print("Reading in test file.....")
    test = pd.read_csv('../wmt_data/test/test_with_stn.csv')
    test = test.set_index('date', drop = False)
    test.drop('Unnamed: 0', axis=1, inplace=True)
    weather = weather.set_index('date', drop = False)

    print('Merging test with weather.....')
    test_merged = pd.merge(test, weather, left_on=['date', 'station_nbr'], right_on=['date', 'station_nbr'], how='left')

    print('Dropping unneccesary columns from test.....')
    test_merged.drop('station_nbr', axis=1, inplace=True)

    return test_merged


def run_model_by_item_and_store(train, test, item_num, store_num):

    X, Y = convert_to_numpy(train, False)  # 2nd argument indicates that model is not being run just at item level
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.25, random_state=42)

    clf = RandomForestRegressor(n_estimators=20, min_samples_split=100, min_samples_leaf=50)

    # Fit model
    clf.fit(X_train, Y_train)
    #clf.fit(X, Y)
    dates = test['date']
    stores = test['store_nbr']
    test.drop('date', axis=1, inplace=True)
    if int(store_num) != 35:

        # write test predictions
        with open('../wmt_data/submissions/by_store/itemstore_{}_{}.csv'.format(item_num, store_num), 'w') as f:

            predict_test_and_write(clf, test, item_num, dates, stores, f, False) # Last argument = run at itemlevel

        f.close()

    X_dev_pred = clf.predict(X_dev).clip(0)

    return Y_dev, X_dev_pred


def run_model_by_item_gridsearch(train, test, item_num):

    X, Y = convert_to_numpy(train, True) # 2nd argument is true because running by item and not store
    n_features = int(X.shape[1])

    # Grid search parameters
    #parameters = {'min_samples_split': [100], 'min_samples_leaf': [1, 50], 'n_estimators': [10]}
    parameters = {'min_samples_split': [5, 100], 'min_samples_leaf': [5, 50]}

    my_scorer = make_scorer(sf.rmsle, greater_is_better=False)

    rf = RandomForestRegressor(n_estimators=15)
    clf = grid_search.GridSearchCV(rf, parameters, cv=7, scoring=my_scorer)
    clf.fit(X, Y)

    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    best_scr = -clf.best_score_
    print("best score and params for item {} = {} and {}".format(item_num, best_scr, clf.best_params_))
    # print("feature importance: {}".format(clf.best_estimator_.feature_importances_))
    if best_scr < best_scores_dict[item_num]:
        update_new_best_score(item_num, best_scr)

        # write test predictions
        with open('../wmt_data/submissions/by_item/item_{}.csv'.format(item_num), 'w') as f:

            dates = test['date']
            stores = test['store_nbr']
            test.drop('date', axis=1, inplace=True)
            predict_test_and_write(clf.best_estimator_, test, item_num, dates, stores, f, True) # Last argument = run at itemlevel

        f.close()

    return


def run_feature_gridsearch(train, test, item_num):

    # Get various training sets with different features to test scores
    train_set = fe.get_feature_sets(train)
    test_set = fe.get_feature_sets(test)

    best_score = -1000
    best_clf = None
    best_combo = None

    for key, val in train_set.iteritems():

        this_train = val  # get a given training set
        X, Y = convert_to_numpy(this_train, True)

        parameters = {'min_samples_split': [100, 500]}
        my_scorer = make_scorer(sf.rmsle, greater_is_better=False)

        rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=50)
        clf = grid_search.GridSearchCV(rf, parameters, cv=7, scoring=my_scorer)
        clf.fit(X, Y)

        for params, mean_score, scores in clf.grid_scores_:
            print("{0:.3f} +/- {1:.03f} for combo {2}".format(mean_score, scores.std() * 2, key))
            if clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_clf = clf.best_estimator_
                best_combo = key

    print("Best combo is {} with score of {}".format(best_combo, best_score))
    best_score_sign_adjusted = -best_score
    if best_score_sign_adjusted < best_scores_dict[item_num]:
        update_new_best_score(item_num, best_score_sign_adjusted)

        # write test predictions
        with open('../wmt_data/submissions/by_item/item_{}.csv'.format(item_num), 'w') as f:

            test_best = test_set[best_combo]
            dates = test_best['date']
            stores = test_best['store_nbr']
            test_best.drop('date', axis=1, inplace=True)
            predict_test_and_write(best_clf, test_best, item_num, dates, stores, f, True)

        f.close()


    return


def convert_to_numpy(train, item_flg):
    y = train['units']   # Get labels
    train.drop('units', axis=1, inplace=True)   # Drop labels from X
    Y = np.array(y)
    X_scaled = process_x_data(train, True, item_flg)   ## process some more and turn into numpy

    return X_scaled, Y


def process_x_data(x_data, train_flag, item_flag):  ## train flag = True if sending train data
                                                    # item flag = False if running store level models

    if item_flag:
        x_data = create_store_dummies(x_data, train_flag)

    x_data.drop('date.1', axis=1, inplace=True)
    # print("TRAIN FLAG: {} Columns for model on are: {}".format(train_flag, list(x_data.columns)))
    x_data = add_all_interactions(x_data)
    X = x_data.as_matrix()
    imputer = Imputer()
    X = imputer.fit_transform(X)   # Fill in mean for missing NaN values

    return X


def add_all_interactions(x_data):

    base_interactions = {'b1': ['weekend', 'weather_event'],
                         'b2': ['weekend', 'tmax'],
                         'b3': ['weekend', 'snowfall'],
                         'b4': ['weekend', 'db_xmaseve'],
                         'b5': ['weather_event', 'dbthanks'],
                         'b6': ['dblab_day', 'weather_event']}

    x_data = create_base_interactions(x_data, base_interactions)

    vars_to_interact_with_store = ['tmax', 'preciptotal', 'weekofyear', 'weekend', 'date_int',
                                   'y2012', 'y2013', 'y2014', 'weather_event', 'b1', 'b2', 'b3', 'b4',
                                   'b5', 'b6']
    for var in vars_to_interact_with_store:
        x_data = add_store_interaction(x_data, var)

    return x_data


def create_base_interactions(x_data, interactions):

    for key, val in interactions.items():
        x_data[key] = x_data[val[0]] * x_data[val[1]]

    return x_data


def add_store_interaction(x_data, varname):

    new_var_name = varname+"_x_store_"
    for i in range(1,45):
        if i != 35:
            new_col = new_var_name + str(i)
            x_data[new_col] = x_data[i] * x_data[varname]

    return x_data

def create_store_dummies(x_data, train_flag):

    store_dummies = pd.get_dummies(x_data['store_nbr'])
    x_data = pd.concat([x_data, store_dummies], axis=1)
    x_data['35_new'] = 0

    # Deal with missing store number 35 from test data
    if train_flag:
        x_data['35_new'] = x_data[35]
        x_data.drop(35, axis=1, inplace=True)

    x_data.drop('store_nbr', axis=1, inplace=True)

    return x_data


def predict_test_and_write(model, test_data, item_num, dates, stores, f, item_flg):
    test = process_x_data(test_data, False, item_flg)  # process and turn into numpy array
    y_pred = model.predict(test).clip(0)
    write_submission(stores, dates, item_num, y_pred, f)

    return


def write_submission(stores, dates, item, preds, f):

    for store, date, units in zip(stores, dates, preds):
        id_ = (str(store) + "_" + str(item) + "_" + str(date))
        f.write('%s,%f\n' % (id_, units))


def update_new_best_score(item_num, score):

    print("Found new low score for this item")
    best_scores_dict[item_num] = score

    myfile = open('../wmt_data/best_scores.csv', 'wb')
    writer = csv.writer(myfile)
    writer.writerow(["item","RMSE"])
    for key, value in best_scores_dict.items():
        writer.writerow([key, value])
    myfile.close()

    return

if __name__ == "__main__":
    pd.options.display.max_columns = 82

    # Global variables
    run_code = 'BASE_GRID'   # BASE_GRID, FEAT_GRID, BY_STORE
    best_scores = pd.read_csv('../wmt_data/best_scores.csv')   # Load the existing best scores by item number
    best_scores_dict = best_scores.set_index('item')['RMSE'].to_dict()
    main()