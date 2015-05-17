import pandas as pd
import numpy as np
import process_weather as pw
import score_funcs as sf
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


def main():

    # Flag to indicate if we should create a seperate model for each item, process weather or not
    run_by_item = True
    process_weather = False

    if process_weather:
        print("Reading in unprocessed weather file.....")
        weather = pd.read_csv('../wmt_data/weather.csv')
        # arguments for function below -- add_wknd, add_trail_lead, drop_other_data, add_holidays
        weather = pw.process_weather_file(weather, True, False, True, True)
        weather.to_csv('../wmt_data/weather_processed.csv')

    else:
        print("Reading in already processed weather file.....")
        weather = pd.read_csv('../wmt_data/weather_processed.csv')

    # Get training data merged with weather data
    train = merge_train_weather(weather)

    # Get test data merged with weather data
    test = merge_test_weather(weather)

    sub_number = 12
    all_labels = np.array([])
    all_preds = np.array([])

    with open('../wmt_data/submissions/submission{}.csv'.format(sub_number), 'w') as f:
        f.write('id,units\n')

        if run_by_item:   # Create a new model for each item
            for i in range(111):
                item_nbr = i+1
                train_item = train[train['item_nbr'] == item_nbr]
                test_item = test[test['item_nbr'] == item_nbr]

                print('running model for item {}'.format(item_nbr))
                # Y_dev, X_dev_pred = run_model_by_item(train_item, test_item, item_nbr, f)
                run_model_by_item_gridsearch(train_item, test_item, item_nbr, f)
                #all_labels = np.hstack((all_labels,Y_dev))
                #all_preds = np.hstack((all_preds,X_dev_pred))
            #print("Overall RMSLE = {}".format(sf.rmsle(all_labels, all_preds)))

        else:   # Just create one model for all items
            print('Preparing to build model on entire dataset')

        f.close()


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


def run_model_by_item(train, test, item_num, f):

    X, Y = convert_to_numpy(train)
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.25, random_state=42)

    #clf = linear_model.LinearRegression()
    #clf = SVR(kernel='linear')
    #clf = DecisionTreeRegressor(random_state=0)
    #clf = linear_model.SGDRegressor(learning_rate='constant', eta0=0.00001)
    clf = RandomForestRegressor(n_estimators=20, min_samples_split=100)

    # Fit model
    clf.fit(X_train, Y_train)
    dates = test['date']
    stores = test['store_nbr']
    test.drop('date', axis=1, inplace=True)
    y_pred = predict_test_and_write(clf, test, item_num, dates, stores, f)

    X_dev_pred = clf.predict(X_dev).clip(0)
    print("RMSLE for item {} = {}".format(item_num, sf.rmsle(Y_dev, X_dev_pred)))

    return Y_dev, X_dev_pred


def run_model_by_item_gridsearch(train, test, item_num, f):

    X, Y = convert_to_numpy(train)
    n_features = int(X.shape[1])

    # Grid search parameters
    parameters = {'min_samples_split': [100, 500], 'min_samples_leaf': [1, 50, 100]}

    my_scorer = make_scorer(sf.rmsle, greater_is_better=False)

    rf = RandomForestRegressor(n_estimators=20)
    clf = grid_search.GridSearchCV(rf, parameters, cv=7, scoring=my_scorer)
    clf.fit(X, Y)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("\n Grid scores on development set:\n")

    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    dates = test['date']
    stores = test['store_nbr']
    test.drop('date', axis=1, inplace=True)
    y_pred = predict_test_and_write(clf.best_estimator_, test, item_num, dates, stores, f)

    return


def convert_to_numpy(train):
    y = train['units']   # Get labels
    train.drop('units', axis=1, inplace=True)   # Drop labels from X
    Y = np.array(y)
    X_scaled = process_x_data(train, True)   ## process some more and turn into numpy

    return X_scaled, Y


def process_x_data(x_data, train_flag):  ## train flag = True if sending train data, due to special processing requirements
    store_dummies = pd.get_dummies(x_data['store_nbr'])
    x_new = pd.concat([x_data,store_dummies], axis=1)
    x_new['35_new'] = 0

    # Deal with missing store number 35 from test data
    if train_flag:
        x_new['35_new'] = x_new[35]
        x_new.drop(35, axis=1, inplace=True)

    x_new.drop('Unnamed: 0', axis=1, inplace=True)
    x_new.drop('store_nbr', axis=1, inplace=True)
    X = x_new.as_matrix()
    imputer = Imputer()
    X = imputer.fit_transform(X)   # Fill in mean for missing NaN values

    return X


def predict_test_and_write(model, test_data, item_num, dates, stores, f):
    test = process_x_data(test_data, False)  # process and turn into numpy array
    y_pred = model.predict(test).clip(0)
    write_submission(stores, dates, item_num, y_pred, f)

    return y_pred


def write_submission(stores, dates, item, preds, f):

    for store, date, units in zip(stores, dates, preds):
        id_ = (str(store) + "_" + str(item) + "_" + str(date))
        f.write('%s,%f\n' % (id_, units))


if __name__ == "__main__":
    pd.options.display.max_columns = 82
    main()