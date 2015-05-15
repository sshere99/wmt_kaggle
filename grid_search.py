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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


def run_tests():

    process_weather = False

    if process_weather:
        print("Reading in unprocessed weather file.....")
        weather = pd.read_csv('../wmt_data/weather.csv')
        weather = pw.process_weather_file(weather, True, False, True, True)

    else:
        print("Reading in already processed weather file.....")
        weather = pd.read_csv('../wmt_data/weather_processed.csv')

    # Get training data merged with weather data
    train = merge_train_weather(weather)

    all_labels = np.array([])
    all_preds = np.array([])

    for i in range(11):
        item_nbr = i+1
        train_item = train[train['item_nbr'] == item_nbr]
        Y_dev, X_dev_pred = run_model_by_item(train_item, item_nbr)
        all_labels = np.hstack((all_labels,Y_dev))
        all_preds = np.hstack((all_preds,X_dev_pred))
    print("Overall RMSLE = {}".format(sf.rmsle(all_labels, all_preds)))


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


def run_model_by_item(train, item_num):

    X, Y = convert_to_numpy(train)
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.25, random_state=42)

    clf = RandomForestRegressor(n_estimators=10)

    # Fit model
    clf.fit(X_train, Y_train)
    X_dev_pred = clf.predict(X_dev).clip(0)
    print("RMSLE for item {} = {}".format(item_num, sf.rmsle(Y_dev, X_dev_pred)))

    return Y_dev, X_dev_pred


def convert_to_numpy(train):
    y = train['units']   # Get labels
    train.drop('units', axis=1, inplace=True)   # Drop labels from X
    Y = np.array(y)
    X_scaled = process_x_data(train, True)   ## process some more and turn into numpy
    #X_scaled = MinMaxScaler().fit_transform(X_scaled, Y)
    #X_scaled = SelectKBest(chi2, k=30).fit_transform(X_scaled, Y)
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
    print("Columns to run the model on are: {}".format(list(x_new.columns)))
    X = x_new.as_matrix()
    imputer = Imputer()
    X = imputer.fit_transform(X)   # Fill in mean for missing NaN values

    return X

if __name__ == "__main__":
    pd.options.display.max_columns = 82
    run_tests()