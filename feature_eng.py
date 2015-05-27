__author__ = 'sheraz'


def get_feature_sets(train):

    train_set = {}
    # 11 = months, years

    # 00
    temp = train.copy()
    temp = drop_years(temp)
    temp = drop_months(temp)
    train_set['00'] = temp.copy()

    # 01
    temp = train.copy()
    temp = drop_months(temp)
    train_set['01'] = temp.copy()

    # 10
    temp = train.copy()
    temp = drop_years(temp)
    train_set['10'] = temp.copy()

    # 11
    train_set['11'] = train.copy()

    return train_set


def drop_months(temp):

    temp.drop('jan', axis=1, inplace=True)
    temp.drop('feb', axis=1, inplace=True)
    temp.drop('mar', axis=1, inplace=True)
    temp.drop('apr', axis=1, inplace=True)
    temp.drop('may', axis=1, inplace=True)
    temp.drop('jun', axis=1, inplace=True)
    temp.drop('jul', axis=1, inplace=True)
    temp.drop('aug', axis=1, inplace=True)
    temp.drop('sep', axis=1, inplace=True)
    temp.drop('oct', axis=1, inplace=True)
    temp.drop('nov', axis=1, inplace=True)
    temp.drop('dec', axis=1, inplace=True)

    return temp


def drop_years(temp):

    temp.drop('y2012', axis=1, inplace=True)
    temp.drop('y2013', axis=1, inplace=True)
    temp.drop('y2014', axis=1, inplace=True)

    return temp


def get_feature_sets_old(train):

    train_set = {}
    # 111 = weekofyear, weather_event, date_int

    # 000
    temp = train.copy()
    temp.drop('weekofyear', axis=1, inplace=True)
    temp.drop('weather_event', axis=1, inplace=True)
    temp.drop('date_int', axis=1, inplace=True)
    train_set['000'] = temp.copy()

    # 001
    temp = train.copy()
    temp.drop('weekofyear', axis=1, inplace=True)
    temp.drop('weather_event', axis=1, inplace=True)
    train_set['001'] = temp.copy()

    # 010
    temp = train.copy()
    temp.drop('weekofyear', axis=1, inplace=True)
    temp.drop('date_int', axis=1, inplace=True)
    train_set['010'] = temp.copy()

    # 011
    temp = train.copy()
    temp.drop('weekofyear', axis=1, inplace=True)
    train_set['011'] = temp.copy()

    # 100
    temp = train.copy()
    temp.drop('weather_event', axis=1, inplace=True)
    temp.drop('date_int', axis=1, inplace=True)
    train_set['100'] = temp.copy()

    # 101
    temp = train.copy()
    temp.drop('weather_event', axis=1, inplace=True)
    train_set['101'] = temp.copy()

    # 110
    temp = train.copy()
    temp.drop('date_int', axis=1, inplace=True)
    train_set['110'] = temp.copy()

    # 111
    train_set['111'] = train.copy()

    return train_set