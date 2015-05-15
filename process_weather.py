__author__ = 'sheraz'

import pandas as pd
import numpy as np


def process_weather_file(weather, add_wknd, add_trail_lead, other_data_flag, add_holiday_flag):

    # Update missing values or T for trace values
    print("Updating missing weather data.....")
    weather = weather.applymap(lambda x: 0 if x == 'M' or x == '-' else x)
    weather.snowfall = weather.snowfall.apply(lambda x: 0.01 if x == '  T' else x)
    weather.preciptotal = weather.preciptotal.apply(lambda x: 0.001 if x == '  T' else x)

    # Add columns for the split values for CODESUM
    print("Parsing CODESUM.....")
    unique_codes = []
    codesums = list(pd.Series(weather.codesum.values.ravel()).unique())   # Get a list with all the diff codesums
    for codes in codesums:
        code_split_list = codes.split()
        for unique_code in code_split_list:
            unique_codes.append(unique_code)
    myset = set(unique_codes)   ## get unique values

    # Create the new columns in the dataframe
    for code in myset:
        weather[code] = 0.0

    # iterate over rows and assign 1 where the code appears
    for index, row in weather.iterrows():
        codes = str(row['codesum']).split()
        for code in codes:
            weather.at[index,code] = 1.0
    weather.drop('codesum', axis=1, inplace=True)

    # Turn dates into float values
    w_dates = pd.to_datetime(weather['date'])
    weather['date_int'] = w_dates.astype(np.int64) / 1000000000000000

    # add flags for each day and add week of year
    weather = create_day_flag(weather)
    weather = add_week_of_year(weather)
    # weather = add_month_of_year(weather)

    # Create a flag indicating if the given day is a weekend
    if add_wknd:
        weather = create_wknd_flag(weather)

    # Add 3 days of pre and post data for precipitation and snow
    if add_trail_lead:
        weather = add_trailing_leading_weather(weather)

    # Add 3 days of pre and post data for precipitation and snow
    if add_holiday_flag:
        weather = add_holidays(weather)

    # xxx
    if other_data_flag:
        weather.drop('sunrise', axis=1, inplace=True)
        weather.drop('sunset', axis=1, inplace=True)
        weather.drop('depart', axis=1, inplace=True)
        weather.drop('dewpoint', axis=1, inplace=True)
        weather.drop('stnpressure', axis=1, inplace=True)
        weather.drop('sealevel', axis=1, inplace=True)

    weather.to_csv("../wmt_data/weather_processed.csv")
    return weather


def create_wknd_flag(weather):

    # Create flags for weekends
    print("Creating weekend flag.....")
    weather['weekend'] = 0
    for index, row in weather.iterrows():
        if pd.to_datetime(row['date']).dayofweek in [5, 6]:
            weather.at[index,'weekend'] = 1

    return weather


def add_week_of_year(weather):

    # Create flags for weekends
    print("adding week of year.....")
    weather['weekofyear'] = 0
    for index, row in weather.iterrows():
        weather.at[index,'weekofyear'] = pd.to_datetime(row['date']).weekofyear

    return weather


def add_month_of_year(weather):

    # Create binary vars for each month
    print("Creating flags for each month.....")
    months = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}

    # Create columns for each month, set to zero
    for month_key in months:
        weather[months[month_key]] = 0

    # Set month to 1
    for index, row in weather.iterrows():
        month_int = pd.to_datetime(row['date']).month
        weather.at[index, months[month_int]] = 1

    return weather


def create_day_flag(weather):

    # Create flags for each day
    print("Creating flags for each day.....")
    weather['mon'] = 0
    weather['tue'] = 0
    weather['wed'] = 0
    weather['thu'] = 0
    weather['fri'] = 0
    weather['sat'] = 0
    weather['sun'] = 0
    for index, row in weather.iterrows():
        if pd.to_datetime(row['date']).dayofweek == 0:
            weather.at[index,'mon'] = 1
        if pd.to_datetime(row['date']).dayofweek == 1:
            weather.at[index,'tue'] = 1
        if pd.to_datetime(row['date']).dayofweek == 2:
            weather.at[index,'wed'] = 1
        if pd.to_datetime(row['date']).dayofweek == 3:
            weather.at[index,'thu'] = 1
        if pd.to_datetime(row['date']).dayofweek == 4:
            weather.at[index,'fri'] = 1
        if pd.to_datetime(row['date']).dayofweek == 5:
            weather.at[index,'sat'] = 1
        if pd.to_datetime(row['date']).dayofweek == 6:
            weather.at[index,'sun'] = 1

    return weather


def add_holidays(weather):

    # Create value for week of year
    print("adding holidays.....")
    holiday_dict = {'2012-01-16':'mlk',
                    '2013-01-21':'mlk',
                    '2014-01-20':'mlk',
                    '2012-01-23':'chinese',
                    '2013-02-10':'chinese',
                    '2012-02-19':'db_prez',
                    '2012-02-20':'prez',
                    '2013-02-17':'db_prez',
                    '2013-02-18':'prez',
                    '2014-02-16':'db_prez',
                    '2014-02-17':'prez',
                    '2012-04-06':'goodfri',
                    '2012-04-08':'easter',
                    '2012-04-09':'easter_mon',
                    '2013-03-29':'goodfri',
                    '2013-03-31':'easter',
                    '2013-04-01':'easter_mon',
                    '2014-04-18':'goodfri',
                    '2014-04-20':'easter',
                    '2014-04-21':'easter_mon',
                    '2012-05-13':'moth_day',
                    '2012-05-12':'db_moth_day',
                    '2013-05-12':'moth_day',
                    '2013-05-11':'db_moth_day',
                    '2014-05-11':'moth_day',
                    '2014-05-10':'db_moth_day',
                    '2012-05-27':'db_mem_day',
                    '2012-05-28':'mem_day',
                    '2013-05-26':'db_mem_day',
                    '2013-05-27':'mem_day',
                    '2014-05-25':'db_mem_day',
                    '2014-05-26':'mem_day',
                    '2012-06-17':'fath_day',
                    '2012-06-16':'db_fath_day',
                    '2013-06-16':'fath_day',
                    '2013-06-15':'db_fath_day',
                    '2014-06-15':'fath_day',
                    '2014-06-14':'db_fath_day',
                    '2012-09-03':'lab_day',
                    '2013-09-02':'lab_day',
                    '2014-09-0x':'lab_day',
                    '2012-02-14':'valentine',
                    '2013-02-14':'valentine',
                    '2014-02-14':'valentine',
                    '2012-12-25':'xmas',
                    '2013-12-25':'xmas',
                    '2012-12-24':'xmaseve',
                    '2013-12-24':'xmaseve',
                    '2012-11-21':'dbthanks',
                    '2012-11-22':'thanks',
                    '2012-11-23':'blackfriday',
                    '2013-11-27':'dbthanks',
                    '2013-11-28':'thanks',
                    '2013-11-29':'blackfriday',
                    '2012-12-31':'nye',
                    '2013-12-31':'nye'}

    # Set all values to 0
    for key, val in holiday_dict.iteritems():
        weather[val] = 0

    # Iterate over holidays
    for key, val in holiday_dict.iteritems():
        print("..adding {}".format(val))
        for index, row in weather.iterrows():
            if row['date'] == key:
                weather.at[index,val] = 1

    return weather


def add_trailing_leading_weather(weather):

    print("Adding 3 day pre and post weather.....")
    # Precip and snowfall 1,2,3 days before and after
    weather['minus_3_precip'] = np.nan
    weather['minus_2_precip'] = np.nan
    weather['minus_1_precip'] = np.nan
    weather['plus_1_precip'] = np.nan
    weather['plus_2_precip'] = np.nan
    weather['plus_3_precip'] = np.nan
    #############
    weather['minus_3_snow'] = np.nan
    weather['minus_2_snow'] = np.nan
    weather['minus_1_snow'] = np.nan
    weather['plus_1_snow'] = np.nan
    weather['plus_2_snow'] = np.nan
    weather['plus_3_snow'] = np.nan

    master_weather = pd.DataFrame()
    for stn in range(1,21):
        print("processing station {}".format(stn))
        weather_stn = weather[weather['station_nbr'] == stn]
        weather_stn = update_precip(weather_stn)
        master_weather = master_weather.append(weather_stn)

    return master_weather


def update_precip(weather_stn):

    weather_stn = weather_stn.sort('date')
    p = np.array(weather_stn['preciptotal'])
    s = np.array(weather_stn['snowfall'])

    count = -3
    for index, row in weather_stn.iterrows():
        if count > -1:
            weather_stn.at[index,'minus_3_precip'] = p[count]
            weather_stn.at[index,'minus_3_snow'] = s[count]
        count += 1

    count = -2
    for index, row in weather_stn.iterrows():
        if count > -1:
            weather_stn.at[index,'minus_2_precip'] = p[count]
            weather_stn.at[index,'minus_2_snow'] = s[count]
        count += 1

    count = -1
    for index, row in weather_stn.iterrows():
        if count > -1:
            weather_stn.at[index,'minus_1_precip'] = p[count]
            weather_stn.at[index,'minus_1_snow'] = s[count]
        count += 1

    count = 1
    for index, row in weather_stn.iterrows():
        if count < len(p):
            weather_stn.at[index,'plus_1_precip'] = p[count]
            weather_stn.at[index,'plus_1_snow'] = s[count]
        count += 1

    count = 2
    for index, row in weather_stn.iterrows():
        if count < len(p):
            weather_stn.at[index,'plus_2_precip'] = p[count]
            weather_stn.at[index,'plus_2_snow'] = s[count]
        count += 1

    count = 3
    for index, row in weather_stn.iterrows():
        if count < len(p):
            weather_stn.at[index,'plus_3_precip'] = p[count]
            weather_stn.at[index,'plus_3_snow'] = s[count]
        count += 1

    return weather_stn

