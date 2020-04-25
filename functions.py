import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


'''
Functions include:
    1) create_dataset
        Used to create a cleaned dataset using a selected set of Airbnb data
    2) get_feature_set
        Used to identify features of interest using backward elimination
    3) get_model
        Used to instantiate a linear model, split the dataset into train & test, and fit the model
    4) coeff_wieghts
        Gets the coefficient weights from the model returined from get_model
    5) drop_col_bythresh
        Drops columns that are sparcly populated by a specified threshold
    6) constrain_occupancy
        Drops columns that are outside of a specified occupancy percentage
'''

def create_dataset(listing, calendar, review, target_y, price_inclusive = 'y', clean_fee_binary = 'n'):
    '''
    Input:
    Airbnb dataframes created from the listing, calendar and review files
    target_y as a string representing the target column of interest for the model
    price_inclusive = n uses the original price columns; 
        whereas price_inclusive = y will combine price and cleaning fee (price_incl = price + cleaning fee)
    clean_fee_bianary - only used when price_inclusive is 'y', if selected as 'y' it will turn the clean_fee into a binary indicator with 
        a value of 1 if there is a fee present, else 0.  Default value is left as 'n'
        
    This function will take in three dataframes to create a single data frame.  The listing data will be the basis 
    of the dataset.  
    The number of bookings from the calendar data set will be joined to the listing data using the listing_id
    and loaded as a proportion of booked throught the year.  
    The number of reviews from the reviews data will be joined to the listing data using the listing id.
    
    Additionally, 
    Currency will be converted into numeric fields, 
    Boolean values will be converted into 0 or 1
    Dates will be converted to datetime
    Encodes identified categorical features
    Cleans zipcode and amenities
    Handles null values
    
    Returns a cleaned up dataset
    '''
    
    #Create dataframe from selected columns from the listing file
    data = listing[['id', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 
                   'host_listings_count', 'neighbourhood_group_cleansed', 'zipcode', 'property_type', 
                   'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
                   'amenities', 'square_feet', 'price', 'security_deposit', 'cleaning_fee', 
                   'guests_included', 'extra_people', 'review_scores_rating', 'review_scores_accuracy', 
                   'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 
                   'review_scores_location', 'review_scores_value'
                  ]].copy()
    
    #Create proportion of available and number of reviews from the calendar and reviews files respectivly
    calendar_booked_prop = calendar[['listing_id','available']]
    calendar_booked_prop['available'] = calendar_booked_prop['available'].map({'t': 0, 'f': 1})
    calendar_booked_prop = calendar_booked_prop[['listing_id','available']].groupby('listing_id').sum()
    calendar_booked_prop['available'] = calendar_booked_prop['available']/365
    calendar_booked_prop = calendar_booked_prop.rename(columns={'available':'calendar_booked_prop'})
    
    reviews_num = review[['listing_id','reviewer_id']]
    reviews_num['reviewer_id'] = 1
    reviews_num = reviews_num[['listing_id','reviewer_id']].groupby('listing_id').sum()
    reviews_num = reviews_num.rename(columns={'reviewer_id':'reviews_num'})
    
    #Join proportion of available and number of reviews to the main dataset
    data = data.join(calendar_booked_prop, on='id')
    data = data.join(reviews_num, on='id')
    
    #convert all currency objects to float and fill null with zero
    currancy = ['price', 'security_deposit', 'cleaning_fee', 'extra_people']
    data[currancy] = data[currancy].replace({'\$':'',',':''}, regex=True).astype(float)
    
    #If identifed within the function parameter, creates an inclusive cost that includes prices and cleaning fees. 
    #Else, makes cleaning fee binary
    if price_inclusive == 'y':
        data['price_incl'] = data['price']+data['cleaning_fee']
        data = data.drop(columns = ['price', 'cleaning_fee'])
        currancy = ['price_incl', 'security_deposit', 'extra_people']
    elif clean_fee_binary == 'y':
        data.loc[(data.cleaning_fee > 0 ), 'cleaning_fee'] = 1
        
   
    #remove any null rows from the target value
    data = data.dropna(axis = 0, subset = [target_y], how='any')   ## WORKING HERE
    
    #fill remaining null currancy values with zero
    for i in currancy:
        data[i] = data[i].fillna(0)
    
    #convert all percent objects to float
    data[['host_response_rate','host_acceptance_rate']] = data[['host_response_rate','host_acceptance_rate']].replace({'\%':''}, regex=True).astype(float)  
    data[['host_response_rate','host_acceptance_rate']] =  data[['host_response_rate','host_acceptance_rate']]/100

    #convert host_is_superhost to t = 1, f = 0
    data['host_is_superhost'] = data['host_is_superhost'].map({'t': 1, 'f': 0})
    
    #convert zip to int
    data['zipcode'] = data['zipcode'].replace({'\n':''}, regex=True)
    data['zipcode'] = data['zipcode'].replace({'^.{6,}$':np.NaN}, regex=True)
    data['zipcode'] = data['zipcode'].astype(float)
    
    #encode categorical features (less amenities)
    cat_features = ['neighbourhood_group_cleansed', 'property_type', 'room_type', 'bed_type']  
    data = pd.get_dummies(data = data, prefix = cat_features, prefix_sep = '_', columns = cat_features)
    
    #encode amenities, and remove duplicate columns (i.e. Washer, Dryer, Washer/ Dryer)
    amenities = data['amenities'].tolist()
    amenities = ''.join(amenities)

    amenities = amenities.replace("}",',')
    amenities = amenities.replace('{','')
    amenities = amenities.replace('"','')

    options = {}

    for item in amenities.split(','):
        name = item.lower()
        if item not in options:
            options.update({item:name})
        
    del options['']
    
    for item in options:
        data[options[item]] = data['amenities'].str.contains(item)
        data[options[item]] = data[options[item]].map({True: 1, False: 0})
    
    data = data.drop(columns = ['amenities'])
    
    #remove any rows or columns that are all null
    data = data.dropna(axis = 1, how='all')
    data = data.dropna(axis = 0, how='all')
    
    # remove any remaining null values with fill mean
    fill_mean = lambda col: col.fillna(col.mean())
    data = data.apply(fill_mean, axis=0)
    
    return data


def get_feature_set(df, y_target,pmax_lower_bound=0.05):
    '''
    Inputs:
    Cleaned Dataframe
    Target variable as string
    pmax lower bound to be used as the feature elimination criteria.
    
    Performs backward elimination to identify features of interest
    
    Returns new dataframe using only the features of interest
    '''
    data = df.dropna(axis=0, subset = [y_target], how='any').copy()  
    X = data.drop(y_target, axis = 1)
    y = data[y_target]
    
    #Backward Elimination
    features = list(X.columns)
    pmax = 1
    while (len(features)>0):
        p= []
        X_1 = X[features]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = features)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>pmax_lower_bound):
            features.remove(feature_with_p_max)
        else:
            break
    features.append(y_target)
    feature_set = pd.DataFrame(data = data, columns = features).copy()
        
    return feature_set

def get_model(data, y_var, test_size_var = .3, random_state_var = 42):
    '''
    Takes in the price data set as dataframe, and y_var as string
    [Optional] test size as float and random state as int
    Establishes variables, normalizes, splits, and instantiates a linear model
    '''
    
    X = data.drop(y_var, axis = 1)
    y = data[y_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_var, random_state = random_state_var)
    
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)
    
    y_train_preds = lm_model.predict(X_train)
    y_test_preds = lm_model.predict(X_test)
    
    return X_train, X_test, y_train, y_test, y_test_preds, y_train_preds, lm_model

def coef_weights(model, coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = model.coef_
    coefs_df['abs_coefs'] = np.abs(model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

def drop_col_bythresh(df, y_target, percent_thresh = 0.01):
    '''
    INPUT:
    data - dataframe to remove low impact columns
    y_target - string value for the column of interest
    percent_thresh - any column with percentage of populated values less then the specified threshold will be removed
    
    Output:
    dataframe with removed columns
    '''
    data = df
    denom = data.shape[0]
    col = data.columns.to_list()
    col.remove(y_target)

    for c in col:
        col_data = pd.DataFrame(data[c] == 0)
        col_data = col_data.drop(col_data[~col_data[c]].index)
        if (col_data.shape[0]/denom) < percent_thresh:
            data = data.drop(columns = [c])
    
    return data

def constrain_occupancy(data, min_occupy, max_occupy):
    '''
    INPUTS:
    data - dataframe to reduce
    min_occupy - the minimum occupancy rate entered as a decimal percent between 0 and 1 (i.e. 0.2)
    max_occupy - the maximum occupancy rate entered as a decimal percent between 0 and 1 (i.e. 0.8)
    
    This function takes in the data created from the create_dataset (or create_dataset2), and eliminates rows that are at or below 
    the defined min_occupy, and at or above the max_occupy.  
    
    OUTPUT:
    data - updated dataframe with reduced row count
    '''
    data = data.drop(data.query('calendar_booked_prop < @min_occupy or calendar_booked_prop > @max_occupy').index)
    
    return data
    
