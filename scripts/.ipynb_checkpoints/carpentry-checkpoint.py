import os
import re
import pandas as pd
import numpy as np
import subprocess
from sklearn.impute import SimpleImputer


#### Load Data ####
data_dir = '/Users/jonyarber/Documents/Projects/spaceship_titanic/data/'

ss_titanic_train_raw = pd.read_csv(f'{data_dir}train.csv')

ss_titanic_test_raw = pd.read_csv(f'{data_dir}test.csv')



#### Functions ####

def make_dummies(df, var_list):
    
    for var in var_list:
        df = pd.concat([df,
                        pd.get_dummies(df[var],
                                       prefix = var,
                                       dtype = 'int')],
                       axis = 1)

    df = df.drop(var_list, axis = 1)
        
    return df


def clean_dfs(df):
    
    # Create a copy of DF
    df = df.copy()

    # Convert True/False columns to binary
    binary_vars = ['CryoSleep', 'VIP', 'Transported']

    # 'Transported' won't be in test DF
    binary_vars = list(set(df.columns).intersection(binary_vars))

    #df.loc[:, binary_vars] = df[binary_vars].map(lambda x: 1 if x == True else 0 if x == False else x)
   
    for var in binary_vars:
        df[var] = [1 if x == True else 0 if x == False else x for x in df[var]]

    
    #### Billing ####
    # Billing Vars
    billing_vars = ['RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa']

    # Fill the NAs with 0
    df.loc[:, billing_vars] = df[billing_vars].fillna(0) 

    # Create TotalSpent
    df['TotalSpent'] = df[billing_vars].sum(axis = 1)

    
    #### Group ####
    # Check if in group
    df['GroupNo'] = [x.split('_')[0] for x in df['PassengerId']]
    df['InGroup'] = np.where((df['GroupNo'] == df['GroupNo'].shift(1))  | (df['GroupNo'] == df['GroupNo'].shift(-1)) , 1, 0)

    
    #### Cabin ####
    # Separate out cabin
    df[['Deck', 'CabinNo', 'Side']] = df['Cabin'].str.split("/", expand = True)


    #### Home Planet ####
    # Some Home Planets can be determined by Deck
    # If Deck is A-C, Europa
    df.loc[df['Deck'].isin(['A', 'B', 'C']), 'HomePlanet'] = df['HomePlanet'].fillna('Europa')

    # If Deck is D, Mars
    df.loc[df['Deck'] == 'D', 'HomePlanet'] = df['HomePlanet'].fillna('Mars')

    # If Deck is G, Earth
    df.loc[df['Deck'] == 'G', 'HomePlanet'] = df['HomePlanet'].fillna('Earth')


    #### Age ####
    # Impute Age based on Home Planet
    median_age_by_planet = df.groupby('HomePlanet')['Age'].median()

    for planet in median_age_by_planet.index:
        df.loc[df['HomePlanet'] == planet, 'Age'] = df['Age'].fillna(median_age_by_planet[planet])

    # Fill remaining with median age
    df['Age'] = df['Age'].fillna(df['Age'].median())


    #### Cryo Sleep ####
    # If money was spent, not in CryoSleep (where NA)
    df['CryoSleep'] = df.apply(lambda x: 0 if x['TotalSpent'] > 0 & pd.isna(x['CryoSleep']) else x['CryoSleep'], axis = 1)
    
    # If Age < 12 and no money spent, not in CryoSleep (where NA)
    df['CryoSleep'] = df.apply(lambda x: 1 if x['TotalSpent'] == 0 & int(x['Age']) < 12 & pd.isna(x['CryoSleep']) else x['CryoSleep'], axis = 1)
    
    # Fill rest of CyroSleep with 1 (no money spent - in CryoSleep)
    df['CryoSleep'] = df['CryoSleep'].fillna(1)

    
    #### VIP ####
    # Most people didn't travel VIP - can fill NA with 0
    df['VIP'] = df['VIP'].fillna(0)


    #### Final Clean Up ####
    # Clean the Destinations for final DF
    df['Destination'] = df['Destination'].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', str(x)).upper() if pd.notnull(x) else np.nan)

    # Drop Cabin, Name, and GroupNo
    df.drop(['GroupNo', 'Name', 'Cabin', 'CabinNo'], axis = 1, inplace = True)

    # Dummy categorical variables
    df = make_dummies(df, ['HomePlanet', 'Destination', 'Deck', 'Side'])

    return df      


#### Clean Data ####

ss_titanic_train, ss_titanic_test = [clean_dfs(df) for df in [ss_titanic_train_raw, ss_titanic_test_raw]]



#### Send to next script ####

def get_data_frames():
    return ss_titanic_train, ss_titanic_test