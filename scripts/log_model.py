#### Load Libraries ####
import os
os.chdir('/Users/jonyarber/Documents/Projects/spaceship_titanic/scripts')
import carpentry
import subsampling

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Models
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#### Functions ####
def load_dfs():
    train_df, test_df = carpentry.get_data_frames()
    
    return train_df, test_df


def df_prep(df, scaler):
    
    df_copy = df.copy().drop('PassengerId', axis = 1)
    
    y = df_copy.pop('Transported')
    X = scaler.fit_transform(df_copy)
    
    X = pd.DataFrame(X, columns = df_copy.columns)
    
    return X, y


def get_random_states(use_script, df, scaler, model, states):
    
    if use_script:
        ran_state_best_score, ran_state_best_features = subsampling.random_state_generator(df, scaler = scaler, model = model, n_states = states)
        random_states = [ran_state_best_score, ran_state_best_features]
    else:
        # If best states are known
        ran_state_best_score = 515
        ran_state_best_features = 754
        random_states = [ran_state_best_score, ran_state_best_features]
    
    return random_states


def run_grid_search(df, scaler, model, cv, param_grid):

    X, y = df_prep(df, scaler)

    grid_results = GridSearchCV(model, 
                                param_grid = param_grid, 
                                scoring = 'accuracy',
                                n_jobs = -1, 
                                cv = cv)
    
    grid_results.fit(X, y)

    grid_df = pd.DataFrame(grid_results.cv_results_)
    grid_df = grid_df.iloc[:, 4:].drop(['params'], axis = 1)
    grid_df['mean_test_score'] = round(grid_df['mean_test_score'], 4)
    grid_df['std_test_score'] = round(grid_df['std_test_score'], 4)
    grid_df.drop('rank_test_score', axis = 1, inplace = True)
    grid_df = grid_df.sort_values(by = ['mean_test_score', 'std_test_score'], ascending = [False, True])
    
    return grid_df


def get_accuracy_score(df, scaler, model, random_state):
    
    X, y = df_prep(df, scaler)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = random_state)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_pred, y_test)

    return accuracy


def create_final(train_df, test_df, model, scaler):
    
    # Train tuned model to entire dataset
    X, y = df_prep(train_df, scaler)
    
    model.fit(X, y)
    
    # Final prediction
    X_final = scaler.fit_transform(test_df.copy().drop("PassengerId", axis = 1))

    final_pred_df = pd.DataFrame({'Transported': model.predict(X_final)})
    final_pred_df = final_pred_df['Transported'].apply(lambda x: True if x == 1 else False)
    final_pred_df = pd.concat([test_df['PassengerId'], final_pred_df], axis = 1)
    
    return final_pred_df



#### Create Model ####

# Load data frames
ss_titanic_train, ss_titanic_test = load_dfs()

# Set model to be used
model = LogisticRegression(n_jobs = -1, random_state = 0)

# Set scaler to be used
scaler = StandardScaler()

# Set CV method to be used
sss = StratifiedShuffleSplit(n_splits = 5, random_state  = 33) # 33 determined in a test of 100


# For logistic regression there are 3 penalty types that have to be considered separately

# L2
# Set parameters for search
model_params_l2 = {'penalty': ['l2'],
                   'C': [.01, 1, 10, 100],
                   'max_iter':[100, 200, 300],
                   'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear', 'newton-cholesky']}

# Run grid search
grid_df_l2 = run_grid_search(df = ss_titanic_train, 
                             scaler = scaler,
                             model = model,
                             cv = sss, 
                             param_grid = model_params_l2)


# Elastic-Net
model_params_en = {'penalty': ['elasticnet'],
                   'C': [.01, 1, 10, 100],
                   'max_iter': np.arange(100, 300, 50),
                   'solver': ['saga'],
                   'l1_ratio': np.arange(.1, .9, .1)}


grid_df_en = run_grid_search(df = ss_titanic_train, 
                             scaler = scaler,
                             model = model,
                             cv = sss, 
                             param_grid = model_params_en)

# L1
model_params_l1 = {'penalty': ['l1'],
                   'C': [.01, 1, 10, 100],
                   'max_iter':[100, 200, 300],
                   'solver': ['liblinear', 'saga']}

# Run grid search
grid_df_l1 = run_grid_search(df = ss_titanic_train, 
                             scaler = scaler,
                             model = model,
                             cv = sss, 
                             param_grid = model_params_l1)

# Tune model
model_tuned = LogisticRegression(n_jobs = -1,
                                 random_state = 0,
                                 C = 1,
                                 solver = 'lbfgs',
                                 penalty = 'l2')

# Get random states
random_states = get_random_states(use_script = True, 
                                  df = ss_titanic_train,
                                  scaler = scaler, 
                                  model = model_tuned,
                                  states = 100)



for state in random_states:
    
    pre_tuned_score = get_accuracy_score(ss_titanic_train, scaler, model, state)
    
    print(f'Pre-Tuned Model Result: {pre_tuned_score}')

    post_tuned_score = get_accuracy_score(ss_titanic_train, scaler, model_tuned, state)
    
    print(f'Post-Tuned Model Result: {post_tuned_score}')

    del pre_tuned_score, post_tuned_score



ss_titanic_final = create_final(ss_titanic_train, ss_titanic_test, model_tuned, scaler)

ss_titanic_final.to_csv('/Users/jonyarber/Documents/Projects/spaceship_titanic/data/ss_titanic_final_log.csv', index = False)
