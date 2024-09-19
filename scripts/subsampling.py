import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score


def get_accuracy_score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_pred, y_test)

    return accuracy


def random_state_generator(df, scaler, model, n_states):

    best_score = 0
    best_features_diff = 1000000

    # Do not need PassengerId for this process
    df_copy = df.copy().drop('PassengerId', axis = 1)
    
    # Create X to compare train results
    y = df_copy.pop('Transported')
    X = scaler.fit_transform(df_copy)
    X = pd.DataFrame(X, columns = df_copy.columns)

    for n in np.arange(n_states + 1):

        # Create split will prep df
        X_train, x_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = n)
        
        score = get_accuracy_score(model, X_train, x_test, y_train, y_test)
                
        X_train = pd.DataFrame(X_train, columns = df_copy.columns)

        features_diff = sum(abs(X.mean() - X_train.mean()))

        if score > best_score:
            best_score = score
            best_score_state = n

        if features_diff < best_features_diff:
            best_features_diff = features_diff
            best_features_state = n
    
    return best_score_state, best_features_state


