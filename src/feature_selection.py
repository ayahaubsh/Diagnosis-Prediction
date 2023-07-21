''' This module loads the training dataset, and performs correlation-based feature selection. 
    by detecting the features with correlation greater than or equal the threshold value specified. '''

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from configs import load
from typing import Tuple, Iterable

# load threshold value from the configs
threshold = load(key = "threshold")

# Load the training dataset
training_data = pd.read_csv('datasets\Training.csv')
X_tr = training_data.iloc[:, :-1].values
y_tr = training_data.iloc[:,-1].values

# Check the correlations between features:
def correlation(dataset: pd.DataFrame, threshold: float) -> set:
    """
    This function is used to detect the features with correlation greater than or equal the threshold value specified.

    Parameters
    ----------
    dataset : pd.DataFrame
        The training dataset.
    threshold : float
        The threshold value.

    Returns
    -------
    set
        A set containing the features with correlation greater than or equal the threshold value specified.
    """
    col_corr = set()  
    corr_matrix = dataset.corr(numeric_only = True)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]  
                col_corr.add(colname)
    return col_corr

# calling the fucnction:
corr_features = correlation(training_data, threshold)
print(f'Number of Correlating Features is: {len(corr_features)}')
# print(f'List of correlating Features: {corr_features}')

# feature selection by deleting the features with greater than or equal to the threshold value:
training_data_new = training_data.drop(corr_features, axis = 1)
# print(training_data_new.shape[1])
# print(training_data_new.columns)


# function to detect to load the new training dataset and split it into training and test sets:
def load_data(dataset: pd.DataFrame = training_data_new) -> Tuple[Iterable[pd.DataFrame]]:
    # Load the training dataset
    training_data = dataset
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:,-1].values
    # print(training_data_new.head())

    # Convert the 'prognosis' column to a categorical data type
    training_data['prognosis'] = training_data['prognosis'].astype('category')

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # print(X_test.shape)
    return X_train, X_test, y_train, y_test

