''' This is the main module in the project which takes a new dataset as input, and predicts the prognosis of the patient. '''

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from feature_selection import load_data
from model_selection import model_select
from feature_selection import corr_features
from typing import List
X_train, X_test, y_train, y_test= load_data()
best_models = model_select(X_train, y_train)

def main(X: list[int]) -> str:

    # Determine the best model based on the highest F1 score
    best_model = max(best_models.values(), key = lambda x: x['score'])

    # Load the best model
    model = best_model['model']

    # Drop the features with correlation greater than or equal the threshold value specified
    X_new = X.drop(corr_features, axis = 1)

    # Make predictions on the new data
    y_pred = model.predict([X_new])
    return y_pred

if __name__ == "__main__":
    main()