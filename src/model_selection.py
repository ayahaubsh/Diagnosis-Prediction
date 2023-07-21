import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

def model_select(X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    """
    This function is used to select the best models for the dataset.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training dataset features.
    y_train : pd.DataFrame
        The training dataset target
    
    Returns
    -------
    dict
        A dictionary containing the best parameters for each model type.
    """
    # Define the parameter grid for each model
    param_grid = [    {},    {'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'svc__C': [0.1, 1, 10], 'svc__gamma': [0.1, 1, 2.2, 10]},
        {'classifier__max_depth': [4, 6, 8, 10], 'classifier__min_samples_split': [4, 6, 8, 10]},
        {'classifier__n_estimators': [35, 40, 50, 80, 100, 120], 'classifier__max_depth': [7, 8, 9, 10, 12]}
    ]

    # Define the classification models to test
    models = [    {'name': 'Naive Bayes', 'model': GaussianNB()},    {'name': 'Support Vector Machine', 'model': Pipeline([        ('scale', StandardScaler()),        ('svc', SVC())    ])},
        {'name': 'Decision Tree Classifier', 'model': Pipeline([
            ('classifier', DecisionTreeClassifier())
        ])},
        {'name': 'Random Forest Classifier', 'model': Pipeline([
            ('classifier', RandomForestClassifier())
        ])}
    ]

    # Create a dictionary to store the best estimator and its score for each model
    best_models = {}

    # Define the scorer
    scorer = make_scorer(f1_score, greater_is_better = True, average = 'macro')
    # scorer = make_scorer(accuracy_score, greater_is_better=True)

    # Loop through each model and perform a grid search to find the best hyperparameters
    i = 0
    for model in models:
        grid_search = GridSearchCV(model['model'], param_grid[i], cv = 5, scoring = scorer)
        grid_search.fit(X_train, y_train)
        best_models[model['name']] = {'name': model['name'], 'model': grid_search.best_estimator_, 'score': grid_search.best_score_ , 'params': grid_search.best_params_}
        i += 1
    return best_models