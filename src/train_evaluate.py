""" This file is used to evaluate the performance of the models on the splitted training set. """
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from feature_selection import load_data
from model_selection import model_select


# Load and the splitted training dataset 
X_train, X_test, y_train, y_test = load_data()

# Call model_select function
best_models = model_select(X_train, y_train)

def performance_test(X: pd.DataFrame = X_test, y: pd.DataFrame = y_test, best_models: dict = best_models) -> None:
    """
    This function is used to evaluate the performance of the models on the training set.

    Parameters
    ----------
    X : pd.DataFrame
        The training dataset features.
    y : pd.DataFrame
        The training dataset labels.
    best_models : dict
        A dictionary containing the best models for each model type.
    """
    results_list = []

    # Evaluate the performance of each model on the training set
    for model_name, model_info in best_models.items():
        model = model_info['model']
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average = 'weighted', zero_division = 1)
        recall = recall_score(y, y_pred, average = 'weighted')
        f1 = f1_score(y, y_pred, average = 'weighted')

        # Add the model performance metrics to the results list
        results_list.append({
            'Model Name': model_name,
            'Best Parameters': model_info['params'],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'F1 Score using GridSearch CV': model_info['score']
        })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a CSV file
    results_df.to_csv('train_model_performance_results.csv', index = False)

# Call the performance_test function
performance_test()
