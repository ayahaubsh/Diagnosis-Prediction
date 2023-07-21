""" This file is used to evaluate the performance of the models on the testing set. """
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from feature_selection import load_data
from model_selection import model_select
from feature_selection import corr_features


# Load and the dataset from 'Feature_Selection.py' and best parameters for the models from 'Model_Selection.py'
X_train, X_test, y_train, y_test = load_data()
best_models = model_select(X_train, y_train)

# Load the Testing dataset
testing_data = pd.read_csv('datasets\Testing.csv')

# applying feature selection:
testing_data_new = testing_data.drop(corr_features, axis = 1)

X = testing_data_new.iloc[:, :-1].values
y = testing_data_new.iloc[:,-1].values

# Call model_select function
best_models = model_select(X_train, y_train)

def performance_test1(X: pd.DataFrame = X, y: pd.DataFrame = y, best_models: dict = best_models) -> None:
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
    results_df.to_csv('test_model_performance_results.csv', index = False)

# Call the performance_test function
performance_test1()

# Determine the best model based on the highest F1 score
best_model = max(best_models.values(), key = lambda x: x['score'])

# Print the name and performance scores of the best model
print('------------------------------------------------------------')
print(f"The best model based on GridSearch cross-validation technique is {best_model['name']}")
print(f"F1 score using  GridSearch cross-validation technique: {best_model['score']}")
print('------------------------------------------------------------')
   
