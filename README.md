![GitHub Logo](https://s3.ap-south-1.amazonaws.com/greyatom-social/logo.png)

# Decision Trees Classifier

We have seen how decision trees work. Let's try to solve a classification problem using decision trees.

### Task 1: Load the dataset

Write a function `load_data` that loads a dataset with numpy's loadtext api

* Accepts the following parameter
    * path to file (str)
    * skiprows (header rows to be skipped)

### Task 2: Write a function called myDecisionClassifier()

- Accepts the following parameters:
    * X_train, y_train, X_test (Numpy arrays for training, testing; any format acceptable by sklearn will work)
    * paramgrid (list of parameters (including those of the classfier) for RandomizedSearchCV)
    * n_iter_search (Number of iterations the search will be run)
    * KFold (the number of k-folds to be used in cross-validation) (Optional) (Default 3)
    * early_stopping_rounds (Int) (Optional) (Default 10)
    * seed (a number; a subsequent call to the function with the same seed will reproduce the same results)(optional) (Default 42)
    * **kwargs (To set parameters to the base classifier)

- Should return
    * predictions for X_test
    * trained RandomizedSearchCV object

### Task 3:  Figure out the best parameters for Pima Indian Diabetes Dataset

- Write a function called finetune_class which
    * Takes in X_train, X_test, y_train, param_grid, n_iter_search
    * Returns y_pred_test, Trained RandomizedSearch Object

- You will use myDecisionClassifier() function
- Based on the stage-wise optimization values further fine tune the model
- You will need to provide a parameter grid to the said function, be careful to choose which values to be optimized.
