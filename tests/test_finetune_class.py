from unittest import TestCase
from sklearn.metrics import accuracy_score

class TestFinetune_class(TestCase):
    def test_finetune_class(self):
        from build import load_data, finetune_class

        X_train, X_test, y_train, y_test = load_data('./data/diabetes.csv', skiprows=1)

        param_grid = {"max_depth": [6, 8, 10],
                      "max_leaf_nodes": [None, 5, 10, 20],
                      "min_impurity_split": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

        n_iter_search = 5

        y_pred, params = finetune_class(X_train, X_test, y_train, param_grid, n_iter_search)
        acc = accuracy_score(y_pred, y_test)
        self.assertGreater(acc, 0.7)