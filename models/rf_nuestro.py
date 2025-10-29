import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RandomForest:
  def __init__(self, n_estimators=100, max_depth='sqrt',max_features='sqrt', random_state=17):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.max_features = max_features
    self.random_state = random_state
    self.trees = []

  def bootstrap(self, X, y):
    n_samples = len(X)
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

  def fit(self, X, y):
    self.trees = []
    n_features = X.shape[1]

    if self.max_features == 'sqrt':
      max_feats = max(1, int(np.sqrt(n_features)))
    elif self.max_features == 'log2':
      max_feats = max(1, int(np.log2(n_features)))
    else:
      max_feats = n_features


    if self.max_depth == 'sqrt':
      depth = max(1, int(np.sqrt(n_features)))
    elif self.max_depth == 'log2':
      depth = max(1, int(np.log2(n_features)))
    else:
      depth = self.max_depth

    for i in range(self.n_estimators):

      tree = DecisionTreeClassifier(max_depth=depth, max_features=max_feats,random_state=self.random_state + i)

      X_sample, y_sample = self.bootstrap(X, y)
      feature_idxs = np.random.choice(n_features, max_feats, replace=False)
      #X_sample_selected = X_sample[:, feature_idxs]
      tree.fit(X_sample, y_sample)
      self.trees.append(tree)

  def predict(self, X):
    tree_preds = np.array([tree.predict(X) for tree in self.trees])
    return np.array([np.argmax(np.bincount(tree_preds[:, i])) for i in range(len(X))])


  def fit_predict(self, X, y):
    self.fit(X, y)
    return self.predict(X)
