import numpy as np
import pandas as pd
import multiprocessing as mp
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.len_X = len(X)
        self.height_limit = np.ceil(np.log2(self.sample_size))
        
        X_samples = [X[np.random.choice(self.len_X,self.sample_size,replace=False)] for i in range(self.n_trees)]
        
        with mp.Pool(mp.cpu_count()-1) as p:
            self.trees = p.map(IsolationTree(self.height_limit).fit,X_samples)
       

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        with mp.Pool(mp.cpu_count()-1) as p:
            avg_PathLength = np.array(p.map(self.find_avg_length,X))
        return avg_PathLength
    
    
    def find_avg_length(self, x):
        """
        find the avg length of an observation x across all trees
        """
        return sum(t.find(x) for t in self.trees)/self.n_trees

    def score_transform(self, x):
        """
        transforme avg length of an observation
        """
        return 2**(-1.0*x)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        self.c = c_factor(self.sample_size)
        scores = self.path_length(X)
        scores = self.score_transform(scores/self.c)
        return scores
        

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.where(scores >= threshold, 1, 0)
        

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        
        return self.predict_from_anomaly_scores(self.anomaly_score(X),threshold)
        

class IsolationTree:
    def __init__(self, height_limit):
        self.h = height_limit
        self.root = None

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = Node(X,0,self.h)
        self.n_nodes = self.root.n_nodes
        return self.root

        
class Node:
    def __init__(self, X:np.ndarray, current_height, height_limit):
        self.height = current_height
        self.left = None
        self.right = None
        self.size = len(X)
        self.n_nodes = 1
        
        if self.height < height_limit and self.size > 1:
            self.Q = np.shape(X)[1]
            self.q = np.random.choice(self.Q)
            min_ = X[:,self.q].min()
            max_ = X[:,self.q].max()
            self.p = np.random.uniform(min_,max_,1)
            cut = np.where(X[:,self.q] < self.p, True, False)
            X_l = X[cut]
            X_r = X[~cut]
            self.left = Node(X_l, self.height+1, height_limit)
            self.right = Node(X_r, self.height+1, height_limit)
            self.n_nodes += self.left.n_nodes + self.right.n_nodes
      
    def find(self, x):
        """
        find the length of an observation in one tree
        """
        while self.right != None:
            if x[self.q] < self.p: self = self.left
            else: self = self.right
        return self.height + c_factor(self.size)

         
    
def c_factor(size):
    if size == 2: return 1
    elif size > 2: return 2.0*(np.log(size-1)+0.5772156649) - 2.0*(size-1.0)/size
    else: return 0

