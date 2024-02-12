import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distances = np.zeros((len(X), len(self.train_X)))
        for i in range(len(X)):
            for j in range(len(self.train_X)):
                distances[i][j] = np.sum(np.abs(X[i] - self.train_X[j]))
        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((len(X), len(self.train_X)))
        for i in range(len(X)):
            distances[i] = np.sum(np.abs(self.train_X - X[i]), axis = 1)
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        binary_test_X_3d = X.reshape(X.shape[0], 1, X.shape[1])
        distances = np.sum(np.abs(binary_test_X_3d - self.train_X), axis = 2)
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        neighbour_indices = distances.argsort(axis=1)[:,:self.k]
        pred = []

        for i in neighbour_indices:
            if (self.train_y[i] == '1').sum() > (self.train_y[i] == '0').sum():
                pred.append('1')
            else:
                pred.append('0')
        return pred


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """
        neighbour_indices = distances.argsort(axis=1)[:,:self.k]
        pred = []
        for i in neighbour_indices:
            classes, indices = np.unique(self.train_y[i], return_inverse=True)
            pred.append(int(classes[np.bincount(indices).argmax()]))
        return pred
