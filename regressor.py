import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
import time
import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Sequential
import GPy

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


class SklearnGP():
    def __init__(self, kernel, n_restarts_optimizer=1, alpha=0.1):
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        
    def train(self,train):
        print ("=== Train")
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer, alpha=self.alpha)
        self.model.fit(train['x'],train['y'])
        return
    
    def predict(self,test):
        print ("=== Predict")
        y_pred, sigma = self.model.predict(test['x'], return_std=True)
        return {'y_pred':y_pred, 'y_var':sigma**2}
        



