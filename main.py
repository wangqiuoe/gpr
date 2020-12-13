import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
import time

NUM_FEATURE = 4
REPEAT = 5
TRAIN_TEST_SPLIT=0.7

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
import time

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


def gen_random_train_test(df,i):
    # experiment will repeat 20 times, i = 0,...19
    x = df[['AT','V','AP','RH']].to_numpy()
    y = df[['PE']].to_numpy()
    x, y = shuffle(x, y, random_state=i)
    len_train = int(x.shape[0]*TRAIN_TEST_SPLIT)
    train,test = {},{}
    train['x'] = x[:len_train,:]
    train['y'] = y[:len_train,:]
    test['x'] = x[len_train:,:]
    test['y'] = y[len_train:,:]
    train['x'],test['x'],_,_ = normalize(train['x'],test['x'])
    train['y'],_,mu_y,std_y = normalize(train['y'],test['y'])
    return train,test,mu_y,std_y

def normalize(train,test):
    mu = np.mean(train,0)
    std = np.std(train,0)
    train = (train-mu)/std
    test = (test-mu)/std
    return train,test,mu,std

def cal_rmse(y,y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

def logpdf(d, v):
    return -np.log(2 * math.pi * v) / 2 - d**2 / 2 / v

def cal_ll(y,y_pred,y_var):
    return  -logpdf(y - y_pred, y_var).mean()

def run(model):
    df = pd.read_excel('Folds5x2_pp.xlsx') 
    metrics = {}
    mean_metrics = {}
    metrics['rmse'] = []
    t0 = time.time()
    for times in range(REPEAT):
        print ('======================= Experiment: ',times, " ===============================")
        random_seed = times
        train,test,mu_y,std_y = gen_random_train_test(df,random_seed)
        model.train(train)
        pred = model.predict(test)  
        y_pred = pred.get('y_pred')
        y_pred = y_pred * std_y + mu_y
        rmse = cal_rmse(test['y'],y_pred)
        metrics['rmse'].append(rmse)
        print (rmse)
    for key in metrics:
        mean_metrics[key] = {'mean':np.mean(metrics[key]), 'list':metrics[key]}
    t1 = time.time()
    mean_metrics['time'] = t1-t0
    return mean_metrics



from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct

#perodic_kernel = 2*ExpSineSquared(length_scale=0.7, periodicity=2.87 ) 
perodic_kernel = C(0.1, (0.001, 0.1)) * ExpSineSquared(length_scale=0.1, periodicity=0.1,
                                length_scale_bounds=(0.1, 10.0))

rbf_kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
dot_kernel = C(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0))


models= {
    'per_zero_noise_no_opt':SklearnGP(perodic_kernel, alpha=0.001, n_restarts_optimizer=0),
    'per_with_noise_no_opt':SklearnGP(perodic_kernel, alpha=1, n_restarts_optimizer=0),
    'per_zero_noise_opt':SklearnGP(perodic_kernel, alpha=0.001,  n_restarts_optimizer=1),
    'per_with_noise_opt':SklearnGP(perodic_kernel, alpha=1, n_restarts_optimizer=1),
}
#    'dot_zero_noise_no_opt':SklearnGP(dot_kernel, alpha=0.001, n_restarts_optimizer=0),
#    'dot_with_noise_no_opt':SklearnGP(dot_kernel, alpha=1, n_restarts_optimizer=0),
#    'dot_zero_noise_opt':SklearnGP(dot_kernel, alpha=0.001, n_restarts_optimizer=1),
#    'dot_with_noise_opt':SklearnGP(dot_kernel, alpha=1, n_restarts_optimizer=1),
#}


metrics = {}
for model_name, model in models.items():
    print(model_name)
    try:
        metrics[model_name] = run(model)
    except Exception as e:
        print(e)
        metrics[model_name] = 'error'

import json
with open('result_per_kernel.txt', 'w') as f:
    f.write(json.dumps(metrics, indent=2))
