# Comparison of Different Regularization and Variable Selection Techniques
In this project, I will apply and compare the different regularization techniques including **Ridge, Lasso, Elastic Net, Square Root Lasso, and SCAD**.


### 1. Create Sklearn Compliant Functions for Square Root Lasso and SCAD
Create my own sklearn compliant functions for SCAD and Square Root Lasso, so I could use them in conjunction with GridSearchCV for finding optimal hyper-parameters when data such as x and y are given.


Import libraries:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from numba import njit
```

Create functions:

```python
# Square Root Lasso
class SQRTLasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
  
    def fit(self, x, y):
        alpha=self.alpha
        @njit
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        @njit
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = (-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)
          return output.flatten()
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))
        
        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)


# SCAD
class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
        
        beta0 = np.zeros(p)
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
 
 
@njit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part

@njit    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    
```


### 2. Simulate Data Sets
Simulate 2 data sets, each with 1200 features, 200 observations and a toeplitz correlation structure such that the correlation between features i and j is approximately <img width="43" alt="image" src="https://user-images.githubusercontent.com/98488324/162596738-99139dcd-a36e-4284-bac6-7d5117ca1c48.png"> with<img width="66" alt="image" src="https://user-images.githubusercontent.com/98488324/162596771-90313643-f476-41ea-b4a5-5bb7924cb808.png">.    

For the dependent variable y consider the following functional relationship:   
<img width="151" alt="image" src="https://user-images.githubusercontent.com/98488324/162596725-7155c6bc-a5de-4e5e-9bf5-0f34c0d07c6e.png">   
where <img width="83" alt="image" src="https://user-images.githubusercontent.com/98488324/162596803-6fe9f3e3-b367-4434-bff9-feb3e2eb7133.png"> is a column vector with <img width="102" alt="image" src="https://user-images.githubusercontent.com/98488324/162596796-9b0638b5-ab29-4d23-8386-eb4f0a949bb0.png"> (that is to say ϵi is normally distributed) and   
 
<img width="634" alt="image" src="https://user-images.githubusercontent.com/98488324/162596179-c940e997-f1d9-49ac-bea1-cd308826c98f.png">


```python
# 1200 features, 200 observations
n = 200
p = 1200

# B*
beta_star = np.concatenate(([1]*7,[0]*25,[0.25]*5,[0]*50,[0.7]*15,[0]*1098))

# ρ = 0.8
# we need toeplitz([1,0.8,0.8**2,0.8**3,0.8**4,...0.8**1199])
v = []
for i in range(p):
  v.append(0.8**i)
  
# σ = 3.5
mu = [0]*p
sigma = 3.5

# Generate the random samples
np.random.seed(123)
x = np.random.multivariate_normal(mu, toeplitz(v), size=n) # this where we generate some fake data
# y = X.dot(beta) + sigma*np.random.normal(0,1,[num_samples,1])
y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(n,1))

```


### 3. Apply Regressions and Variable Selection Methods
Apply the variable selection methods such as **Ridge, Lasso, Elastic Net, Square Root Lasso, and SCAD** with **GridSearchCV (for tuning the hyper-parameters)** and record the final results, such as the overall (on average) quality of reconstructing the sparsity pattern and the coefficients of β*.   
The final results include the average number of true non-zero coefficients discovered by each method, the L2 distance to the ideal solution, and the Root Mean Squared Error.

```python
# 2 datasets

for j in range(2):
  # Ridge
  model_ridge = Ridge(alpha=10)
  model_ridge.fit(x,y)
  betahat_ridge = model_ridge.coef_
  print('B̂ (Coefficients of B*) for Ridge: ', betahat_ridge)
  pos_ridge = np.where(betahat_ridge != 0)
  print('The average number of true non-zero coefficients for Ridge is: ', np.array(pos_ridge).shape[1])
  grid_ridge = GridSearchCV(estimator=model_ridge,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 20)})
  # print(grid_ridge.fit(x,y))
  grid_results_ridge = grid_ridge.fit(x,y)
  print('The optimal hyper-parameter for Ridge is: ', grid_results_ridge.best_params_)
  print('The mean square error for Ridge is: ', np.abs(grid_results_ridge.best_score_))
  print('The L2 for Ridge is: ', np.linalg.norm((betahat_ridge-beta_star), ord=2))

  # Lasso
  model_lasso = Lasso(alpha=0.1)
  model_lasso.fit(x,y)
  betahat_lasso = model_lasso.coef_
  print('B̂ (Coefficients of B*) for Lasso: ', betahat_lasso)
  pos_lasso = np.where(betahat_lasso != 0)
  print('The average number of true non-zero coefficients for Lasso is: ', np.array(pos_lasso).shape[1])
  grid_lasso = GridSearchCV(estimator=model_lasso,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 20)})
  # print(grid_lasso.fit(x,y))
  grid_results_lasso = grid_lasso.fit(x,y)
  print('The optimal hyper-parameter for Lasso is: ', grid_results_lasso.best_params_)
  print('The mean square error for Lasso is: ', np.abs(grid_results_lasso.best_score_))
  print('The L2 for Lasso is: ', np.linalg.norm((betahat_lasso-beta_star), ord=2))

  # Elastic Net
  model_elasticnet = ElasticNet(alpha=0.2,l1_ratio=0.75)
  # model_elasticnet = ElasticNet()
  model_elasticnet.fit(x,y)
  betahat_elasticnet = model_elasticnet.coef_
  print('B̂ (Coefficients of B*) for ElasticNet: ', betahat_elasticnet)
  pos_elasticnet = np.where(betahat_elasticnet != 0)
  print('The average number of true non-zero coefficients for ElasticNet is: ', np.array(pos_elasticnet).shape[1])
  grid_elasticnet = GridSearchCV(estimator=model_elasticnet,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 20)})
  # print(grid_elasticnet.fit(x,y))
  grid_results_elasticnet = grid_elasticnet.fit(x,y)
  print('The optimal hyper-parameter for ElasticNet is: ', grid_results_elasticnet.best_params_)
  print('The mean square error for ElasticNet is: ', np.abs(grid_results_elasticnet.best_score_))
  print('The L2 for ElasticNet is: ', np.linalg.norm((betahat_elasticnet-beta_star), ord=2))

  # Square Root Lasso
  model_sqrtlasso = SQRTLasso(alpha=0.1)
  model_sqrtlasso.fit(x,y)
  betahat_sqrtlasso = model_sqrtlasso.coef_
  print('B̂ (Coefficients of B*) for SQRT Lasso: ', betahat_sqrtlasso)
  pos_sqrtlasso = np.where(betahat_sqrtlasso != 0)
  print('The average number of true non-zero coefficients for SQRT Lasso is: ', np.array(pos_sqrtlasso).shape[1])
  grid_sqrtlasso = GridSearchCV(estimator=model_sqrtlasso,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 20)})
  # print(grid_sqrtlasso.fit(x,y))
  grid_results_sqrtlasso = grid_sqrtlasso.fit(x,y)
  print('The optimal hyper-parameter for SQRT Lasso is: ', grid_results_sqrtlasso.best_params_)
  print('The mean square error for SQRT Lasso is: ', np.abs(grid_results_sqrtlasso.best_score_))
  print('The L2 for SQRT Lasso is: ', np.linalg.norm((betahat_sqrtlasso-beta_star), ord=2))

  # SCAD
  model_scad = SCAD(a=2,lam=1)
  model_scad.fit(x,y)
  betahat_scad = model_scad.coef_
  print('B̂ (Coefficients of B*) for SCAD: ', betahat_scad)
  pos_scad = np.where(betahat_scad != 0)
  print('The average number of true non-zero coefficients for SCAD is: ', np.array(pos_scad).shape[1])
  grid_scad = GridSearchCV(estimator=model_scad,cv=10,scoring='neg_mean_squared_error',param_grid={'a': np.linspace(0, 1, 20), 'lam': np.linspace(0, 1, 20)})
  # print(grid_scad.fit(x,y))
  grid_results_scad = grid_scad.fit(x,y)
  print('The optimal hyper-parameter for SCAD is: ', grid_results_scad.best_params_)
  print('The mean square error for SCAD is: ', np.abs(grid_results_scad.best_score_))
  print('The L2 for SCAD is: ', np.linalg.norm((betahat_scad-beta_star), ord=2))

```


### Final Results: 

___Ridge:___   
B̂ (Coefficients of B*) for Ridge: [ 0.72309017,  0.80814089,  0.68618491, ..., -0.07410177, -0.05327676, -0.0454597 ]   
The average number of true non-zero coefficients for Ridge is:  1200   
The optimal hyper-parameter for Ridge is:  {'alpha': 0.0}   
The mean square error for Ridge is:  35.9727200236592   
The L2 for Ridge is:  2.9767672732983126   

___Lasso:___   
B̂ (Coefficients of B*) for Lasso: [ 0.86667719,  2.61926535,  0.15155155, ... -0.         , -0.        , -0.        ]   
The average number of true non-zero coefficients for Lasso is:  146   
The optimal hyper-parameter for Lasso is:  {'alpha': 0.21052631578947367}   
The mean square error for Lasso is:  15.979340142738845   
The L2 for Lasso is:  3.9375975290985736   

___Elastic Net:___   
B̂ (Coefficients of B*) for ElasticNet: [ 1.09958876,  1.89168061,  0.5217372, ..., -0.        , -0.        , -0.        ]   
The average number of true non-zero coefficients for ElasticNet is:  147   
The optimal hyper-parameter for ElasticNet is:  {'alpha': 0.7368421052631579}   
The mean square error for ElasticNet is:  15.854581370337865   
The L2 for ElasticNet is:  2.668379467426268   

___Square Root Lasso:___   
B̂ (Coefficients of B*) for SQRT Lasso: [ 1.21229487e+00,  1.46690668e+00,  8.33538418e-01, ..., -6.60427298e-10, -1.16705356e-05, -1.98345762e-09 ]   
The average number of true non-zero coefficients for SQRT Lasso is:  1200   
The optimal hyper-parameter for SQRT Lasso is:  {'alpha': 0.15789473684210525}   
The mean square error for SQRT Lasso is:  15.026424585803081   
The L2 for SQRT Lasso is:  2.021002999805211   

___SCAD:___   
B̂ (Coefficients of B*) for SCAD: [ 0.18914786,  0.19739721,  0.20131751, ..., -0.04041534, -0.03022174, -0.03186249 ]   
The average number of true non-zero coefficients for SCAD is:  1200   



**Since we aim to minimize the mean square error (MSE) for the better results, I conclude that the Square Root Lasso achieved the best result compared to all other regressions.**




##### Jekyll Themes
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/r-fukutoku/Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

##### Support or Contact
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
