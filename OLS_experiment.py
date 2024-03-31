import numpy as np
import pandas as pd
import statsmodels.api as sm


class Task:
    '''
    sample task
    '''

    def __init__(self, w):        
        self.w = w
        
    def sample(self, N, VarE=0):
        '''
        obtain N samples from a linear funciton with weight
        '''
        # torch.randn generates tensor with normal distribution (mean=0, std=1)
        x = np.random.normal(loc=0, scale=1, size=(N,10))
        e = np.random.normal(loc=0, scale=VarE**0.5, size=N)
        y = np.matmul(x, self.w) +e
        
        return x, y

def sample_task():    
    w = np.random.normal(loc=0, scale = 1, size=(10))
    return Task(w)

N_list = [5, 10, 15, 20]
VarE_list = [0, 0.1, 0.3, 0.5]
    
results = list()

for n in N_list:
    for varE in VarE_list:
        example_task = sample_task()
        x, y = example_task.sample(n+1, varE)

        model = sm.OLS(y[:n-1],x[:n-1,:])
        model_fit = model.fit()

        y_test = y[-1]
        y_pred = model_fit.predict(x[-1,:])
        MSE = np.mean((y_test - y_pred)**2)

        results.append([n, varE, MSE])

results_df = pd.DataFrame(results, columns=['N','VarE', 'Loss'])        
results_df.to_csv('Test_error_OLS_20240330.csv', index=False)


