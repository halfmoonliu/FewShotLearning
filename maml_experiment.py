import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad
from tqdm import tqdm

def mlp(x, params):
    '''
    multi layer perceptron function
    2 hidden layers, 40 neurons each
    '''
    h = torch.relu(torch.nn.functional.linear(x, params[0], bias=params[1]))
    h = torch.relu(torch.nn.functional.linear(h, params[2], bias=params[3]))
    return torch.nn.functional.linear(h, params[4], bias=params[5])


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
        x = torch.randn(N, 10)
        e = torch.normal(mean=0, std=torch.ones(10)*(VarE**0.5))
        y = torch.matmul(x, self.w) + e
        loss_fct = nn.MSELoss()
        return x, y, loss_fct

@torch.no_grad()
def sample_task():
    mu = torch.ones(10)
    w = torch.normal(mu, torch.eye(10))
    return Task(w)

def perform_k_training_steps(params, 
                             task, 
                             N, 
                             VarE,
                             inner_steps, 
                             alpha, 
                             device='cpu'):
    # loop through epochs of inner steps
    for epoch in range(inner_steps):
        x_batch, target, loss_fct = task.sample(N, VarE)
        loss = loss_fct(mlp(x_batch.to(device), params), 
                        target.to(device))

        for p in params:  # Zero grad
            p.grad = None
        gradients = grad(loss, params)
        for p, g in zip(params, gradients):  # Grad step
            p.data -= alpha * g
    return params


def maml(p_model, 
         meta_optimizer, 
         inner_steps, 
         n_epochs, 
         N, 
         VarE,
         alpha, 
         M=10, 
         device='cpu'):
    """
    Execute few-shot supervised learning using MAML
    """
    training_loss = []
    # Line 2 in the pseudocode
    for epoch in tqdm(range(n_epochs)):  
        # parameters
        theta_i_prime = []
        # data points
        D_i_prime = []

        # Sample batch of tasks
        # Line 3 in the pseudocode
        tasks = [sample_task() for _ in range(M)]  
        for task in tasks:
            theta_i_prime.append(perform_k_training_steps([p.clone() for p in p_model], task, N,
                                                          VarE,
                                                          inner_steps, alpha, device=device))
            # Sample data points Di' for the meta-update (line 8 in the pseudocode)
            x, y, loss_fct = task.sample(25,VarE)
            D_i_prime.append((x, y, loss_fct))

        # Meta update
        meta_optimizer.zero_grad()
        batch_training_loss = []
        for i in range(M):
            x, y, loss_fct = D_i_prime[i]
            f_theta_prime = theta_i_prime[i]
            # Compute \nabla_theta L(f_theta_i_prime) for task ti
            loss = loss_fct(mlp(x.to(device), f_theta_prime), y.to(device))
            loss.backward()
            batch_training_loss.append(loss.item())

        meta_optimizer.step()  # Line 10 in the pseudocode
        training_loss.append(np.mean(batch_training_loss))
    return training_loss


if __name__ == "__main__":
    device = 'cpu'
    params = [torch.rand(40, 10, 
                         device=device).uniform_(-np.sqrt(6. / 41), 
                                                 np.sqrt(6. / 41)).requires_grad_(),
              torch.zeros(40, device=device).requires_grad_(),
              torch.rand(40, 40, 
                         device=device).uniform_(-np.sqrt(6. / 80), 
                                                 np.sqrt(6. / 80)).requires_grad_(),
              torch.zeros(40, device=device).requires_grad_(),
              torch.rand(1, 40, 
                         device=device).uniform_(-np.sqrt(6. / 41), 
                                                 np.sqrt(6. / 41)).requires_grad_(),
              torch.zeros(1, device=device).requires_grad_()]

    device = 'cpu'
    meta_optimizer = torch.optim.Adam(params, lr=1e-3)
    N_list = [5, 10, 15, 20]
    M_list = [5, 10, 50, 100]
    VarE_list = [0, 0.1, 0.3, 0.5]
    
    results = list()
    for m in M_list:
        for n in N_list:
            for varE in VarE_list:
                training_loss = maml(p_model = params, 
                                 meta_optimizer = meta_optimizer, 
                                 inner_steps = 10, 
                                 n_epochs = 1000, 
                                 N = n, 
                                 VarE = varE,
                                 alpha = 1e-3, 
                                 M = m,
                                 device=device)


                task_new = sample_task()
                x_new, y_new, loss_func = task_new.sample(n+1, varE)
                y_hat_raw = mlp(x_new.to(device), params)

                test_loss_raw = loss_func(y_hat_raw[-1], 
                                y_new[-1].to(device))
                results.append([m, n, varE,test_loss_raw.item()])
    results_df = pd.DataFrame(results, columns=['M', 'N','VarE', 'Loss'])        
    results_df.to_csv('results_error_20240330.csv', index=False)
            
            

