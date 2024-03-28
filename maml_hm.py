import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import grad


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
        
    def sample(self, N):
        '''
        obtain N samples from a linear funciton with weight
        '''
        # torch.randn generates tensor with normal distribution (mean=0, std=1)
        x = torch.randn(N, 10)
        y = torch.matmul(x, self.w)
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
                             inner_steps, 
                             alpha, 
                             device='cpu'):
    # loop through epochs of inner steps
    for epoch in range(inner_steps):
        x_batch, target, loss_fct = task.sample(N)
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
                                                          inner_steps, alpha, device=device))
            # Sample data points Di' for the meta-update (line 8 in the pseudocode)
            x, y, loss_fct = task.sample(25)
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
    training_loss = maml(p_model = params, 
                         meta_optimizer = meta_optimizer, 
                         inner_steps = 1, 
                         n_epochs = 2000, 
                         N = 10, 
                         alpha = 1e-3, 
                         M = 10,
                         device=device,)

    task_new = sample_task()
    x_new, y_new, loss_func = task_new.sample(10)
    y_hat_raw = mlp(x_new.to(device), params)

    test_loss_raw = loss_func(y_hat_raw, 
                        y_new.to(device))
    print(test_loss_raw)
    
    new_params = perform_k_training_steps([p.clone() for p in params], 
                                          task=task_new, 
                                          N = 10, 
                                          inner_steps = 10, 
                                          alpha = 1e-3, 
                                          device=device)
    # After 10 gradient steps
    y_hat_tuned = mlp(x_new.to(device), new_params)
    test_loss_tuned = loss_func(y_hat_tuned, 
                        y_new.to(device))
    
    print(test_loss_tuned)

    