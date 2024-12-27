import numpy as np
from scipy.linalg import norm
import time

def SAGA(x0,problem,xtarget,gamma,eps_tol,n_iter=100,verbose=True,fast=False): 
    """
        SAGA Stochastic Average Gradient Augmented. [Defazio 2014]
        
        Inputs:
            x0: Initial vector
            problem
            xtarget: Target minimum (unknown in practice!)
            gamma : stepsize if < 0, we choose the best gamma 
            n_iter: Number of iterations, used as stopping criterion
            verbose: Boolean indicating whether information should be plot at every iteration (Default: False)
            fast: Boolean indicating whether we want to compute info every iteration or only of "one pass on the data"
            
        Outputs:
            x_output: Final iterate of the method 
            objvals: History of function values (Numpy array of length n_iter at most)
            normits: History of distances between iterates and optimum (Numpy array of length n_iter at most)
            nbgradcomp : History of number of gradient evaluated at every iteration
    """
    objvals = []
    normits = []
    times = [0]
    nb_grad_comp = [0]

    L = problem.lipgrad()
    mu = problem.cvxval()
    n = problem.n
    d = problem.d
    x = x0.copy()
    nx = norm(x)

    memory_table = np.zeros((n,d))
    grad_mean = np.zeros(d)
                            

    #Compute best parameters
    if gamma < 0 : 
        if mu == 0 :
            gamma = 1 / (3*L)
        else : 
            gamma =  1 / (2*(mu*n+L)) 

    k=0
    
    # Current objective
    obj = problem.fun(x) 
    objvals.append(obj)

    # Current distance to the optimum
    nmin = norm(x-xtarget)
    normits.append(nmin)
    
    if verbose:
        # Display initial quantities of interest
        print(f"SAGA, n={n} with stepsize gamma = {gamma}")
        print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
        print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))

    start_time = time.time()
    # Main loop
    while (k < n_iter and nx < 10**100):

        #Pick j uniformly at random
        j = np.random.randint(0,n)

        #Compute grad_j of x 
        phi_j = problem.grad_i(x,j) 

        #Compute the update
        v = phi_j - memory_table[j] + grad_mean #np.mean(memory_table,axis=0)

        #Updata x 
        x = x - gamma * v

        #Compute the mean gradient
        grad_mean += (-memory_table[j] + phi_j) / n 

        #Store phi in the table 
        memory_table[j] = phi_j

        
        k += 1

        nx = norm(x)
        
        if not(fast):  
            obj = problem.fun(x)
            nmin = norm(x-xtarget)
            objvals.append(obj)
            end_time = time.time()
            times.append(end_time - start_time)
            normits.append(nmin)
            nb_grad_comp.append(k)

            if verbose:
                print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))
        else : 
            #Compute every epochs
            if k % n == 0 :
                nx = norm(x) 
                obj = problem.fun(x)
                nmin = norm(x-xtarget)
                objvals.append(obj)
                end_time = time.time()
                times.append(end_time - start_time)
                normits.append(nmin)
                nb_grad_comp.append(k)

                if verbose:
                    print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))


        if not(fast):
            if abs(problem.fun(x)-problem.fun(xtarget)) < eps_tol*abs(problem.fun(x0)-problem.fun(xtarget)):
                print("Algorithm end because it reached precision.")
                break
    
    # Outputs
    x_output = x.copy()
    
    return x_output, np.array(objvals), np.array(normits), np.array(nb_grad_comp), times 