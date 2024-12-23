import numpy as np 
from scipy.linalg import norm 



def stoch_grad(x0,problem,xtarget,stepchoice=0,step0=1, n_iter=1000,nb=1,with_replace=False,verbose=True,fast=False): 
    """
        Stochastic gradient descent.
        
        Inputs:
            x0: Initial vector
            problem
            xtarget: Target minimum (unknown in practice!)
            stepchoice: Strategy for computing the stepsize 
                0: Constant step size equal to 1/L
                <0: Step size decreasing in 1/(k+1)**t
                >0: Constant step size equal to step
            step0: Initial steplength (only used when stepchoice is not 0)
            n_iter: Number of iterations, used as stopping criterion
            nb: Number of components drawn per iteration/Batch size 
            with_replace: Boolean indicating whether components are drawn with or without replacement
                True: Components drawn with replacement
                False: Components drawn without replacement (Default)
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
    nb_grap_comp = [0]

    L = problem.lipgrad()
    n = problem.n
    x = x0.copy()
    nx = norm(x)


    k=0
    
    # Current objective
    obj = problem.fun(x) 
    objvals.append(obj)

    # Current distance to the optimum
    nmin = norm(x-xtarget)
    normits.append(nmin)
    
    if verbose:
        # Display initial quantities of interest
        print("Stochastic Gradient, batch size=",nb,"/",n)
        print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
        print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))
    
    ################
    # Main loop
    while (k < n_iter and nx < 10**100):
        # Draw the batch indices
        ik = np.random.choice(n,nb,replace=with_replace)# Batch gradient
        # Stochastic gradient calculation
        sg = np.zeros(problem.d)
        for j in range(nb):
            gi = problem.grad_i(x,ik[j])
            sg = sg + gi
        sg = (1/nb)*sg
            
        if stepchoice==0:
            x[:] = x - (step0/L) * sg
        elif stepchoice>0:
            sk = float(step0/((k+1)**stepchoice))
            x[:] = x - sk * sg
        
        
        k += 1

        nx = norm(x) #Computing the norm to measure divergence 

        if not(fast):
            obj = problem.fun(x)
            nmin = norm(x-xtarget)
            objvals.append(obj)
            nb_grap_comp.append(k*nb)
            normits.append(nmin)

            if verbose:
                print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)])) 
        else:
            if (k*nb) % n == 0:
                obj = problem.fun(x)
                nmin = norm(x-xtarget)
                objvals.append(obj)
                nb_grap_comp.append(k*nb)
                normits.append(nmin)

                if verbose:
                    print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))  


    # Plot quantities of interest for the last iterate (if needed)
    if (k*nb) % n > 0:
        objvals.append(obj)
        normits.append(nmin)
        if verbose:
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))              
    
    # Outputs
    x_output = x.copy()
    
    return x_output, np.array(objvals), np.array(normits), nb_grap_comp