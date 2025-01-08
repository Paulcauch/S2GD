import numpy as np 
from scipy.linalg import norm 
import time 

#S2GD and S2GD+
def S2GD(x0,problem,xtarget,h,m,nu,eps_tol,plus=False,alpha=2,n_iter=100,verbose=True): 
    """
        S2GD Semi Stochastic Gradient Descent. [Konecny 2013]
        
        Inputs:
            x0: Initial vector
            problem
            xtarget: Target minimum 
            h : stepsize if < 0, we choose the best parameters m,h,n_iter
            m : max of # of Stochastic Gradient per epoch
            nu : lower bound on mu (the strongly convex cste of f)
            plus : Boolean if True -> Run SG2D+ 
            alpha : SG2D+ scaling parameter for # of Stochastic Gradient per epoch
            n_iter: Number of iterations, used as stopping criterion
            verbose: Boolean indicating whether information should be plot at every iteration (Default: False)
            
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
    kappa = problem.kappa_val
    n = problem.n
    x = x0.copy()
    nx = norm(x)


    #Compute best parameters
    #old
    if h < 0 : 
        n_iter = int(np.floor(np.log( 1 / eps_tol )) + 1)
        delta = eps_tol ** (1/n_iter)
        h = 1 / ((4/delta) * (L - mu) + 2*L)
        if nu == mu: 
            m = (4*(kappa - 1)/delta + 2*kappa) * np.log(2/delta + (2*kappa - 1)/(kappa - 1))
            m = int(m) + 1
        elif nu == 0:
            m = 8*(kappa - 1) / delta**2 + 8*kappa/delta + (2*kappa**2) / (kappa-1)
            m = int(m) + 1
        h = h / 3
        print("h=",h,"m=",m,"n_iter=",n_iter,"delta=",delta,)

    #new 
    # if h < 0 : 
    #     n_iter = int(np.floor(np.log( 1 / eps_tol )) + 1)
    #     delta = eps_tol ** (1/n_iter)
    #     if nu == 0 :
    #         m = 8*(kappa - 1) / delta**2 + 8*kappa/delta 
    #         m = int(m) + 1
    #         h = 1 / ((4/delta) * (L - mu) + 4*L)
    #     elif nu == mu :    
    #         h = 1 / ((4/delta) * (L - mu) + 2*L)
    #         if kappa < 2 : 
    #             m = (4*(kappa - 1)/delta + 2*kappa) * np.log(2/delta + (2*kappa - 1)/(kappa - 1))
    #             m = int(m) + 1
    #         else :
    #             m = (6*kappa/delta) * np.log(5/delta)
    #             m = int(m) + 1
    #     h = h / 10

    #Compute probability and expectaiton for t
    t_val = np.arange(1,m+1) 
    t_prob = (1- nu*h)**(m-t_val)
    t_probs = t_prob / np.sum(t_prob)
    t_expectation = np.dot(t_val,t_probs)
    t = 0

    print("h=",h,"m=",m,"n_iter=",n_iter,"beta=",np.sum(t_prob))

    k=0
    
    # Current objective
    obj = problem.fun(x) 
    objvals.append(obj)

    # Current distance to the optimum
    nmin = norm(x-xtarget)
    normits.append(nmin)
     
    start_time = time.time()
    
    if verbose:
        # Display initial quantities of interest
        if plus: 
            print("S2GD+, n=",n,".  Number of inner loop at each iteration:",alpha*n)
            print("Running a single pass of Stochastic Gradient:")
            print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))
        else :
            print(f"S2GD, n={n}. Exepcted number of inner loop at each iteration: {int(t_expectation)} in [{int((m+1)/2)},{int(m)})")
            print(' | '.join([name.center(8) for name in ["iter", "fval", "normit","t"]]))
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8),("%.2e" % t).rjust(8)]))
    
    if plus :
        ik = np.random.choice(n,n,replace=False)
        for i in ik : 
            x = x - h * problem.grad_i(x,i)
        obj = problem.fun(x)
        objvals.append(obj)
        nmin = norm(x-xtarget)
        normits.append(nmin)
        nb_grad_comp.append(n)

        end_time = time.time()
        times.append(end_time - start_time)

        if verbose : 
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))
            print("Running S2GD with t=", alpha*n)
            print(' | '.join([name.center(8) for name in ["iter", "fval", "normit","t"]]))
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8),("%.2e" % t).rjust(8)]))

    # Main loop
    while (k < n_iter and nx < 10**100):
       
        #Compute full gradient 
        g = problem.grad(x)

        #init
        y = np.copy(x)

        #Draw the number of sto grad 
        if plus : 
            t = alpha * n
        else : 
            t = np.random.choice(t_val,size=1,p=t_probs)[0]
            #print("t=",t)

        #Inner Loop
        for _ in range(t):
            ind = np.random.choice(n,1,replace=True)
            y[:] = y - h * (g + problem.grad_i(y,ind) - problem.grad_i(x,ind))
        
        #Update
        x = y 

        k += 1
        
        nx = norm(x) 
        obj = problem.fun(x)
        nmin = norm(x-xtarget)

        objvals.append(obj)
        normits.append(nmin)
        nb_grad_comp.append(nb_grad_comp[-1] + n + 2*t)

        if verbose:
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8),("%.2e" % t).rjust(8)]))

        if abs(problem.fun(x)-problem.fun(xtarget)) < eps_tol*abs(problem.fun(x0)-problem.fun(xtarget)):
            # print(problem.fun(x)-problem.fun(xtarget))
            # print(eps_tol*abs(problem.fun(x0)-problem.fun(xtarget)))
            print("Algorithm end because it reached precision.")
            break

        end_time = time.time()
        times.append(end_time - start_time)
    
    #print(problem.fun(x)-problem.fun(xtarget))
    #print(eps_tol*abs(problem.fun(x0)-problem.fun(xtarget)))
    # Outputs
    x_output = x.copy()
    
    return x_output, np.array(objvals), np.array(normits), np.array(nb_grad_comp), times