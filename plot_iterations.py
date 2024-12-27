import matplotlib.pyplot as plt
import numpy as np 


def plot_with_nb_gradients(nb_gradients, obj_functions, name_of_models):
    """ 
    Plot the iterations of ||f(x) - f(x*)|| in function of the number 
    of gradienbts calculated.

    Args
    ----
    nb_gradients : for each row, number of gradients calculated at each iteration for each model
    obj_functions : array containing for each row all values of the objective 
                   function for one model at each iteration, and each row correponding 
                   to a model
    name_of_models : list containing the name of each model comprised in obj_functions
    """
    N = len(obj_functions)
    colors = ['red', 'blue', 'green', 'black', 'pink']

    for k in range(N):
        final_value_f = obj_functions[k][-1]

        plt.semilogy(nb_gradients[k], np.array(obj_functions[k]) - final_value_f, label=name_of_models[k], marker = 's', color=colors[k])
        plt.xlabel('Stochastic gradient evaluations')
        plt.ylabel(r"$||f(x) - f(x^*)||$")
        plt.title('Comparison between the different models')
        plt.legend()
        
    plt.grid()
    plt.show()



def plot_with_time(times, obj_functions, name_of_models):
    """ 
    Plot the iterations of ||f(x) - f(x*)|| in function of the number 
    of gradienbts calculated.

    Args
    ----
    nb_gradients : time of exectution beginning at iteration 0 until the end
    obj_functions : array containing for each row all values of the objective 
                   function for one model at each iteration, and each row correponding 
                   to a model
    name_of_models : list containing the name of each model comprised in obj_functions
    """
    N = len(obj_functions)
    colors = ['red', 'blue', 'green', 'black', 'pink']

    for k in range(N):
        final_value_f = obj_functions[k][-1]

        plt.semilogy(times[k], np.array(obj_functions[k]) - final_value_f, label=name_of_models[k], marker = 's', color=colors[k])
        plt.xlabel('time (s)')
        plt.ylabel(r"$||f(x) - f(x^*)||$")
        plt.title('Comparison between the different models')

    plt.show()
