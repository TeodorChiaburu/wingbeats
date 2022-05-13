"""Library for model callbacks to be used during training"""



# Import libraries
import numpy as np



def lr_triangle(epoch, max_lr, max_ep = 30, step_size = 15):
    """Define a decreasing triangular learning rate schedule. \    
    Each isosceles triangle represents a new cycle with length 2 * **step_size**. 
    
    :param epoch: Current epoch.
    :type epoch: int
    :param max_lr: Maximum learning rate.
    :type max_lr: float
    :param max_ep: How many epochs the pattern should repeat before the learning rate stays constant. Defaults to 30.
    :type max_ep: int
    :param step_size: Number of epochs until monotony reverses (half of triangle). Defaults to 15.
    :type step_size: int
    :return: Learning rate
    :rtype: float
    """ 
        
    min_lr = max_lr * 0.1 # lr at the base of triangle
    
    if epoch < max_ep:
        cycle = np.floor(1+epoch/(2*step_size)) # current cycle/triangle
        x = np.abs(epoch/step_size - 2*cycle + 1) # local time point in the current triangle
        lr = min_lr + (max_lr-min_lr) * np.maximum(0, (1-x)) / float(2**(cycle-1))
    else:
        lr = min_lr

    return lr

#############################################################################################

def lr_exp(epoch, max_lr, max_ep = 30):
    """Define an exponentially decaying learning rate schedule.
    
    :param epoch: Current epoch.
    :type epoch: int
    :param max_lr: Maximum learning rate.
    :type max_lr: float
    :param max_ep: How many epochs the pattern should repeat before the learning rate stays constant. Defaults to 30.
    :type max_ep: int
    :return: Learning rate
    :rtype: float
    """
    
    # Model: current_lr = max_lr * rate ** epoch
    # Restriction: min_lr = max_lr * rate ** max_ep
    #              => ln(rate) = decay_rate [see below]
    
    min_lr = max_lr * 0.01 # lower limit of decay

    if epoch < max_ep:
        decay_rate = (np.log(min_lr) - np.log(max_lr)) / max_ep 
        lr = max_lr * np.exp(decay_rate*epoch)
    else:
        lr = min_lr

    return lr


