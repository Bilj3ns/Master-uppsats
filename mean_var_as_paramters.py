#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import norm
from scipy.stats import gamma

class normal_mean_var(object):
    '''
    Reparametrize the norm function from scipy.stats
    Instead of taking the mean and standard devetion it now takes
    mean and varation as parameters.
    '''
    def __init__(self, x, mean, var, size):
        self.x = x
        self.mean = mean
        self.var = var
        self.size = size
        
    def pdf(x,mean,var):
        '''
    Parameters:
    x : float or int
        
    mean : float or int
        the desired mean of the norm-pdf

    var: float or int
        the desired var of the norm-pdf


    Returns
    -------
    Retuns value: the value of the pdf at point x.
    '''
        return(norm.pdf(x,mean,var**0.5))
    
    def rvs(mean,var,size):
        '''
    Parameters:
        
    mean : float or int
        the desired mean of the norm-pdf

    var: float or int
        the desired var of the norm-pdf
    
    size: int
        the desired size of the sample from norm(mean,var)

    Returns
    -------
    Returns value: a random sample from norm(mean,var)
    '''
        return(norm.rvs(mean,var**0.5,size))
    
class gamma_mean_var(object):
    '''
    Reparametrize the gamma function from scipy.stats
    Instead of taking the shape and scale it now takes
    mean and varation as parameters.
    '''
    def __init__(self, x, mean, var, size):
        self.x = x
        self.mean = mean
        self.var = var
        self.size = size
        
    def pdf(x,mean,var):
        '''
    Parameters:
    x : float or int
        
    mean : float or int
        the desired mean of the reparameterized-gamma-pdf

    var: float or int
        the desired var of the reparameterized-gamma-pdf


    Returns
    -------
    Retuns value: the value of the reparameterized-gamma(mean,var)-pdf at point x.
    '''
        shape = (mean**2)/var
        scale = var/mean
        return(gamma.pdf(x,shape,0,scale))
    
    def rvs(mean,var,size):
        '''
    Parameters:
        
    mean : float or int
        the desired mean of the reparameterized-gamma-pdf

    var: float or int
        the desired var of the reparameterized-gamma-pdf
    
    size: int
        the desired size of the sample from norm(mean,var)

    Returns
    -------
    Returns value: a random sample from reparameterized-gamma(mean,var)
    '''
        shape = (mean**2)/var
        scale = var/mean
        return(gamma.rvs(shape,0,scale,size))
     
    

