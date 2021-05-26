#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distributions_by_mixtures_Algorithm_1

Created on Mon Feb  8 12:11:34 2021

@author: maximilianottosson
"""

from mean_var_as_paramters import normal_mean_var as normal
from mean_var_as_paramters import gamma_mean_var as gamma
from scipy.optimize import brentq
import math
import random
import numpy 


def distributions_by_mixtures_Algorithm_1_gamma(i_max, mu_1, sigma_1, p):
    '''
    Parameters:
    p : function 
        p(x) the target distribution.
        
    i_max : integer 
        the size of the sample 

    mu_1 : float or integer 
        starting mean

    sigma_1 : float or integer 
        starting var
 
    Returns
    -------
    Retun value: a list of lits:
        sample: 
            array or float
        mu_list: 
            array or float
        sigma_list: 
            array or float
        mu_list: 
            array or float
        
    Description:
    A sample from p(x) using the Approximating prescribed distributions by mixtures 
    in the guasian case from Edgar Bueno article with gamma proposal
    '''
    error_in_brentq = 0
    j = 1
    i = 0
    mu_sigma_list = [(mu_1,sigma_1)]
    sample = [mu_1]
    mu_sigma_list = numpy.vstack((mu_sigma_list,(mu_1,sigma_1)))
    
    while i < i_max:
        g_j_sample = random.choice(mu_sigma_list)
        x_i = float(gamma.rvs(g_j_sample[0],g_j_sample[1],1))
        sample.append(x_i)
        px_i = p(x_i)
        gx_i = numpy.mean(normal.pdf(x_i,mu_sigma_list[:,0],mu_sigma_list[:,1]))
        px_i = float(px_i)

        if px_i > gx_i:
            constant = px_i*(j+1)- j*gx_i
            func = lambda sigma: gamma.pdf(x_i, x_i,sigma) - constant
            try:
                sigma_j_1 = brentq(func,1e-210,99999999)
            except ValueError:
                sample.pop(i)
                i - 1
                error_in_brentq += 1
                continue
            mu_sigma_list = numpy.vstack((mu_sigma_list,(x_i,sigma_j_1)))
            i += 1
            j += 1

        else:
            i += 1
    print('There were ' + str(error_in_brentq) + ' errors in brentq of ' +str(i) + ' iterations')
    return([sample,mu_sigma_list])


def distributions_by_mixtures_Algorithm_1_gausian(i_max, mu_1, sigma_1, p):
    '''
    Parameters:
    p : function 
        p(x) the target distribution.
        
    i_max : integer 
        the size of the sample 

    mu_1 : float or integer 
        starting mean

    sigma_1 : float or integer 
        starting var
 
    Returns
    -------
    Retun value: a list of lits:
        sample: 
            array or float
        mu_list: 
            array or float
        sigma_list: 
            array or float
        mu_list: 
            array or float
        
    Description:
    A sample from p(x) using the Approximating prescribed distributions by mixtures 
    in the guasian case from Edgar Bueno article with normal proposal
    '''
    j = 1
    i = 0
    mu_sigma_list = [(mu_1,sigma_1)]
    sample = [mu_1]
    mu_sigma_list = numpy.vstack((mu_sigma_list,(mu_1,sigma_1)))
    
    while i < i_max:
        g_j_sample = random.choice(mu_sigma_list)
        x_i = float(normal.rvs(g_j_sample[0],g_j_sample[1],1))
        sample.append(x_i)
        px_i = p(x_i)
        gx_i = numpy.mean(normal.pdf(x_i,mu_sigma_list[:,0],mu_sigma_list[:,1]))
        px_i = float(px_i)

        if px_i > gx_i:
            mu_j_1 = x_i  # Mu restriction          
            sigma_j_1 = 1 / (2*math.pi*((px_i*(j+1)-j*gx_i)**2))
            mu_sigma_list = numpy.vstack((mu_sigma_list,(mu_j_1,sigma_j_1)))
            i += 1
            j += 1

        else:
            i += 1
    return([sample,mu_sigma_list])

def p(x):
    if x > 0:
        return(x*math.e**((-x**2)/2))
    else:
        return(0)
    


