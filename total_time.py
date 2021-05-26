#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:06:21 2021

@author: maximilianottosson
"""
from mean_var_as_paramters import normal_mean_var as normal
import math
import random
from timeit import default_timer as timer
from mean_var_as_paramters import gamma_mean_var as gamma
from scipy.optimize import brentq
import numpy
from scipy.stats import uniform 


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
        starting standard deviation
 
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
    start = timer()
    time = []
    mu_sigma_list = [(mu_1,sigma_1)]
    sample = [mu_1]
    error_in_brentq = 0
    j = 1
    i = 0
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
        end = timer()
        time.append(end - start)

    return((time,sample))


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
        starting standard deviation
 
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
    start = timer()
    time = []
    mu_sigma_list = [(mu_1,sigma_1)]
    sample = [mu_1]
    j = 1
    i = 0
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
            j += 1

        i += 1
        end = timer()
        time.append(end - start)
    return([time,sample])


def distributions_by_mixtures_Algorithm_3_gausian(i_max, mu_1, sigma_1, P):
    '''
    Parameters:
    P : function
        P(x) the target distribution.

    g : function
        g(x) the proposal distribution

    i_max : integer
        the size of the sample

    mu_1 : float or integer
        starting mean

    sigma_1 : float or integer
        starting standard deviation

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
    A sample from P(x) using the Approximating prescribed distributions by mixtures
    in the guasian case from Edgar Bueno article 
    '''
    start = timer()
    time = [0]
    n_1 = 2
    j = 2
    i = 2
    sample = [mu_1]
    P_x = []
    mu_sigma_list = numpy.full((1,2),(mu_1,sigma_1))
    x1n1_list = [float(normal.rvs(mu_1,sigma_1,1)),float(normal.rvs(mu_1,sigma_1,1))]
    Px_list = list(map(lambda x_1n1: P(x_1n1),x1n1_list))
    
    
    while max(Px_list) == 0:
        n_1 += 1
        x_1i = float(normal.rvs(mu_1,sigma_1,1))
        Px_1i = P(x_1i)
        Px_list.append(Px_1i)
        x1n1_list.append(x_1i)

    gx_list = list(map(lambda x_1n1: normal.pdf(x_1n1, mu_1, sigma_1),x1n1_list))
    gx_list = normal.pdf(x1n1_list,mu_1, sigma_1)
    ratio = Px_list/gx_list
    t_hat_1 = (1/n_1)*sum(ratio)
    indic = numpy.argmax(ratio)
    
    x_1 = x1n1_list[indic]
    sample.append(x_1)
    P_x.append(P(x_1))
    constant = (P(x_1)/t_hat_1)*2 - normal.pdf(x_1,mu_1,sigma_1)
    sigma_2 = 1/((2*math.pi)*(constant**2))

    mu_sigma_list = numpy.vstack((mu_sigma_list,(x_1,sigma_2)))
    vk = numpy.full((1,2),(1,2))
    vk = [1,2]
    t_hat_i = [t_hat_1]
    
    while i < i_max:
        g_j_sample = random.choice(mu_sigma_list)
        x_i = float(normal.rvs(g_j_sample[0],g_j_sample[1],1))
        sample.append(x_i)
        P_xi = P(x_i)
        g_xi = numpy.mean(normal.pdf(x_i,mu_sigma_list[:,0],mu_sigma_list[:,1]))
        t_hat_i.append(P_xi/g_xi)
        t_hat = numpy.sum(numpy.multiply(vk,t_hat_i))/(i*(i+1)/2)
        p_xi = float(P_xi)/t_hat
        vk.append(i+1)

        if p_xi > g_xi:
            constant = p_xi*(j+1)- j*g_xi
            sigma_j_1 = 1/((2*math.pi)*(constant**2))
            mu_sigma_list = numpy.vstack((mu_sigma_list,(x_i,sigma_j_1)))
            i += 1
            j += 1

        else:
            i += 1
        end = timer()
        time.append(end - start)

    return(time,sample)

def distributions_by_mixtures_Algorithm_3_gamma(i_max, mu_1, sigma_1, P):

    '''
    Parameters:
    p : function
        p(x) the target distribution.

    i_max : integer
        the size of the sample

    mu_1 : float or integer
        starting mean

    sigma_1 : float or integer
        starting standard deviation

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
    in the guasian case from Edgar Bueno article
    '''
    start = timer()
    n_1 = 1
    j = 2
    i = 2
    sample = [mu_1]
    P_x = []
    time = [0]
    mu_sigma_list = numpy.full((1,2),(mu_1,sigma_1))
    x1n1_list = [float(gamma.rvs(mu_1,sigma_1,1)),float(gamma.rvs(mu_1,sigma_1,1))]
    Px_list = list(map(lambda x_1n1: P(x_1n1),x1n1_list))
    
    while max(Px_list) == 0:
        n_1 += 1
        x_1i = float(gamma.rvs(mu_1,sigma_1,1))
        Px_1i = P(x_1i)
        Px_list.append(Px_1i)
        x1n1_list.append(x_1i)
    
    gx_list = gamma.pdf(x1n1_list,mu_1, sigma_1)
    ratio = Px_list/gx_list
    t_hat_1 = (1/n_1)*sum(ratio)
    indic = numpy.argmax(ratio)
    
    x_1 = x1n1_list[indic]
    sample.append(x_1)
    P_x.append(P(x_1))
    constant = (P(x_1)/t_hat_1)*2 - gamma.pdf(x_1,mu_1,sigma_1)
    sigma_2 = 1/((2*math.pi)*(constant**2))

    mu_sigma_list = numpy.vstack((mu_sigma_list,(x_1,sigma_2)))
    vk = numpy.full((1,2),(1,2))
    vk = [1,2]
    t_hat_i = [t_hat_1]
    end = timer()
    time.append(end - start)
    while i < i_max:
        g_j_sample = random.choice(mu_sigma_list)
        x_i = float(gamma.rvs(g_j_sample[0],g_j_sample[1],1))
        sample.append(x_i)
        P_xi = P(x_i)
        g_xi = numpy.mean(gamma.pdf(x_i,mu_sigma_list[:,0],mu_sigma_list[:,1]))
        t_hat_i.append(P_xi/g_xi)
        t_hat = numpy.sum(numpy.multiply(vk,t_hat_i))/(i*(i+1)/2)
        p_xi = float(P_xi)/t_hat
        vk.append(i+1)

        if p_xi > g_xi:
            constant = p_xi*(j+1)- j*g_xi
            func = lambda sigma: gamma.pdf(x_i, x_i,sigma) - constant
            try:
                sigma_j_1 = brentq(func,1e-210,999999999)
            except ValueError:
                sample.pop(i)
                i - 1
                continue
            mu_sigma_list = numpy.vstack((mu_sigma_list,(x_i,sigma_j_1)))
            j += 1
        

        i += 1
        end = timer()
        time.append(end - start)
    return((time,sample))

def Metrpolis_Hastings_gausian_proposal(i_max,inital,var,target):
    start = timer()
    time = [0]
    current = inital
    sample = [inital]
    i = 0
    while i < i_max:
        
        proposal = normal.rvs(current,var,1)
        ratio = target(proposal)/target(current)
        alpha = min([1,ratio])
        u = uniform.rvs(0,1)

        if u < alpha:
            sample.append(float(proposal))
            current = proposal
        else:
            sample.append(float(current))
        i += 1
        end = timer()
        time.append(end - start)
    return((time,sample))

def Metrpolis_Hastings_gamma_proposal(inital,var,i_max,target):
    start = timer()
    current = inital
    sample = [inital]
    time = []
    i = 0
    
    while i < i_max:
        try:
            proposal = gamma.rvs(current,var,1)
        except ValueError:
            if current < 1e-155:
                current = 1e-155
                proposal = gamma.rvs(current,var,1)
            else:
                print('proposal error')
                break

        try:
            ratio = (target(proposal)*gamma.pdf(current,proposal,var))/(target(current)*gamma.pdf(proposal,current,var))
        except ValueError:
            print('Problem ratio')
            break   
        alpha = min([1,ratio])
        u = uniform.rvs(0,1)

        if u < alpha:
            sample.append(float(proposal))
            current = proposal
        else:
            sample.append(float(current))

        i += 1
        end = timer()
        time.append(end - start)
        
    return((time,sample))                