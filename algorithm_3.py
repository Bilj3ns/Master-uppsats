#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:57:49 2021

@author: Edgar Bueno and Maximilian Ottosson
"""

import math
from mean_var_as_paramters import normal_mean_var as normal
from mean_var_as_paramters import gamma_mean_var as gamma
from scipy.optimize import brentq
import numpy
import random

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
    n_1 = 2
    j = 2
    i = 2
    sample = []
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

    return([sample,mu_sigma_list])



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
    n_1 = 1
    j = 1
    i = 1
    sample = []
    P_x = []
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

    return([sample,mu_sigma_list])





from scipy.stats import wald

def P(x):
    return (1.4/(math.sqrt(2*math.pi))*(math.e**((-1/2)*(x+5)**2) + math.e**((-1/2)*(x-5)**2)))
p = lambda x: 10*wald.pdf(x,loc=3, scale=1)
a = distributions_by_mixtures_Algorithm_3_gamma(500, 1, 1, p)
