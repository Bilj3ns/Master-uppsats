#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:32:01 2021

@author: maximilianottosson
"""

from scipy.stats import kstest
from scipy.stats import dweibull
from numpy import mean
import algorithm_1
import algorithm_3
import Metropolis_Hastings
from scipy.stats import alpha as alpha_distribution
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=1995)

p_1_pdf = lambda x: alpha_distribution.pdf(x, 2, loc=0, scale=1)
p_1_cdf = lambda x: alpha_distribution.cdf(x, 2, loc=0, scale=1)
p_1_rvs = lambda n: alpha_distribution.rvs( 2, loc=0, scale=1,size = n)

p_2_pdf = lambda x: dweibull.pdf(x, 2)
p_2_cdf = lambda x: dweibull.cdf(x, 2)
p_2_rvs = lambda n: dweibull.rvs(2,size = n)

p_3_pdf = lambda x: 10*alpha_distribution.pdf(x, 2, loc=0, scale=1)
p_3_cdf = lambda x: alpha_distribution.cdf(x, 2, loc=0, scale=1)
p_3_rvs = lambda n: alpha_distribution.rvs( 2, loc=0, scale=1,size = n)

p_4_pdf = lambda x: 100*dweibull.pdf(x, 2)
p_4_cdf = lambda x: dweibull.cdf(x, 2)
p_4_rvs = lambda n: dweibull.rvs(2,size = n)

def iteration_exact(alpha,length,sampler_target,cdf_target):
    i = length
    p_value = 0
    sample = list(sampler_target(length))
    while p_value < alpha:
        p_value = kstest(sample[i-length:i], cdf = cdf_target, alternative='two-sided', mode='auto')[1]
        sample.append(float(sampler_target(1)))
        i += 1
    return(float(i-1))

    
def Kolmogorov_Smirnov__random_sample_from_target(alpha,sample,length,target_cdf):
    i = length
    p_value = 0
    while p_value < alpha:
        p_value = kstest(sample[i-length:i], cdf = target_cdf, alternative='two-sided', mode='auto')[1]
        i += 1
    return(i-1)



def hist_ploter(list_,title): 
    plt.title(title)
    plt.xlabel('Iterations to pass the Kolmogorov-Smirnov test')
    plt.ylabel('Frequency')
    plt.hist(list_)
    plt.show()

def main_1():  
    ks_tests = [iteration_exact(0.1,10,p_1_rvs,p_1_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_1.distributions_by_mixtures_Algorithm_1_gamma(1000, 1, 2, p_1_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(1,2,1000,p_1_pdf)
        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_1_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_1_cdf))
        except ValueError:
            i - 1
            continue
        i += 1
    hist_ploter(dbm_test,'ADPBM- algorithm-1 with proposal: gamma(1,2) \n Target: alpha(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithms with proposal: gamma(1,2) \n Target: alpha(2) ')
    hist_ploter(ks_tests,'Direct sampler from alpha(2)')
    return('gamma(1,2): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))

def main_2():  
    ks_tests = [iteration_exact(0.1,10,p_1_rvs,p_1_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_1.distributions_by_mixtures_Algorithm_1_gamma(10, 10, 5, p_1_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(10,5,10,p_1_pdf)
        
        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_1_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_1_cdf))
        except ValueError:
            i - 1
            continue
        i += 1
    hist_ploter(dbm_test,'ADPBM-algorithm-1 with proposal: gamma(10,5) \n Target: alpha(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithm with proposal: gamma(10,5) \n Target: alpha(2) ')
    hist_ploter(ks_tests,'Direct sampler from alpha(2)')
    return('gamma(10,5): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))



def main_3():  
    ks_tests = [iteration_exact(0.1,10,p_2_rvs,p_2_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_1.distributions_by_mixtures_Algorithm_1_gausian(1000, 0, 1, p_2_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(0,1,1000,p_2_pdf)

        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_2_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_2_cdf))
        except ValueError:
            i - 1
            print('error')
            continue
        i += 1
    hist_ploter(dbm_test,'ADPBM-algorithm-1 with proposal: norm(0,1) \n Target: Double-Weibull(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithms with proposal: norm(0,1) \n Target: Double-Weibull(2) ')
    hist_ploter(ks_tests,'Direct sampler from Double-Weibull(2)')
    return('norm(0,1): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))


def main_4():  
    ks_tests = [iteration_exact(0.1,10,p_2_rvs,p_2_cdf) for x in range(0,1000)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_1.distributions_by_mixtures_Algorithm_1_gausian(10000, 10, 10, p_2_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(10,10,1000,p_2_pdf)

        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_2_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_2_cdf))
        except ValueError:
            i - 1
            print('error')
            continue
        i += 1
    hist_ploter(dbm_test,'ADPBM-algorithm-1 with proposal: norm(10,10) \n Target: Double-Weibull(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithms with proposal: norm(10,10) \n Target: Double-Weibull(2) ')
    hist_ploter(ks_tests,'Direct sampler from Double-Weibull(2)')
    return('norm(10,10): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))

def main_5():  
    ks_tests = [iteration_exact(0.1,10,p_1_rvs,p_1_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_3.distributions_by_mixtures_Algorithm_3_gamma(5000, 1, 2, p_3_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(1,2,5000,p_3_pdf)

        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_1_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_1_cdf))
        except ValueError:
            i - 1
            print('error')
            continue
        i += 1       
    hist_ploter(dbm_test,'ADPBM-algorithm-2 with proposal: gamma(1,2) \n Target: alpha(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithms with proposal: gamma(1,2) \n Target: alpha(2) ')
    hist_ploter(ks_tests,'Direct sampler from alpha(2)')
    return('procon_unknown,gamma(1,2): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))

def main_6():  
    ks_tests = [iteration_exact(0.1,10,p_1_rvs,p_1_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_3.distributions_by_mixtures_Algorithm_3_gamma(5000, 10, 5, p_3_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(10,5,5000,p_3_pdf)

        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_1_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_1_cdf))
        except ValueError:
            i - 1
            print('error')
            continue
        i += 1
        
    hist_ploter(dbm_test,'ADPBM-algorithm-2 with proposal: gamma(10,5) \n Target: alpha(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithm with proposal: gamma(10,5) \n Target: alpha(2) ')
    hist_ploter(ks_tests,'Direct sampler from alpha(2)')
    return('propcon_unknown_gamma(10,5): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))


def main_7():  
    ks_tests = [iteration_exact(0.1,10,p_1_rvs,p_1_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_3.distributions_by_mixtures_Algorithm_3_gausian(1000, 1, 2, p_4_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(1,2,1000,p_4_pdf)
        
        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_4_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_4_cdf))
        except ValueError:
            i - 1
            continue
        i += 1
    hist_ploter(dbm_test,'ADPBM-algorithm-2 with proposal: norm(0,1) \n Target: Double-Weibull(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithms with proposal: norm(0,1) \n Target: Double-Weibull(2) ')
    hist_ploter(ks_tests,'Direct sampler from Double-Weibull(2)')
    return('propcon_unknown_norm(0,1): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))

def main_8():  
    ks_tests = [iteration_exact(0.1,10,p_1_rvs,p_1_cdf) for x in range(0,100)]      
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = algorithm_3.distributions_by_mixtures_Algorithm_3_gausian(1000, 10, 10, p_4_pdf)[0]
        Metrpolis_Hastings = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(10,10,1000,p_4_pdf)
        
        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_4_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_4_cdf))
        except ValueError:
            i - 1
            continue
        i += 1
        
    hist_ploter(dbm_test,'ADPBM-algorithm-2 with proposal: norm(10,10) \n Target: Double-Weibull(2) ')
    hist_ploter(Metrpolis_Hastings_test ,'Metrpolis_Hastings- algorithms with proposal: norm(10,10) \n Target: Double-Weibull(2) ')
    hist_ploter(ks_tests,'Direct sampler from Double-Weibull(2)')
    return('propcon_unknown_norm(10,10): '+ str(mean(ks_tests)) +' ' + str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))


a = main_1()
#b = main_2()
#c = main_3()
#d = main_4()
#e = main_5()
#f = main_6()
#g = main_7() 
#i = main_8()
#print(a)
#print(b)
#print(c)
#print(d)
#print(e)
#print(f)
#print(g)
#print(i)