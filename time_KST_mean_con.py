#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:36:20 2021

@author: maximilianottosson
"""

from scipy.stats import kstest
from scipy.stats import dweibull
from numpy import mean
import total_time
from scipy.stats import alpha as alpha_distribution
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
np.random.seed(seed=1995)

'--Targets---'
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

p_5_pdf = lambda x: expon.pdf(x, loc = 3 , scale = 5)
p_6_pdf = lambda x: 10*expon.pdf(x, loc = 3 , scale = 5)
'--------------'

def mean_convergence(sample):
    list_of_means = [sum(sample[0:i])/len(sample[0:i]) for i in range(1,len(sample))] 
    return(list_of_means)

def mean_convergence_ploter(mean,sample_1,sample_2,name_sample_1,name_sample_2,title):
    '''
    Parameters:
    mean : float  or int 
        The mean of the target  
        
    sample_1 : iterable   
        A sample that you want to compre the other sample to

    sample_2 : iterable   
        A sample that you want to compre the other sample to

    name_sample_1 : String
        A name of sample_1 in the legend

    name_sample_2 : String
        A name of sample_2 in the legend
        
    title: String 
        title of the plot
 
    Returns
    -------
    Retun value: none
              
    Description:
    Plots time to converge to the mean
    '''
    mean_con_1 = mean_convergence(sample_1[1])  
    mean_con_2 = mean_convergence(sample_2[1])
    mean_plot =[mean for x in range(len(mean_con_2))]
    
    plt.title(title)
    plt.xlabel('Seconds')
    plt.ylabel('Mean convergence')
    plt.plot(sample_1[0][0:len(mean_con_1)],mean_con_1,label = name_sample_1,color = '0')
    plt.plot(sample_2[0][0:len(mean_con_2)],mean_con_2,label= name_sample_2,color = '0.4')
    plt.plot(sample_2[0][0:len(mean_con_2)],mean_plot,label= 'Mean=' +str(mean),color = '0.2')
    plt.legend()
    plt.show()

def Kolmogorov_Smirnov__random_sample_from_target(alpha,sample,length,target_cdf):
    '''
    Parameters:
    alpha : float  
        Significance Level of the test
        
    sample : iterable   
        The sample that you want to perfrom the iterable Kolmogorov Smirnov test on

    length : integer 
        The length of the sequence you want to do the Kolmogorov Smirnov test on

    target_cdf: function 
        the cdf of the target
 
    Returns
    -------
    Retun value: integer
        Retunrns the time it took to not reject H_0 of the Kolmogorov Smirnov test
        
    Description:
    Performs an iterative Kolmogorov Smirnov test on a sample and returns the time it took. 
    '''
    i = length
    p_value = 0
    while p_value < alpha:
        p_value = kstest(sample[1][i-length:i], cdf = target_cdf, alternative='two-sided', mode='auto')[1]
        i += 1
    return(sample[0][i-1])

def bar_ploter(list_,title):
    id_ = []
    frequency = []
    for i in range(min(list_),max(list_)):
        freq = (list_.count(i))/len(list_)
        if freq > 0:
            frequency.append(freq)
            id_.append(str(i))
    
    plt.title(title)
    plt.xlabel('Iterations to pass the Kolmogorov-Smirnov test')
    plt.ylabel('Frequency')
    y_pos = np.arange(len(id_))
    plt.bar(y_pos, frequency)
    plt.xticks(y_pos, id_)
    plt.show()

'The main functions 1-5 solves the problems described in: "When APDBM-algorithm is faster iteration by iteration" '

'Time-wise goodness of fit with the Kolmogorov Smirnov test page 41'    

def main_1():       
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = total_time.distributions_by_mixtures_Algorithm_1_gamma(500, 1, 2, p_1_pdf)
        Metrpolis_Hastings = total_time.Metrpolis_Hastings_gamma_proposal(1,2,500,p_1_pdf)
        
        dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_1_cdf))
        Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_1_cdf))
        i += 1
        print(i)
    print(str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))


def main_2():        
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = total_time.distributions_by_mixtures_Algorithm_1_gausian(1000, 0, 1, p_2_pdf)
        Metrpolis_Hastings = total_time.Metrpolis_Hastings_gausian_proposal(1000,0,1,p_2_pdf)

        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_2_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_2_cdf))
        except ValueError:
            i - 1
            print('error')
            continue
        i += 1
        print(i)
    print(str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))



def main_3():        
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = total_time.distributions_by_mixtures_Algorithm_3_gamma(5000, 1, 2, p_3_pdf)
        Metrpolis_Hastings = total_time.Metrpolis_Hastings_gamma_proposal(1,2,5000,p_3_pdf)

        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_1_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_1_cdf))
        except ValueError:
            i - 1
            print('error')
            continue
        i += 1
        print(i)
        
    print(str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))


def main_4():       
    dbm_test = []
    Metrpolis_Hastings_test = []
    i = 0
    while i < 100:
        dbm = total_time.distributions_by_mixtures_Algorithm_3_gausian(1000, 1, 2, p_4_pdf)
        Metrpolis_Hastings = total_time.Metrpolis_Hastings_gausian_proposal(1000,1,2,p_4_pdf)
        
        try:
            dbm_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,dbm,10,p_4_cdf))
            Metrpolis_Hastings_test.append(Kolmogorov_Smirnov__random_sample_from_target(0.1,Metrpolis_Hastings,10,p_4_cdf))
        except ValueError:
            i - 1
            continue
        i += 1
    print(str(mean(dbm_test)) +' ' +str(mean(Metrpolis_Hastings_test)))

'Time-wise convergence of mean page 41' 
 
def main_5():
    dbm = total_time.distributions_by_mixtures_Algorithm_1_gamma(2000, 8, 2, p_5_pdf)
    Metrpolis_Hastings = total_time.Metrpolis_Hastings_gamma_proposal(8,2,16600,p_5_pdf)

    mean_convergence_ploter(8,dbm,Metrpolis_Hastings,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: gamma(8,2) \n Target: exp(loc=3,scale=5)')       

    dbm = total_time.distributions_by_mixtures_Algorithm_1_gausian(2000, 0, 1, p_2_pdf)
    Metrpolis_Hastings = total_time.Metrpolis_Hastings_gausian_proposal(2000,0,1,p_2_pdf)

    mean_convergence_ploter(0,dbm,Metrpolis_Hastings,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: norm(0,1) \n Target: Double-Weibull(2)')

    dbm = total_time.distributions_by_mixtures_Algorithm_3_gamma(1600, 16, 10, p_6_pdf)
    Metrpolis_Hastings = total_time.Metrpolis_Hastings_gamma_proposal(16,10,14000,p_6_pdf)

    mean_convergence_ploter(8,dbm,Metrpolis_Hastings,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: gamma(16,10) \n Target: exp(1,loc=3,scale=5)')       



main_1()
main_2()
main_3()
main_4()
main_5()