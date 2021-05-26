#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:09:15 2021

@author: maximilianottosson
"""

import matplotlib.pyplot as plt
from scipy.stats import expon
from numpy import mean
import algorithm_1
import algorithm_3
import Metropolis_Hastings
from scipy.stats import dweibull
import numpy as np
import save_sample_to_csv
np.random.seed(seed=1995)


'--Targets---'
p_1_pdf = lambda x: expon.pdf(x, loc = 3 , scale = 5)
p_2_pdf = lambda x: dweibull.pdf(x, 2)
p_3_pdf = lambda x: 10*expon.pdf(x, loc = 3 , scale = 5)
p_4_pdf = lambda x: 100*dweibull.pdf(x, 2)
'------------'

def sample_var(data):
    x_bar = mean(data)
    sum_ = 0
    for i in range(len(data)):
        sum_ += (data[i]-x_bar)**2
    return((sum_/(len(data)-1)))

def mean_convergence(sample):
    list_of_means = [sum(sample[0:i])/len(sample[0:i]) for i in range(1,len(sample))] 
    return(list_of_means)

def var_convergence(sample):
    list_of_var = [sample_var(sample[0:i]) for i in range(len(sample))] 
    return(list_of_var)

def mean_convergence_ploter(mean,sample_1,sample_2,direct_sample,name_sample_1,name_sample_2,title):
    mean_con_1 = mean_convergence(sample_1)  
    mean_con_2 = mean_convergence(sample_2)
    mean_con_3 = mean_convergence(direct_sample)
    i = [x for x in range(len(mean_con_1))]
    mean_plot =[mean for x in range(len(mean_con_1))]
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Mean convergence')
    plt.plot(i,mean_con_1) 
    plt.plot(i,mean_con_1,label= name_sample_1,color = '0')
    plt.plot(i,mean_con_2,label= name_sample_2,color = '0.4')
    plt.plot(i,mean_con_3,label= 'Direct sampler',color = '0.6')
    plt.plot(i,mean_plot,label= 'Mean=' +str(mean),color = '0.2')
    plt.legend()
    plt.show()

def var_convergence_ploter(var,sample_1,sample_2,direct_sample,name_sample_1,name_sample_2,title):
    var_con_1 = var_convergence(sample_1)  
    var_con_2 = var_convergence(sample_2)
    var_con_3 = var_convergence(direct_sample)
    i = [x for x in range(len(var_con_1))]
    
    var_plot =[var for x in range(len(var_con_1))]
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Variance convergence')
    plt.plot(i,var_con_1,label= name_sample_1,color = '0')
    plt.plot(i,var_con_2,label= name_sample_2,color = '0.4')
    plt.plot(i,var_con_3,label= 'Direct sampler',color = '0.6')
    plt.plot(i,var_plot,label= 'Var=' +str(var),color = '0.2')
    plt.legend()
    plt.show()


dbm_1 = algorithm_1.distributions_by_mixtures_Algorithm_1_gamma(50000, 8, 2, p_1_pdf)[0]
Metrpolis_Hastings_1 = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(8,2,50001,p_1_pdf)
direct_sample_1 = expon.rvs(loc = 3 , scale = 5, size=50001)

mean_convergence_ploter(8,dbm_1,Metrpolis_Hastings_1,direct_sample_1,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: gamma(8,2) \n Target: exp(1, loc = 3, scale = 5)')
var_convergence_ploter(25,dbm_1,Metrpolis_Hastings_1,direct_sample_1,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: gamma(8,2) \n Target: exp(1, loc = 3, scale = 5)')

dbm_2 = algorithm_1.distributions_by_mixtures_Algorithm_1_gamma(60000, 16, 10, p_1_pdf)[0]
Metrpolis_Hastings_2 = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(16,10,60001,p_1_pdf)
direct_sample_2 = expon.rvs(loc = 3 , scale = 5, size=60001)

mean_convergence_ploter(8,dbm_2[:15000],Metrpolis_Hastings_2[:15000],direct_sample_2[:15000],'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: gamma(16,10) \n Target: exp(1, loc = 3, scale = 5)')
var_convergence_ploter(25,dbm_2[:15000],Metrpolis_Hastings_2[:15000],direct_sample_2[:15000],'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: gamma(16,10) \n Target: exp(1, loc = 3, scale = 5)')

dbm_3 = algorithm_1.distributions_by_mixtures_Algorithm_1_gausian(20000, 0, 1, p_2_pdf)[0]
Metrpolis_Hastings_3 = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(0,1,20001,p_2_pdf)
direct_sample_3 = dweibull.rvs(2, size=20001)

mean_convergence_ploter(0,dbm_3,Metrpolis_Hastings_3,direct_sample_3,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: norm(0,1) \n Target: Double-Weibull(2)')
var_convergence_ploter(1,dbm_3,Metrpolis_Hastings_3,direct_sample_3,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: norm(0,1) \n Target: Double-Weibull(2)')


dbm_4 = algorithm_1.distributions_by_mixtures_Algorithm_1_gausian(20000, 10, 10, p_2_pdf)[0]
Metrpolis_Hastings_4 = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(10,10,20001,p_2_pdf)

mean_convergence_ploter(0,dbm_4,Metrpolis_Hastings_4,direct_sample_3,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: norm(10,10) \n Target: Double-Weibull(2)')
var_convergence_ploter(1,dbm_4,Metrpolis_Hastings_4,direct_sample_3,'APDBM-algorithm-1','Metrpolis-Hastings','Proposal: norm(10,10) \n Target: Double-Weibull(2)')



'----------------------------'

dbm_5 = algorithm_3.distributions_by_mixtures_Algorithm_3_gamma(70000, 8, 2, p_3_pdf)[0]
Metrpolis_Hastings_5 = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(8,2,70000,p_3_pdf)
direct_sample_5 = expon.rvs(loc = 3 , scale = 5, size=70000)

mean_convergence_ploter(8,dbm_5,Metrpolis_Hastings_5,direct_sample_5,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: gamma(8,2) \n Target: exp(1, loc = 3, scale = 5)')
var_convergence_ploter(25,dbm_5,Metrpolis_Hastings_5,direct_sample_5,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: gamma(8,2) \n Target: exp(1, loc = 3, scale = 5)')

dbm_6 = algorithm_3.distributions_by_mixtures_Algorithm_3_gamma(70000, 16, 10, p_3_pdf)[0]
Metrpolis_Hastings_6 = Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(16,10,70000,p_3_pdf)

mean_convergence_ploter(8,dbm_6,Metrpolis_Hastings_6,direct_sample_5,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: gamma(16,10) \n Target: exp(1, loc = 3, scale = 5)')
var_convergence_ploter(25,dbm_6,Metrpolis_Hastings_6,direct_sample_5,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: gamma(16,10) \n Target: exp(1, loc = 3, scale = 5)')

dbm_7 = algorithm_3.distributions_by_mixtures_Algorithm_3_gausian(4001, 0, 1, p_4_pdf)[0]
Metrpolis_Hastings_7 = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(0,1,4000,p_4_pdf)
direct_sample_7 = dweibull.rvs(2, size=4000)

mean_convergence_ploter(0,dbm_7,Metrpolis_Hastings_7,direct_sample_7,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: norm(0,1) \n Target: Double-Weibull(2)')
var_convergence_ploter(1,dbm_7,Metrpolis_Hastings_7,direct_sample_7,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: norm(0,1) \n Target: Double-Weibull(2)')

dbm_8 = algorithm_3.distributions_by_mixtures_Algorithm_3_gausian(20001, 10, 10, p_4_pdf)[0]
Metrpolis_Hastings_8 = Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(10,10,20000,p_4_pdf)
direct_sample_8 = dweibull.rvs(2, size=20000)

mean_convergence_ploter(0,dbm_8,Metrpolis_Hastings_8,direct_sample_8,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: norm(10,10) \n Target: Double-Weibull(2)')
var_convergence_ploter(1,dbm_8,Metrpolis_Hastings_8,direct_sample_8,'APDBM-algorithm-2','Metrpolis-Hastings','Proposal: norm(10,10) \n Target: Double-Weibull(2)')

save_sample_to_csv.sample_to_csv(dbm_1,'dbm_1')
save_sample_to_csv.sample_to_csv(dbm_2,'dbm_2')
save_sample_to_csv.sample_to_csv(dbm_3,'dbm_3')
save_sample_to_csv.sample_to_csv(dbm_4,'dbm_4')
save_sample_to_csv.sample_to_csv(dbm_5,'dbm_5')
save_sample_to_csv.sample_to_csv(dbm_6,'dbm_6')
save_sample_to_csv.sample_to_csv(dbm_7,'dbm_7')
save_sample_to_csv.sample_to_csv(dbm_8,'dbm_8')

save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_1,'Metrpolis_Hastings_1')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_2,'Metrpolis_Hastings_2')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_3,'Metrpolis_Hastings_3')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_4,'Metrpolis_Hastings_4')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_5,'Metrpolis_Hastings_5')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_6,'Metrpolis_Hastings_6')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_7,'Metrpolis_Hastings_7')
save_sample_to_csv.sample_to_csv(Metrpolis_Hastings_8,'Metrpolis_Hastings_8')