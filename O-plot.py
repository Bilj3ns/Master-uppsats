#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:18:35 2021

@author: maximilianottosson
"""

import matplotlib.pyplot as plt
from scipy.stats import alpha as alpha_distribution
from scipy.stats import dweibull
import Time_per_iteration_algorithm

'--Targets---'
p_1_pdf = lambda x: alpha_distribution.pdf(x, 2, loc=0, scale=1)
p_2_pdf = lambda x: dweibull.pdf(x, 2)
p_3_pdf = lambda x: 10*alpha_distribution.pdf(x, 2, loc=0, scale=1)
p_4_pdf = lambda x: 100*dweibull.pdf(x, 2)
'------------'



def time_ploter(sample_1,sample_2,name_sample_1,name_sample_2,title,name_of_sampler,ylim):
    i = [x for x in range(len(sample_1))]
    
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Time per iteration')
    plt.ylim(ylim[0], ylim[1])
    plt.plot(i,sample_1,label= name_sample_1,color = '0')
    plt.plot(i,sample_2,label= name_sample_2,color = '0.4')
    plt.legend()
    plt.show()




#dbm = Time_algorithm.distributions_by_mixtures_Algorithm_1_gamma(10001, 1, 2, p_1_pdf)
#Metrpolis_Hastings = Time_algorithm.Metrpolis_Hastings_gamma_proposal(1,2,10000,p_1_pdf)
 
#dbm = Time_algorithm.distributions_by_mixtures_Algorithm_1_gausian(10000, 10, 10, p_2_pdf)
#Metrpolis_Hastings = Time_algorithm.Metrpolis_Hastings_gausian_proposal(10000,10,10,p_2_pdf)


'----------------------------'

dbm = Time_per_iteration_algorithm.distributions_by_mixtures_Algorithm_3_gamma(10000, 10, 5, p_3_pdf)
Metrpolis_Hastings = Time_per_iteration_algorithm.Metrpolis_Hastings_gamma_proposal(10,5,10000,p_3_pdf)

#dbm = Time_algorithm.distributions_by_mixtures_Algorithm_3_gausian(10001, 10, 10, p_4_pdf)
#Metrpolis_Hastings = Time_algorithm.Metrpolis_Hastings_gausian_proposal(10000,10,100,p_4_pdf)

time_ploter(dbm[0],Metrpolis_Hastings[0],'APDBM-algorithm','Metrpolis-Hastings','Proposal: gamma(10,5) \n Target: alpha(2)','alpha(2)',[0,0.025])

