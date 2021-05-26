#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:40:36 2021

@author: maximilianottosson
"""


import algorithm_1
import algorithm_3
import Metropolis_Hastings
from timeit import default_timer as timer
from scipy.stats import alpha as alpha_distribution
from scipy.stats import dweibull

p_1_pdf = lambda x: alpha_distribution.pdf(x, 2, loc=0, scale=1)
p_2_pdf = lambda x: dweibull.pdf(x, 2)
p_3_pdf = lambda x: 10*alpha_distribution.pdf(x, 2, loc=0, scale=1)
p_4_pdf = lambda x: 100*dweibull.pdf(x, 2)

def main():
    time = []

    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(1,2,10000,p_1_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_1.distributions_by_mixtures_Algorithm_1_gamma(10000,1,2,p_1_pdf)
    end = timer()
    time.append('DM ' +str(end - start))
    
    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(10,5,10000,p_1_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_1.distributions_by_mixtures_Algorithm_1_gamma(10000,10,5,p_1_pdf)
    end = timer()
    time.append('DM ' +str(end - start))
    
    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(0,1,10000,p_2_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_1.distributions_by_mixtures_Algorithm_1_gausian(10000,0,1,p_2_pdf)
    end = timer()
    time.append('DM ' +str(end - start))
    
    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(10,10,10000,p_2_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_1.distributions_by_mixtures_Algorithm_1_gausian(10000,10,10,p_2_pdf)
    end = timer()
    time.append('DM ' +str(end - start))
   
    '------prop con----------'
    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(1,2,10000,p_3_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_3.distributions_by_mixtures_Algorithm_3_gamma(10000,1,2,p_3_pdf)
    end = timer()
    time.append('DM ' +str(end - start))

    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gamma_proposal(10,5,10000,p_3_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_3.distributions_by_mixtures_Algorithm_3_gamma(10000,10,5,p_3_pdf)
    end = timer()
    time.append('DM ' +str(end - start))

    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(1,2,10000,p_4_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_3.distributions_by_mixtures_Algorithm_3_gausian(10000,1,2,p_4_pdf)
    end = timer()
    time.append('DM ' +str(end - start))
    
    start = timer()   
    Metropolis_Hastings.Metrpolis_Hastings_gausian_proposal(10,10,10000,p_4_pdf)
    end = timer()
    time.append('MH' +str(end - start))

    start = timer()   
    algorithm_3.distributions_by_mixtures_Algorithm_3_gausian(10000,10,10,p_4_pdf)
    end = timer()
    time.append('DM ' +str(end - start))
    
    print(time)
    
main()    