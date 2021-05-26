import mathfrom scipy.stats import uniformfrom scipy.stats import expon#from scipy.stats import gammaimport numpy as npimport matplotlib.pyplot as plt#from mean_var_as_paramters import beta_mean_var as beta#from scipy.stats import loggammafrom scipy.stats import alpha as alpha_distributionfrom scipy.stats import waldfrom scipy.stats import dweibullfrom scipy.stats import kstest"""Metrpolis_HastingsCreated on Wen Mar 01 10:56:41 2021@author: maximilianottosson"""from mean_var_as_paramters import gamma_mean_var as gammafrom mean_var_as_paramters import normal_mean_var as normdef Metrpolis_Hastings_gausian_proposal(inital,var,i_max,target):    '''    Parameters:   target : function             the target distribution.                i_max : integer         the size of the sample     inital : float or integer         starting mean    var : float or integer         var of the proposal     Returns    -------    Retunrns: a list of lits:        sample:             array or float            Description:    Returns a sample from p(x) using Metrpolis-Hastings with normal proposal    '''    current = inital    sample = []    i = 0        while i < i_max:        proposal = norm.rvs(current,var,1)        ratio = target(proposal)/target(current)                alpha = min([1,ratio])        u = uniform.rvs(0,1)        if u < alpha:            sample.append(float(proposal))            current = proposal        else:            sample.append(float(current))        i += 1                return(sample)def Metrpolis_Hastings_gamma_proposal(inital,var,i_max,target):    '''    Parameters:   target : function             the target distribution.                i_max : integer         the size of the sample     inital : float or integer         starting mean    var : float or integer         var of the proposal     Returns    -------    Retunrns: a list of lits:        sample:             array or float            Description:    Returns a sample from p(x) using Metrpolis-Hastings with gamma proposal    '''    current = inital    sample = []    i = 0        while i < i_max:                try:            proposal = gamma.rvs(current,var,1)        except ValueError:            if current < 1e-155: # As close to zero we can get with the scipy gamma distribution                current = 1e-155                proposal = gamma.rvs(current,var,1)            else:                print('proposal error')                break        try:            ratio = (target(proposal)*gamma.pdf(current,proposal,var))/(target(current)*gamma.pdf(proposal,current,var))        except ValueError:            print('Error in ratio')            break           alpha = min([1,ratio])        u = uniform.rvs(0,1)        if u < alpha:            sample.append(float(proposal))            current = proposal        else:            sample.append(float(current))        i += 1                return(sample)            def mean_convergence(list_of_means):    list_of_means = [sum(list_of_means[0:i])/(i+1) for i in range(len(list_of_means))]     return(list_of_means)#p = lambda x: expon.pdf(x, loc = 3 , scale = 5)#lines = np.linspace(3,5,200)#curve = [p(l) for l in lines]#sample = Metrpolis_Hastings_gamma_proposal(3,1,100000,p)#plt.hist(sample, density = True ,bins = 100) #plt.plot(lines,curve)#plt.show()#mean_con = mean_convergence(sample)#i = [x for x in range(len(sample))]#plt.plot(i,mean_con) #mean = lambda x: 0#imeans = [mean(x) for x in range(len(sample))]#plt.plot(i,imeans ) #plt.show()#def Kolmogorov_Smirnov__random_sample_from_target(sample,n,target_cdf):#    i = 0#    p_values = []#    rejcted = 0#    while i < len(sample):#        if i > n:#             if kstest(sample[(i-n):i], target_cdf, alternative='two-sided',N=10, mode='auto')[1] < 0.05:#                rejcted += 1#        i += 1#    print(rejcted/i)#    return(p_values)#b = Kolmogorov_Smirnov__random_sample_from_target(sample,10,p_1_cdf)#mean_con = mean_convergence(b)#i = [x for x in range(len(mean_con))]#plt.plot(i,mean_con) #mean = lambda x: 0.05#imeans = [mean(x) for x in range(len(mean_con))]#plt.plot(i,imeans ) 