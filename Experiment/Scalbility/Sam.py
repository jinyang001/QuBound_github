import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import linalg as la
import scipy.stats
import seaborn as sns
from statsmodels.stats.correlation_tools import cov_nearest, corr_nearest
import copy
import math
import pickle
from datetime import datetime
import warnings
import time


#### Supporting routines ####
def get_marginal_dicts(colnames_time0, colnames_time1, means_std_vals_time0, means_std_vals_time1):
    # we assume normal
    marginals_summary_time0 = {}
    k = 0
    for colname in colnames_time0:
        marginals_summary_time0[colname] = {}
        marginals_summary_time0[colname]['means_std_vals'] = means_std_vals_time0[k];
        k = k + 1
        marginals_summary_time0[colname]['type'] = 'normal'
        marginals_summary_time0[colname]['mu'] = marginals_summary_time0[colname]['means_std_vals'][0]
        marginals_summary_time0[colname]['sigma'] = marginals_summary_time0[colname]['means_std_vals'][1]

    marginals_summary_time1 = {}
    k = 0
    for colname in colnames_time1:
        marginals_summary_time1[colname] = {}
        marginals_summary_time1[colname]['means_std_vals'] = means_std_vals_time1[k];
        k = k + 1
        marginals_summary_time1[colname]['type'] = 'normal'
        marginals_summary_time1[colname]['mu'] = marginals_summary_time1[colname]['means_std_vals'][0]
        marginals_summary_time1[colname]['sigma'] = marginals_summary_time1[colname]['means_std_vals'][1]

    return marginals_summary_time0, marginals_summary_time1


def get_bhattacharya_coefficient_normalized(marginals_summary_time0, marginals_summary_time1, num_mcmc_samples):
    BC_normalized_marginal = []
    for (key0, value0), (key1, value1) in zip(marginals_summary_time0.items(), marginals_summary_time1.items()):
        copula_data_time0 = generate_copula_data({key0: value0}, num_mcmc_samples, correlationMatrix=np.array([1]))
        copula_data_time1 = generate_copula_data({key1: value1}, num_mcmc_samples, correlationMatrix=np.array([1]))

        # if any negative values are generated (because of using normal), replace with mean val
        copula_data_time0_sanitized = sanitize_copula_data(marginals_summary_time0, copula_data_time0)
        # copula_data_time1_sanitized = sanitize_copula_data(marginals_summary_time1, copula_data_time1)

        _, BC_normalized = get_hellinger_using_mcmc(g_params={key1: value1},
                                                    f_params={key0: value0},
                                                    f_samples=copula_data_time0_sanitized)

        BC_normalized_marginal.append(BC_normalized)

    return BC_normalized_marginal


def get_smax(BC_normalized_marginal, default='average', c=1):
    H_marginal = np.sqrt(1 - np.array(BC_normalized_marginal) ** d)
    H_marginal_average = np.mean(H_marginal)

    H_marginal_normalized = np.sqrt(1 - np.array(BC_normalized_marginal))
    hellinger_joint_normalized = np.sqrt(1 - np.prod(1 - H_marginal_normalized ** 2))

    if default == 'average':
        smax = 2 * c * H_marginal_average * np.sqrt(2 - H_marginal_average ** 2)  # uses the average hellinger
    else:
        smax = 2 * c * np.sqrt(
            1 - (1 - hellinger_joint_normalized ** 2) ** (2 * d))  # uses the dimension normalized hellinger

    return smax


def get_hellinger_normal(mu0, mu1, sigma0, sigma1, d=1):
    # mu0    = means_std_vals_time0[5][0];
    # mu1    = means_std_vals_time1[5][0];
    # sigma0 = means_std_vals_time0[5][1];
    # sigma1 = means_std_vals_time1[5][1];

    coeff_term = np.sqrt(2 * sigma0 * sigma1 / (sigma0 ** 2 + sigma1 ** 2))
    exp_term = np.exp(-0.25 * (mu0 - mu1) ** 2 / (sigma0 ** 2 + sigma1 ** 2))
    BC = (coeff_term * exp_term)
    return np.sqrt(1 - BC), BC ** (1 / d)


def sanitize_copula_data(marginals, copula_data):
    for colname in copula_data.keys():
        # print(copula_data[colname])
        copula_data[copula_data[colname] < 0] = marginals[colname]['mu']

    return copula_data


def generate_copula_data(marginals, num_mcmc_samples, correlationMatrix):
    # first_key, first_value = next(iter(marginals_summary_time0.items()))
    # marginals = {first_key:first_value }
    # num_mcmc_samples = num_mcmc_samples
    # correlationMatrix = np.array([1])

    # correlationMatrix_monthly = cov_nearest(correlationMatrix_monthly_sampleEstimate, threshold=1e-5)
    colnames = [item for item in marginals.keys()]
    mcmc_samples = pd.DataFrame(columns=colnames)
    mvnorm = scipy.stats.multivariate_normal(mean=np.zeros(len(marginals.keys())),
                                             cov=correlationMatrix)
    x = mvnorm.rvs(num_mcmc_samples)
    x_unif = scipy.stats.norm.cdf(x)  # this must stay the same always

    k = 0
    for colname in colnames:  # marginals.keys():
        # print(colname)
        if marginals[colname]['type'] == 'normal':
            m = scipy.stats.norm(loc=marginals[colname]['mu'],
                                 scale=marginals[colname]['sigma'])
            if len(colnames) > 1:
                mcmc_samples[colname] = m.ppf(x_unif[:, k])
            else:
                mcmc_samples[colname] = m.ppf(x_unif)

        k = k + 1

    return mcmc_samples


def get_hellinger_using_mcmc(g_params, f_params, f_samples):
    # THESE MUST BE EVALUATED AT SAME SAMPLES!!!
    # g_params=f_params
    log_copula_pdf_of_f_at_fsamples = get_log_copula(f_params, f_samples)
    log_copula_pdf_of_g_at_fsamples = get_log_copula(g_params, f_samples)

    copula_pdf_of_f_at_fsamples = np.exp(log_copula_pdf_of_f_at_fsamples)
    copula_pdf_of_g_at_fsamples = np.exp(log_copula_pdf_of_g_at_fsamples)

    ind = copula_pdf_of_f_at_fsamples > 1e-8
    ratio = np.sqrt(copula_pdf_of_g_at_fsamples[ind] / copula_pdf_of_f_at_fsamples[ind])

    # ratio = np.sqrt(copula_pdf_of_g_at_fsamples / copula_pdf_of_f_at_fsamples) #beware of precision issues from crazy small numbers

    BC = ratio.mean()
    # BC
    hellinger = np.sqrt(1 - BC)
    return hellinger, BC ** (1 / d)


def get_log_copula(g_params, f_samples, correlationMatrix=np.array([1])):
    # g_params=f_params; f_samples=sample1
    f_colnames = np.array([item for item in f_samples.keys()])
    g_colnames = np.array([item for item in g_params.keys()])

    F = {}
    logf = {}
    k = -1
    for colname in g_params.keys():
        k = k + 1
        fcolname = f_colnames[k]
        gcolname = g_colnames[k]
        data = f_samples[fcolname]

        if g_params[gcolname]['type'] == 'normal':
            F[gcolname] = scipy.stats.norm.cdf(data,
                                               g_params[gcolname]['mu'],
                                               g_params[gcolname]['sigma'])

            logf[gcolname] = scipy.stats.norm.logpdf(data,
                                                     g_params[gcolname]['mu'],
                                                     g_params[gcolname]['sigma'])

    logf_df = pd.DataFrame.from_dict(logf)
    F_df = pd.DataFrame.from_dict(F)

    rv = scipy.stats.multivariate_normal(mean=np.zeros(len(g_colnames)),
                                         cov=correlationMatrix)

    log_copula_pdf = rv.logpdf(F_df) + np.sum(logf_df, axis=1)

    return log_copula_pdf


#################

#### Initial raw data (BEWARE: SOME OF THIS IS HARD CODED) ####

start_time = time.time()
num_noise_factors = 2  # because there are two factors: T1 and T2
num_mcmc_samples = 16000  # this number is chosen so that the answer of monte carlo integration converges



colnames_time0 = [
    "T1_q0_3pm",
    "T1_q1_3pm",
    "T1_q4_3pm",

    "T1_q7_3pm",
    "T1_q10_3pm",
    "T1_q12_3pm",

    "T1_q13_3pm",
    "T1_q14_3pm",
    "T1_q16_3pm",

    "T1_q19_3pm",
    "T1_q22_3pm",
    "T1_q25_3pm",

    "T1_q24_3pm",
    "T1_q23_3pm",
    "T1_q21_3pm",
    ####
    "T2_q0_3pm",
    "T2_q1_3pm",
    "T2_q4_3pm",

    "T2_q7_3pm",
    "T2_q10_3pm",
    "T2_q12_3pm",

    "T2_q13_3pm",
    "T2_q14_3pm",
    "T2_q16_3pm",

    "T2_q19_3pm",
    "T2_q22_3pm",
    "T2_q25_3pm",

    "T2_q24_3pm",
    "T2_q23_3pm",
    "T2_q21_3pm",
]

means_std_vals_time0 = np.array([
    [115.68321186672864, 1.591650559280998],
    [133.03775258644987, 1.9265356400415727],
    [58.710257968347115, 0.676804937],

    [131.92302691923382, 1.9043455202021187],
    [88.94491586345774, 1.1232956083824572],
    [101.76395282962258, 1.3402071662705035],

    [96.80618426273146, 1.254582220559766],
    [100.90059745761329, 1.3251432094825106],
    [265.61181226691326, 5.130137178730692],

    [95.85611057929447, 1.2384193758931765],
    [88.74669250563292, 1.1200585442578082],
    [135.6599265, 1.9790909414658548],

    [155.54563121793194, 2.3934866364122804],
    [156.6002417324256, 2.416222433],
    [149.11316857247053, 2.256441114919914],

    ######

    [54.08831902365048, 2.164772875],
    [21.136600488130874, 0.7499240354706954],
    [56.14674548691419, 2.272206879008025],

    [14.05084011858371, 0.5623991020252903],
    [42.018312209627986, 1.5746610024840662],
    [16.912982892976917, 0.6281199884670133],

    [27.356284590238378, 0.9635406540584406],
    [20.87559824098902, 0.7417561274746862],
    [51.491057228558134, 2.0319403801613776],

    [36.07981497186464, 1.3116087182039533],
    [21.852187268191148, 0.7726881970868276],
    [12.076471834662332, 0.5302624045011051],

    [25.200214481579014, 0.8857162371446586],
    [31.534808103283662, 1.1240754654977587],
    [54.62124758417954, 2.192406427238331],

])
# data is ordered as ["T1_q0_time1", "T1_q1_time1", "T1_q2_time1", "T2_q0_time1", "T2_q1_time1", "T2_q2_time1",]

colnames_time1 = [
    "T1_q0_0pm",
    "T1_q1_0pm",
    "T1_q4_0pm",

    "T1_q7_0pm",
    "T1_q10_0pm",
    "T1_q12_0pm",

    "T1_q13_0pm",
    "T1_q14_0pm",
    "T1_q16_0pm",

    "T1_q19_0pm",
    "T1_q22_0pm",
    "T1_q25_0pm",

    "T1_q24_0pm",
    "T1_q23_0pm",
    "T1_q21_0pm",
    ####
    "T2_q0_0pm",
    "T2_q1_0pm",
    "T2_q4_0pm",

    "T2_q7_0pm",
    "T2_q10_0pm",
    "T2_q12_0pm",

    "T2_q13_0pm",
    "T2_q14_0pm",
    "T2_q16_0pm",

    "T2_q19_0pm",
    "T2_q22_0pm",
    "T2_q25_0pm",

    "T2_q24_0pm",
    "T2_q23_0pm",
    "T2_q21_0pm",
]

means_std_vals_time1 = np.array([
    [110.7194693767539, 1.5001646968355051],
    [144.81722590571832, 2.166494835],
    [57.42827750388674, 0.6602883434806818],

    [127.19639733993968, 1.8112745348004646],
    [67.51932485216606, 0.7962134768747511],
    [114.06950037206597, 1.5616922555156547],

    [109.91390499633441, 1.485504777],
    [111.41420836622272, 1.512850104359075],
    [121.04681479906473, 1.692696244004555],

    [108.07163776783513, 1.4521778927346396],
    [92.21779006104865, 1.177270576],
    [125.48303075400148, 1.7779486696642843],

    [125.44834509626263, 1.7772762991593294],
    [133.15050074362588, 1.9287851094127533],
    [160.95566408186176, 2.5109049424527203],

    ######
    [74.00517667185491, 3.279114438496866],
    [81.11045126074616, 3.714480831],
    [39.141890433296815, 1.4448245939985425],

    [76.08343784021567, 3.4045067104080995],
    [15.99391670656049, 0.6051471254982562],
    [42.76573111209611, 1.609114936],

    [35.600029016135906, 1.291221986911659],
    [21.529841341092087, 0.7623677458128073],
    [49.14965881082377, 1.914864984387817],

    [8.268611832947999, 0.5407249832557479],
    [27.810806521231648, 0.9804030198036239],
    [30.97943274942743, 1.1020504169084788],

    [42.47013859999808, 1.5954542168133838],
    [10.481326909209892, 0.5177414196598529],
    [67.49726560082864, 2.8972177138772532],
])

num_qubits = int(len(colnames_time0) / num_noise_factors)
d = int(num_qubits * num_noise_factors)  # dimensionality of the joint distribution (6 for our example)
# print(num_qubits)
# print(d)
#### Data structure for monte carlo based hellinger intergation ####
marginals_summary_time0, marginals_summary_time1 = get_marginal_dicts(colnames_time0,
                                                                      colnames_time1,
                                                                      means_std_vals_time0,
                                                                      means_std_vals_time1)

#### bhattacharya_coefficient (Core monte carlo intergration) ####
BC_normalized_marginal = get_bhattacharya_coefficient_normalized(marginals_summary_time0,
                                                                 marginals_summary_time1,
                                                                 num_mcmc_samples)

#### Bound on stability ####
smax = get_smax(BC_normalized_marginal, default='average', c=0.26)
# print(get_smax(BC_normalized_marginal, default = 'weighted'))
endtime = time.time()
exec_time = endtime - start_time
expectation_value_range = 2 * smax

# previous_day = 0.84699998  # 9/12 GHZ4
# previous_day=0.03800001 #9/12 GHZ3
# previous_day = 0.73799999  # 9/12 RB3
# previous_day = 0.20200001  # 9/12 HS4
# previous_day = 0.44199998  # 9/12 VQE4
# previous_day = -0.18400001  # 9/12 QAOA4

# print("upper bound:", previous_day + smax)
# print("lower bound:", previous_day - smax)
print("Range of Expectation Values:", expectation_value_range)
print("Total Execution Time:", exec_time, "seconds")
