# -*- coding: utf-8 -*-
"""
@author: lyu
https://www.statsmodels.org/devel/vector_ar.html#
https://towardsdatascience.com/a-quick-introduction-on-granger-causality-testing-for-time-series-analysis-7113dc9420d2

"""
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
maxlag=15
test = 'ssr_chi2test'
import numpy as np


def grangers_causation_matrix(data, variables, test='ssr_ftest', verbose=False):    
   
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.max(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

