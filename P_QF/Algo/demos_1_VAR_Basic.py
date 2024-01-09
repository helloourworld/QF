# -*- coding: utf-8 -*-
"""
@author: lyu
https://www.statsmodels.org/devel/vector_ar.html#
https://towardsdatascience.com/a-quick-introduction-on-granger-causality-testing-for-time-series-analysis-7113dc9420d2

"""
# some example data
from statsmodels.tsa.base.datetools import dates_from_str
import numpy as np

import pandas

import statsmodels.api as sm

from statsmodels.tsa.api import VAR

mdata = sm.datasets.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)

quarterly = dates["year"] + "Q" + dates["quarter"]


quarterly = dates_from_str(quarterly)

mdata = mdata[['realgdp', 'realcons', 'realinv']]

mdata.index = pandas.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

# make a VAR model
model = VAR(data)

results = model.fit(2)

results.summary()


results.plot()

results.plot_acorr()

model.select_order(15)

results = model.fit(maxlags=15, ic='aic')


lag_order = results.k_ar

results.forecast(data.values[-lag_order:], 5)

results.plot_forecast(10)


irf = results.irf(10)

irf.plot(orth=False)

irf.plot(impulse='realgdp')

# cumulative effects
irf.plot_cum_effects(orth=False)

# Forecast Error Variance Decomposition (FEVD)
fevd = results.fevd(5)
fevd.summary()

results.fevd(20).plot()


# Granger causality
from demos_1_VAR_GrangerCausality import grangers_causation_matrix

grangers_causation_matrix(data, variables = data.columns)


results.test_causality('realgdp', ['realinv'], kind='f').summary()
"""
Granger causality F-test. H_0: ['realinv', 'realcons'] 
do not Granger-cause realgdp. Conclusion: reject H_0 at 5% significance level.

“The variables representing real investment and real consumption do have
 a statistically significant Granger-causal effect on the real gross domestic
   product, according to the Granger causality F-test.”
"""
results.test_causality('realcons', ['realgdp', 'realinv'], kind='f').summary()
"""
the Granger causality F-test did not provide sufficient evidence to conclude
 that ‘realgdp’ and ‘realinv’ have a predictive effect on ‘realcons’ at the 
 5% significance level. It’s important to note that failing to reject the
   null hypothesis does not prove that there is no causality; 
   it simply means that the evidence is not strong enough to make that claim
     based on the test and the data used.
"""


results.test_normality().summary()
