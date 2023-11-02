# -*- encode: utf-8 -*-

import numpy as np
# MaxDrawDown
def maxdrawdown(arr):

    i = np.argmax((np.maximum.accumulate(arr) - arr)/np.maximum.accumulate(arr)) # end of the period

    j = np.argmax(arr[:i]) # start of period

    return (1-arr[i]/arr[j])

return_list = [100,200,50,20,300,150,100,200]
mdd = MaxDrawdown(return_list)
print(mdd)

