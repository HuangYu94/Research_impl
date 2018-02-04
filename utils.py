# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 21:15:19 2018

@author: Yu Huang
"""

import numpy as np

def npNonlinear(pre_act, Nonlinear):
    '''
    numpy array processing function to add nonlinearity
    pre_act must be in shape [batch_size, data_dim]
    Nonlinear options: sigmoid, ReLU
    return shape is the same as pre_act
    '''
    if Nonlinear == 'sigmoid':
        ret_out = 1./(1+np.exp(-pre_act))
        return ret_out
    elif Nonlinear == 'ReLU':
        ret_out = (pre_act > 0.)*pre_act
        return np.asarray(ret_out)
    else:
        raise ValueError('Unexpected Nonlinear layer specifier!')

def procActionBatch(action_arr):
    '''
    function to preprocess action array to facilitate applying tf.gather_nd()
    '''
    ret_lst = [[i,j] for i,j in enumerate(action_arr)]
    ret_lst = np.asarray(ret_lst,dtype=np.int32)
    return ret_lst




