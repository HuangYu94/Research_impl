# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:48:14 2018

@author: Yu Huang
"""

import tensorflow as tf

def helloTensorflow(inp):
    sess = tf.Session()
    hello = tf.constant('Hello, tensorflow!')
    sess.run(tf.global_variables_initializer())
    str_ret = sess.run(hello)
#    print(str_ret)
    return str_ret, inp

# mymod.py
"""Python module demonstrates passing MATLAB types to Python functions"""
def search(words):
    """Return list of words containing 'son'"""
    newlist = [w for w in words if 'son' in w]
    return newlist

def theend(words):
    """Append 'The End' to list of words"""
    words.append('The End')
    return words