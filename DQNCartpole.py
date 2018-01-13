# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:02 2018

@author: Yu Huang
"""
import tensorflow as tf
import gym

class DQNagentFC:
    def __init__(self,dim_in=4,dim_out=2,hidden_units_num=[10,10,10],
                 NonLinear=['sigmoid','sigmoid','sigmoid']):
        '''
        Using deep Q network to solve openAI gym control environment
        the default testing environment is cartpole control problem, so
        default input dimension is 4, output dimension is 2.
        Options for NonLinearity are {'sigmoid', 'ReLU'}
        '''
        if not len(hidden_units_num) == len(NonLinear):
            raise ValueError('layer number is not equal to Nonlinear layer number')
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_units_num = hidden_units_num
        self.NonLinear = NonLinear
        self.layerNum = len(hidden_units_num)
        self.input_ph = tf.placeholder(dtype=tf.float32,[None,dim_in])
        for idx, units_num in hidden_units_num:
            with tf.variable_scope('hidden_layer_'+str(idx+1)):
                if idx ==0:
                    Weights = tf.Variable(tf.truncated_normal(
                            shape=[self.dim_in,self.hidden_units_num[idx]]),name='W')
                    Bias = tf.Variable(tf.truncated_normal(shape=[self.hidden_units_num[idx]],
                                                           name='b'))
                    pre_act = tf.nn.bias_add(tf.matmul(self.input_ph,Weights), Bias)
                else:
                    Weights = tf.Variable(tf.truncated_normal(
                            shape=[self.hidden_units_num[idx-1],self.hidden_units_num[idx]]),name='W')
                    Bias = tf.Variable('b',shape=[self.hidden_units_num[idx]])
                    pre_act = tf.nn.bias_add(tf.matmul(out,Weights), Bias)
                    
                if self.NonLinear[idx] == 'sigmoid':
                    out = tf.nn.sigmoid(pre_act,name='out')
                elif self.NonLinear[idx] == 'ReLU':
                    out = tf.nn.relu(pre_act,name='out')
                    
        with tf.variable_scope('output_final'):
            Weights = tf.Variable(tf.truncated_normal(
                    shape=[self.hidden_units_num[self.layerNum-1],self.dim_out]),name='W')
            Bias = tf.Variable(tf.truncated_normal(
                    shape=[self.hidden_units_num[self.layerNum-1]],name='b'))
            pre_act = tf.nn.bias_add(tf.matmul(out,Weights), Bias)
            if self.NonLinear[self.layerNum-1] == 'sigmoid':
                    out = tf.nn.sigmoid(pre_act,name='out')
            elif self.NonLinear[self.layerNum-1] == 'ReLU':
                    out = tf.nn.relu(pre_act,name='out')
        self.output = out
        
    
                    
        
