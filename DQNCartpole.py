# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:02 2018

@author: Yu Huang
"""
import numpy as np
import random
import tensorflow as tf
import gym

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
        return np.asarray(ret_out,dtype=np.float32)
    else:
        raise ValueError('Unexpected Nonlinear layer specifier!')


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
        for idx, units_num in enumerate(hidden_units_num):
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
        self.output = pre_act
        
    def getTarget(self, sess):
        '''
        designed to evaluate varibles in dqn and store them as numpy array
        call this function will update target network
        This is in-place function to update target!
        sess: tf session to convert variables to numpy array
        '''
        self.target_params = {}
        '''
        dictionary naming rule: hidden_layer_[num]_W or hidden_layer_[num]_b;
        output_final
        '''
        for idx, units_num in enumerate(self.hidden_units_num):
            with tf.variable_scope('hidden_layer_'+str(idx+1),reuse=True):
                Weights = tf.get_variable('W').eval(sess)
                self.target_params['hidden_layer_'+str(idx+1)+'_W'] = Weights
                Bias = tf.get_variable('b').eval(sess)
                self.target_params['hidden_layer_'+str(idx+1)+'_b'] = Bias
        
        with tf.variable_scope('output_final',reuse=True):
            Weights = tf.get_variable('W').eval(sess)
            self.target_params['output_final_W'] = Weights
            Bias = tf.get_variable('b').eval(sess)
            self.target_params['output_final_b'] = Bias
            
    def evalTarget(self, nextStateBatch):
        '''
        self.getTarget() must be called first to call this function
        evaluate the value of next state
        INPUT: nextStateBatch must be in shape [batch_size, dim_in]
        RETURN: nextStateValue in shape [batch_size]
        '''
        for idx, nonlinear_type in enumerate(self.NonLinear):
            Weights_name = 'hidden_layer_'+str(idx+1)+'_W'
            Bias_name = 'hidden_layer_'+str(idx+1)+'_b'
            Weights = self.target_params[Weights_name]
            Bias = self.target_params[Bias_name]
            if idx ==0:
                pre_act = np.dot(nextStateBatch, Weights) + Bias
            else:
                pre_act = np.dot(out, Weights) + Bias
            
            out = npNonlinear(pre_act, nonlinear_type)
        
        target_score = np.dot(out,self.target_params['output_final_W']) 
        + self.target_params['output_final_b']
        target_score = np.max(target_score, axis=1)
        return target_score
    
    
    
class ReplayMemory:
    def __init__(self, state_dim, mem_length):
        self.state_dim = state_dim
        self.mem_length = mem_length
        self.current_size = 0
        
        
    def addSample(self, state, reward, next_state, terminal):
        if self.current_size == 0:
            self.state_pool = state
            self.reward_pool = reward
            self.next_state_pool = next_state
            self.terminal_pool = terminal
            self.current_size += 1
            
        elif self.current_size < self.mem_length:
            self.state_pool = np.concatenate((self.state_pool, state), axis=0)
            self.reward_pool = np.concatenate((self.reward_pool, reward), axis=0)
            self.next_state_pool = np.concatenate((self.next_state_pool, next_state), axis=0)
            self.terminal_pool = np.concatenate((self.terminal_pool, terminal), axis=0)
            self.current_size += 1
            
        else:
            self.state_pool = np.concatenate((self.state_pool[1:],state),axis=0)
            self.reward_pool = np.concatenate((self.reward_pool[1:],reward),axis=0)
            self.next_state_pool = np.concatenate((self.next_state_pool[1:], next_state), axis=0)
            self.terminal_pool = np.concatenate((self.terminal_pool[1:],terminal), axis=0)
            
    def getSample(self, sample_size):
        idx_sampled = random.sample([i for i in range(0,self.current_size)], sample_size)
        sample = {}
        sample['state'] = self.state_pool[idx_sampled]
        sample['reward'] = self.reward_pool[idx_sampled]
        sample['next_state'] = self.next_state_pool[idx_sampled]
        sample['terminal'] = self.terminal_pool[idx_sampled]

        return sample            
        

class Policy:
    '''
    this one only work for discrete action space!!!
    '''
    def __init__(self, start_epsilon, end_epsilon, decay_step):
        '''
        Linear decay epsilon greedy policy
        '''
        self.start_epsilon = start_epsilon
        self.current_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_rate = (start_epsilon - end_epsilon)/decay_step
    
    def pickAction(self, Qvalue):
        
        
        
# main function begins here


