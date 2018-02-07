# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 21:05:54 2018

@author: Yu Huang
"""
import numpy as np
import utils
import random
import tensorflow as tf


class NNagentFC:
    def __init__(self,name_scope,dim_in=4,dim_out=2,hidden_units_num=[10,10,10],
                 NonLinear=['sigmoid','sigmoid','sigmoid'],drop_out=False):
        '''
        Basic model to solve Deep RL problem.
        Make sure you correctly entered the name_scope
        Using deep Q network to solve openAI gym control environment
        the default testing environment is cartpole control problem, so
        default input dimension is 4, output dimension is 2.
        Options for NonLinearity are {'sigmoid', 'ReLU'}
        '''
        if not len(hidden_units_num) == len(NonLinear):
            raise ValueError('layer number is not equal to Nonlinear layer number')
        
        self.update_step = 0
        self.is_continuous = False
        self.name_scope = name_scope
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_units_num = hidden_units_num
        self.NonLinear = NonLinear
        self.layerNum = len(hidden_units_num)
        self.input_ph = tf.placeholder(dtype=tf.float64,shape=[None,dim_in],name=name_scope+'state_input')
        self.drop_out_rate = tf.placeholder(dtype=tf.float64,shape=())
        self.param = []
        self.l2_loss = 0
        for idx, units_num in enumerate(hidden_units_num):
            with tf.variable_scope(self.name_scope+'_hidden_layer_'+str(idx+1)):
                if idx ==0:
                    Weights_init = tf.constant(np.random.uniform(-1,1,
                            size=[self.dim_in,self.hidden_units_num[idx]]))
                    Weights = tf.get_variable('W',initializer=Weights_init)
                    Bias_init = tf.constant(0.,shape=[self.hidden_units_num[idx]],dtype=tf.float64)
                    Bias = tf.get_variable('b',initializer=Bias_init)
                    pre_act = tf.nn.bias_add(tf.matmul(self.input_ph,Weights), Bias)
                else:
                    Weights_init = tf.constant(np.random.uniform(-1,1,
                            size=[self.hidden_units_num[idx-1],self.hidden_units_num[idx]]))
                    Weights = tf.get_variable('W',initializer=Weights_init)
                    Bias_init = tf.constant(0.,shape=[self.hidden_units_num[idx]],dtype=tf.float64)
                    Bias = tf.get_variable('b',initializer=Bias_init)
                    pre_act = tf.nn.bias_add(tf.matmul(out,Weights), Bias)
                    
                if self.NonLinear[idx] == 'sigmoid':
                    out = tf.nn.sigmoid(pre_act,name='out')
                elif self.NonLinear[idx] == 'ReLU':
                    out = tf.nn.relu(pre_act,name='out')
                    
                if drop_out:
                    out = tf.nn.dropout(out,keep_prob=self.drop_out_rate)
                self.param.append(Weights)
                self.l2_loss += tf.nn.l2_loss(Weights)
                self.param.append(Bias)
                    
        with tf.variable_scope(self.name_scope+'_output_final'):
            Weights_init = tf.constant(np.random.uniform(-1,1,
                    size=[self.hidden_units_num[self.layerNum-1],self.dim_out]))
            Weights = tf.get_variable('W',initializer=Weights_init)
            Bias_init = tf.constant(0.,shape=[self.dim_out],dtype=tf.float64)
            Bias = tf.get_variable('b',initializer=Bias_init)
            pre_act = tf.nn.bias_add(tf.matmul(out,Weights), Bias)
            self.param.append(Weights)
            self.l2_loss += tf.nn.l2_loss(Weights)
            self.param.append(Bias)
        self.output = pre_act
    
    def evalCurrentState(self, state, sess):
        '''
        evaluate current state and return score to each action
        '''
        state = state.reshape((1,self.dim_in))
        action_score = sess.run(self.output,feed_dict={self.input_ph:state,
                                                       self.drop_out_rate:1.0})
        return action_score
    
    def getTarget(self, sess):
        '''
        designed to evaluate varibles in dqn and store them as numpy array
        call this function will update target network
        This is in-place function to update target!
        sess: tf session to convert variables to numpy array
        This one will be used if we apply target fixing
        '''
        self.target_params = {}
        '''
        dictionary naming rule: hidden_layer_[num]_W or hidden_layer_[num]_b;
        output_final
        '''
        for idx, units_num in enumerate(self.hidden_units_num):
            with tf.variable_scope(self.name_scope+'_hidden_layer_'+str(idx+1),reuse=True):
                Weights = tf.get_variable('W',dtype=tf.float64).eval(sess)
                self.target_params['hidden_layer_'+str(idx+1)+'_W'] = Weights
                Bias = tf.get_variable('b',dtype=tf.float64).eval(sess)
                self.target_params['hidden_layer_'+str(idx+1)+'_b'] = Bias
        
        with tf.variable_scope(self.name_scope+'_output_final',reuse=True):
            Weights = tf.get_variable('W',dtype=tf.float64).eval(sess)
            self.target_params['output_final_W'] = Weights
            Bias = tf.get_variable('b',dtype=tf.float64).eval(sess)
            self.target_params['output_final_b'] = Bias
            
    def updateTarget(self,sess,tau):
        '''
        This one is designed to apply slowly shifting target
        '''
        if self.update_step ==0:
            self.getTarget(sess)
            self.update_step += 1
        else:
            for idx, units_num in enumerate(self.hidden_units_num):
                with tf.variable_scope(self.name_scope+'_hidden_layer_'+str(idx+1),reuse=True):
                    Weights = tf.get_variable('W',dtype=tf.float64).eval(sess)
                    self.target_params['hidden_layer_'+str(idx+1)+'_W'] = \
                    tau*Weights + (1-tau)*self.target_params['hidden_layer_'+str(idx+1)+'_W']
                    Bias = tf.get_variable('b',dtype=tf.float64).eval(sess)
                    self.target_params['hidden_layer_'+str(idx+1)+'_b'] = \
                    tau*Bias + (1-tau)*self.target_params['hidden_layer_'+str(idx+1)+'_b']
            
            with tf.variable_scope(self.name_scope+'_output_final',reuse=True):
                Weights = tf.get_variable('W',dtype=tf.float64).eval(sess)
                self.target_params['output_final_W'] = \
                tau*Weights + (1-tau)*self.target_params['output_final_W']
                Bias = tf.get_variable('b',dtype=tf.float64).eval(sess)
                self.target_params['output_final_b'] = \
                tau*Bias + (1-tau)*self.target_params['output_final_b']
            self.update_step += 1
            
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
#            print(nextStateBatch)
            if idx ==0:
                pre_act = np.dot(nextStateBatch, Weights) + Bias
            else:
                pre_act = np.dot(out, Weights) + Bias
            
            out = utils.npNonlinear(pre_act, nonlinear_type)
        
        target_score = np.dot(out,self.target_params['output_final_W']) 
        + self.target_params['output_final_b']
#        print(target_score.shape)
#        time.sleep(1)
        if self.is_continuous:
            return target_score
        target_score = np.max(target_score, axis=1)
        return target_score
    
    
    
class ReplayMemory:
    def __init__(self, state_dim, mem_length):
        self.state_dim = state_dim
        self.mem_length = mem_length
        self.current_size = 0
        
        
    def addSample(self, state, action, reward, next_state, terminal):
        state = state.reshape((1,-1))
        next_state = next_state.reshape((1,-1))
        if self.current_size == 0:
            self.state_pool = state
            self.action_pool = action
            self.reward_pool = reward
            self.next_state_pool = next_state
            self.terminal_pool = terminal
            self.current_size += 1
            
        elif self.current_size < self.mem_length:
            self.state_pool = np.concatenate((self.state_pool, state), axis=0)
#            print(self.action_pool)
#            print(action)
            self.action_pool = np.concatenate((self.action_pool, action), axis=0)
            self.reward_pool = np.concatenate((self.reward_pool, reward), axis=0)
            self.next_state_pool = np.concatenate((self.next_state_pool, next_state), axis=0)
            self.terminal_pool = np.concatenate((self.terminal_pool, terminal), axis=0)
            self.current_size += 1
            
        else:
            self.state_pool = np.concatenate((self.state_pool[1:],state),axis=0)
            self.action_pool = np.concatenate((self.action_pool[1:],action),axis=0)
            self.reward_pool = np.concatenate((self.reward_pool[1:],reward),axis=0)
            self.next_state_pool = np.concatenate((self.next_state_pool[1:], next_state), axis=0)
            self.terminal_pool = np.concatenate((self.terminal_pool[1:],terminal), axis=0)
            
    def getSample(self, sample_size):
        idx_sampled = random.sample([i for i in range(0,self.current_size)], sample_size)
        sample = {}
        sample['state'] = self.state_pool[idx_sampled]
        sample['action'] = self.action_pool[idx_sampled]
        sample['reward'] = self.reward_pool[idx_sampled]
        sample['next_state'] = self.next_state_pool[idx_sampled]
        sample['terminal'] = self.terminal_pool[idx_sampled]

        return sample            
        

class Policy_disc:
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
        '''
        Qvalue must be in shape [None, action_space_n]
        '''
        Qvalue = np.asarray(Qvalue)
        if self.current_epsilon <= self.end_epsilon:
            self.current_epsilon = self.end_epsilon
        else:
            self.current_epsilon -= self.decay_rate
        
        random_token = np.random.uniform()
        ret_action = 0
        if random_token > self.current_epsilon:
            opt_action = np.argmax(Qvalue,axis=1)
            ret_action = opt_action[0]
        else:
            rand_action = np.random.randint(low=0,high=Qvalue.shape[1])
            ret_action = rand_action
        
        return ret_action
        
    def reset(self):
        self.current_epsilon = self.start_epsilon
        
        
class Policy_cont:
    '''
    policy for continous action space I employ additive noise from Ornstein-Uhlenbeck process
    this one only work for continous action space!!!
    '''
    def __init__(self, random_process_param, noise_param):
        '''
        random_process_param should have {theta, mu, sigma} to construct 
        Ornstein-Uhlenbeck process
        noise_param should have {init_scale, decay_rate, action_space_range} to
        compute noise in each step note that init_scale shuold be in interval (0,1)
        note that action_space_range is [action_dim,1] vector
        '''
        self.random_process_param = random_process_param
        self.noise_param = noise_param
        self.scale = self.noise_param['init_scale']
        self.noise = self.noise_param['action_space_range']*self.scale
        self.process_noise = np.zeros(shape=self.noise.shape)
        
    def pickAction(self,opt_action):
        '''
        opt_action: optimal action indicated by actor network
        return: optmal action with additive noise
        '''
        self.process_noise = self.random_process_param['theta']*(self.random_process_param['mu'] - 
                                           self.process_noise) + self.random_process_param['sigma']*np.random.randn(self.process_noise.shape[0],
                                                             self.process_noise.shape[1])
        self.scale = self.scale*self.noise_param['decay_rate']
        return opt_action+self.process_noise*self.scale
    
    def reset(self):
        self.process_noise = np.zeros(shape=self.process_noise.shape)
        self.scale = self.noise_param['init_scale']
        
    
    
        
    
    
    
    
    
