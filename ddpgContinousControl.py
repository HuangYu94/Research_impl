# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:27:58 2018

@author: Yu Huang
"""
import gym
import tensorflow as tf
import models
import utils


env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
learning_rate = 0.001
discount_rate = 0.99
global_train_step = 20000
target_update_tau = 0.01
replay_memory_size = 1000
batch_size = 32

hidden_units_num = [16,16,16]
NonLinear = ['ReLU','ReLU','ReLU']

critic = models.NNagentFC('critic',state_dim+action_dim,1,hidden_units_num,NonLinear)
actor = models.NNagentFC('actor',state_dim,action_dim,hidden_units_num,NonLinear)

replayMem = models.ReplayMemory(state_dim,replay_memory_size)
random_process_param={'theta':0.15,'mu':0.0,'sigma':0.2}
noise_param = {'init_scale':0.1,'decay_rate':0.99}
policy = models.Policy_cont(random_process_param, noise_param)

reward_ph = tf.placeholder(dtype=tf.float64,shape=[None,1],name='reward_ph')
state_ph = tf.placeholder(dtype=tf.float64,shape=[None,state_dim],name='state_ph')
action_ph = tf.placeholder(dype=tf.float64,shape=[None,action_dim],name='action_ph')
next_state_ph = tf.placeholder(dtype=tf.float64,shape=[None,state_dim],name='next_state_ph')
terminal_ph = tf.placeholder(dtype=tf.float64,shape=[None,1],name='terminal_ph')

target_ph = tf.placeholder(dtype=tf.float64,shape=[None,1],name='target_ph')


