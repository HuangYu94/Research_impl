# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:27:58 2018

@author: Yu Huang
"""
import numpy as np
import gym
import tensorflow as tf
import models
import utils


def createActorLoss(actor, critic, state_ph):
    '''
    function to create actor loss without change weights in critic
    return:
        actorLoss: tf node to be minimized
    '''
    loss = 0
    for idx, units_num in enumerate(critic.hidden_units_num):
        with tf.variable_scope(critic.name_scope+'_hidden_layer_'+str(idx+1),reuse=True):
            Weights = tf.get_variable('W',dtype=tf.float64)
            loss += tf.nn.l2_loss(Weights)
            Bias = tf.get_variable('b',dtype=tf.float64)
            if idx == 0:
                critic_input = tf.concat([state_ph, actor.output],axis=1)
                pre_act = tf.nn.bias_add(tf.matmul(critic_input, Weights),Bias)
            else:
                pre_act = tf.nn.bias_add(tf.matmul(out, Weights),Bias)
                
            if critic.NonLinear[idx]=='sigmoid':
                out = tf.nn.sigmoid(pre_act,name='out')
            elif critic.NonLinear[idx]=='ReLU':
                out = tf.nn.relu(pre_act,name='out')
    
    with tf.variable_scope(critic.name_scope+'_output_final',reuse=True):
        Weights = tf.get_variable('W',dtype=tf.float64)
        loss += tf.nn.l2_loss(Weights)
        Bias = tf.get_variable('b',dtype=tf.float64)
        pre_act = tf.nn.bias_add(tf.matmul(out,Weights),Bias)
        
    loss = -tf.reduce_mean(pre_act)
    
    return loss
    
                



env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
learning_rate = 0.001
l2_reg = 1e-3
discount_rate = 0.99
global_train_step = 20000
target_update_tau = 0.01
replay_memory_size = 10000
batch_size = 1024

hidden_units_num = [16,16,16]
NonLinear = ['ReLU','ReLU','ReLU']

critic = models.NNagentFC('critic',state_dim+action_dim,1,hidden_units_num,NonLinear,drop_out=True)
critic.is_continuous=True
actor = models.NNagentFC('actor',state_dim,action_dim,hidden_units_num,NonLinear,drop_out=True)
actor.is_continuous=True

action_space_range = (env.action_space.high-env.action_space.low).reshape((-1,1))
actor.output = env.action_space.low.reshape((-1,1)) + tf.nn.sigmoid(actor.output)*action_space_range # scale control input

replayMemory = models.ReplayMemory(state_dim,replay_memory_size)
random_process_param={'theta':0.15,'mu':0.0,'sigma':0.2}
noise_param = {'init_scale':0.1,'decay_rate':0.99,'action_space_range':action_space_range}
policy = models.Policy_cont(random_process_param, noise_param)

reward_ph = tf.placeholder(dtype=tf.float64,shape=[None],name='reward_ph')
state_ph = tf.placeholder(dtype=tf.float64,shape=[None,state_dim],name='state_ph')
action_ph = tf.placeholder(dtype=tf.float64,shape=[None,action_dim],name='action_ph')
next_state_ph = tf.placeholder(dtype=tf.float64,shape=[None,state_dim],name='next_state_ph')
terminal_ph = tf.placeholder(dtype=tf.float64,shape=[None,1],name='terminal_ph')

target_ph = tf.placeholder(dtype=tf.float64,shape=[None,1],name='target_ph')

loss_critic = tf.reduce_mean(tf.square(reward_ph+discount_rate*target_ph-critic.output))/2 + l2_reg*critic.l2_loss

loss_actor = createActorLoss(actor,critic,state_ph) + l2_reg*actor.l2_loss
gradient_on_actor_param = tf.gradients(loss_actor,actor.param)



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
critic_train_op = optimizer.minimize(loss_critic)
actor_train_op = optimizer.apply_gradients(zip(gradient_on_actor_param,actor.param))
train_step = 0
state = env.reset()
episode_counter = 0
reward_rec=0
for i in range(0, batch_size):
    action = env.action_space.sample()
    next_state, reward, terminal, _ =env.step(action)
    action = np.asarray(action,dtype=np.float64).reshape((1,-1))
    reward = np.asarray(reward).reshape((1,))
    terminal = np.asarray(terminal,dtype=np.int32).reshape((1,))
    replayMemory.addSample(state,action,reward,next_state,terminal)
    if terminal:
        state = env.reset()
    else:
        state = next_state

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    state = env.reset()
    while train_step < global_train_step:
        opt_action = actor.evalCurrentState(state,sess)
        action = policy.pickAction(opt_action)
        next_state, reward, terminal, _ = env.step(action)
        action = action.reshape((1,-1))
        reward = np.asarray(reward).reshape((1,))
        terminal = np.asarray(terminal,dtype=np.int32).reshape((1,))
        replayMemory.addSample(state,action,reward,next_state,terminal)
        if terminal:
            state = env.reset()
            episode_counter += 1
            print('reward obtained from No. %d episode: %d final noise scale %.4f'%(episode_counter,reward_rec,policy.scale))
            policy.reset()
#            print('current episode %.6f'%policy.current_epsilon)
            reward_rec =0
        else:
            state = next_state
            reward_rec += reward     
        
        #apply training operation
        sample=replayMemory.getSample(batch_size)
        actor.updateTarget(sess,target_update_tau)
        critic.updateTarget(sess,target_update_tau)
        opt_action_batch = actor.evalTarget(sample['next_state'])
        target_score = critic.evalTarget(np.concatenate((sample['next_state'],
                                                         opt_action_batch),axis=1))
        _, crit_loss_show = sess.run([critic_train_op,loss_critic],feed_dict={
                state_ph:sample['state'],action_ph:sample['action'],reward_ph:sample['reward'],
                next_state_ph:sample['next_state'],critic.input_ph:np.concatenate((sample['state'],
                                    sample['action']),axis=1),target_ph:target_score,
                                    critic.drop_out_rate:0.5})
#        actor_loss = createActorLoss(actor,critic,state_ph,sess)
        
#        actor_train_op = optimizer.minimize(actor_loss)
        _, actor_loss_show = sess.run([actor_train_op,loss_actor],feed_dict={
                state_ph:sample['state'],action_ph:sample['action'],reward_ph:sample['reward'],
                next_state_ph:sample['next_state'],actor.input_ph:sample['state'],
                actor.drop_out_rate:0.5})
        
        train_step += 1
        env.render()
        if train_step % 10 ==0:
            print('train step %d with actor loss %.4f and crtic loss %.4f'%
                  (train_step,crit_loss_show,actor_loss_show))
            
    
        


