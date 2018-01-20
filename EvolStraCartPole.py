# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 19:32:25 2017

@author: Yu Huang

Use evolution strategy to make expert playing data
"""
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

noice_scaling = 0.01
batch_num = 0
num_batches = 20

env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
xDim, = env.observation_space.shape
xDim += 1

num_batches = 20
batch_num = 0
Max_demo_step = 300

train_X0 = np.zeros((xDim,num_batches))
train_X = np.zeros((xDim,Max_demo_step, num_batches))
train_R = np.zeros((num_batches, Max_demo_step))

rec_step = 50
reward_rec = np.zeros((1,rec_step))

max_reward=500
best_params = np.random.rand(4) * 2 - 1
best_reward = 0
for i_episode in range(rec_step):
    params = best_params + (np.random.rand(4) * 2 - 1) * noice_scaling
    ob = env.reset()
    total_reward = 0
    t = 0
    while True:
        t += 1
        env.render()
        value = np.matmul(params, ob)
        action = 0
        if value > 0:
            action = 1
        ob_next, reward, done, info = env.step(action)
        total_reward += reward
        if total_reward > max_reward:
            done = True
        if done:
            if total_reward > Max_demo_step:
                train_X[:,:,batch_num] = current_X
                train_X0[:,batch_num] = current_X0
                train_R[batch_num,:] = current_r
                batch_num += 1
            if total_reward >= best_reward:
                best_reward = total_reward
                best_params = params
                noice_scaling = max(0.1, noice_scaling/2)
            else:
                noice_scaling = min(2, noice_scaling * 2)
            break
        state_action = np.concatenate((ob.reshape(-1,1), np.asarray(action).reshape((-1,1))))
        if t == 1:
            current_X0 = state_action[:,0]
            current_r = np.asarray(reward).reshape((-1,1))
        elif t == 2:
            current_X = state_action
            current_r = np.concatenate((current_r, np.asarray(reward).reshape((-1,1))), axis=1)
        elif t <= Max_demo_step:
            current_X = np.concatenate((current_X, state_action),axis=1)
            current_r = np.concatenate((current_r, np.asarray(reward).reshape((-1,1))), axis=1)
        elif t == Max_demo_step+1:
            current_X = np.concatenate((current_X, state_action),axis=1)
#            current_r = np.concatenate((current_r, np.asarray(reward)), axis=1)
        ob = np.copy(ob_next)
    if batch_num > num_batches-1:
        break
        
    reward_rec[:,i_episode] = total_reward
    print i_episode, t

#np.savetxt('train_X0.txt', train_X0)
#np.savetxt('train_X.txt', train_X.reshape((xDim*Max_demo_step, num_batches)))
#np.savetxt('train_R.txt', train_R)

'''
following code for ploting
'''
ax1 = plt.figure(1).add_subplot(111)
step_ = [i for i in range(0,rec_step)]
ax1.plot(step_, reward_rec[0,:])
ax1.set_title('accumulative reward obtained each episode')
ax1.set_xlabel('episode number')
ax1.set_ylabel('reward sum')







