# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:02 2018

@author: Yu Huang
"""
import numpy as np
import random
import tensorflow as tf
import gym
import time

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
        self.input_ph = tf.placeholder(dtype=tf.float64,shape=[None,dim_in],name='state_input')
        for idx, units_num in enumerate(hidden_units_num):
            with tf.variable_scope('hidden_layer_'+str(idx+1)):
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
                    
        with tf.variable_scope('output_final'):
            Weights_init = tf.constant(np.random.uniform(-1,1,
                    size=[self.hidden_units_num[self.layerNum-1],self.dim_out]))
            Weights = tf.get_variable('W',initializer=Weights_init)
            Bias_init = tf.constant(0.,shape=[self.dim_out],dtype=tf.float64)
            Bias = tf.get_variable('b',initializer=Bias_init)
            pre_act = tf.nn.bias_add(tf.matmul(out,Weights), Bias)
        self.output = pre_act
    
    def evalCurrentState(self, state, sess):
        '''
        evaluate current state and return score to each action
        '''
        state = state.reshape((1,self.dim_in))
        action_score = sess.run(self.output,feed_dict={self.input_ph:state})
        return action_score
    
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
                Weights = tf.get_variable('W',dtype=tf.float64).eval(sess)
                self.target_params['hidden_layer_'+str(idx+1)+'_W'] = Weights
                Bias = tf.get_variable('b',dtype=tf.float64).eval(sess)
                self.target_params['hidden_layer_'+str(idx+1)+'_b'] = Bias
        
        with tf.variable_scope('output_final',reuse=True):
            Weights = tf.get_variable('W',dtype=tf.float64).eval(sess)
            self.target_params['output_final_W'] = Weights
            Bias = tf.get_variable('b',dtype=tf.float64).eval(sess)
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
#            print(nextStateBatch)
            if idx ==0:
                pre_act = np.dot(nextStateBatch, Weights) + Bias
            else:
                pre_act = np.dot(out, Weights) + Bias
            
            out = npNonlinear(pre_act, nonlinear_type)
        
        target_score = np.dot(out,self.target_params['output_final_W']) 
        + self.target_params['output_final_b']
#        print(target_score.shape)
#        time.sleep(1)
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
        '''
        Qvalue must be in shape [None, action_space_n]
        '''
        Qvalue = np.asarray(Qvalue)
        if self.current_epsilon <= self.end_epsilon:
            self.current_epsilon = self.end_epsilon
        else:
            self.current_epsilon -= self.decay_rate
        
        random_token = np.random.uniform()
        if random_token > self.current_epsilon:
            opt_action = np.argmax(Qvalue,axis=1)
            return opt_action[0]
        else:
            rand_action = np.random.randint(low=0,high=Qvalue.shape[1])
            return rand_action
    def reset(self):
        self.current_epsilon = self.start_epsilon
            
        
        
# main function begins here
# setting hyperparameters here
# set up learning environment and training agent
env = gym.make('Pendulum-v0')
learning_rate = 0.001
discount_rate = 0.99
global_train_step = 40000
target_update_step = 100
replay_memory_size = 10000
batch_size = 128
dim_in = env.observation_space.shape[0]
dim_out = env.action_space.n
hidden_units_num = [16,16,16]
NonLinear = ['ReLU','ReLU','ReLU']


# build computation graph
dqnCartPoleAgent = DQNagentFC(dim_in,dim_out,hidden_units_num,NonLinear)
policy = Policy(1,0.1,4000)
replayMemory = ReplayMemory(dim_in, replay_memory_size)
target_ph = tf.placeholder(dtype=tf.float64,shape=[None],name='target_score') # feed target_score
reward_ph = tf.placeholder(dtype=tf.float64,shape=[None],name='reward')
terminal_ph = tf.placeholder(dtype=tf.float64,shape=[None],name='terminal')
state_ph = tf.placeholder(dtype=tf.float64,shape=[None, dim_in],name='state')
action_ph = tf.placeholder(dtype=tf.int32,shape=[None, 2],name='action')

update_target = reward_ph + discount_rate*(1-terminal_ph)*target_ph
#print(tf.gather_nd(dqnCartPoleAgent.output,action_ph))
loss = tf.reduce_mean(tf.square(update_target - tf.gather_nd(dqnCartPoleAgent.output,action_ph)))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# make sure the initial sample pool at least has 1 batch
state = env.reset()
for i in range(0, batch_size):
    action = env.action_space.sample()
    next_state, reward, terminal, _ =env.step(action)
    action = np.asarray(action,dtype=np.int32).reshape((1,))
    reward = np.asarray(reward).reshape((1,))
    terminal = np.asarray(terminal,dtype=np.int32).reshape((1,))
    replayMemory.addSample(state,action,reward,next_state,terminal)
    if terminal:
        state = env.reset()
    else:
        state = next_state
state = env.reset()
reward_rec = 0
episode_counter = 0
with tf.Session() as sess:
    # training process
    sess.run(tf.global_variables_initializer())
    train_step = 0
    
    while train_step < global_train_step:
        # update target when the steps condition is satisfied
        if train_step % target_update_step == 0:
            dqnCartPoleAgent.getTarget(sess)
            print('updating target...')
            
            
        
        action_score = dqnCartPoleAgent.evalCurrentState(state,sess)
#        print(action_score)
        action = policy.pickAction(action_score)
#        print(action)
        next_state, reward, terminal,_ = env.step(action)
        env.render()
        action = np.asarray(action,dtype=np.int32).reshape((1,))
        reward = np.asarray(reward).reshape((1,))
        terminal = np.asarray(terminal,dtype=np.int32).reshape((1,))
        replayMemory.addSample(state,action,reward,next_state,terminal)
        if terminal:
            state = env.reset()
            episode_counter += 1
            print('reward obtained from No. %d episode: %d'%(episode_counter,reward_rec))
            print('current episode %.6f'%policy.current_epsilon)
            reward_rec =0
        else:
            state = next_state
            reward_rec += reward
        
        # apply 1 step training
        sample = replayMemory.getSample(batch_size)
        target_score_batch = dqnCartPoleAgent.evalTarget(sample['next_state'])
        action_batch = procActionBatch(sample['action'])
        feed_d = {target_ph:target_score_batch,reward_ph:sample['reward'],
                  terminal_ph:sample['terminal'],
                  action_ph:action_batch,dqnCartPoleAgent.input_ph:sample['state']}
        _, loss_show = sess.run([train_op, loss],feed_dict=feed_d)
        train_step +=1
        if train_step % 20 ==0:
            print('training step %.4f with loss %.4f'%(train_step,loss_show))
#            time.sleep(0.5)
    
env.close()




