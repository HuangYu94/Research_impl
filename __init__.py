# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:54:00 2018

@author: Yu Huang

register customized new environment
"""

from gym.envs.registration import register
import numpy as np

register(
    id='Pendulum-v1',
    entry_point='pendulum_env:PendulumEnv')

