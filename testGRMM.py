# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:16:19 2018

@author: Yu Huang
"""
from collections import namedtuple
import GRMM as GR

if __name__ == "__main__":
    
    GRMMParam = namedtuple('GRMMParam', 'Param1D')
    Param1D = namedtuple('Param1D', 'BaseThres BaseAvg_y BaseVar_y InitLocalModelParam EMIterNum AddThres PruneThres RspnsThres')
    LocalModelParam = namedtuple('LocalModelParam', 'EN FuncParam Mean_x Cov_x Avg_y Var_y AdaptRate ForgetRate')
    FuncParam = namedtuple('FuncParam', 'Mean_w Cov_w')

    TempFuncParam = FuncParam(Mean_w = [1.0,2.0], Cov_w = [[1.0,0.5],
                                                           [0.5,2.0]])
    TempLocalModelParam = LocalModelParam(EN = 1.0, 
                                          FuncParam = TempFuncParam,
                                          Mean_x = [1.0,2.0],
                                          Cov_x = [[1.0,0.5],
                                                   [0.5,2.0]],
                                          Avg_y = 3.0,
                                          Var_y = 0.1,
                                          AdaptRate = 0.01,
                                          ForgetRate = 0.001)
    TempParam1D = Param1D(BaseThres = 1e-4,
                          BaseAvg_y = 1.0,
                          BaseVar_y = 2.0,
                          InitLocalModelParam = TempLocalModelParam,
                          EMIterNum = 1,
                          AddThres = 0.75,
                          PruneThres = 0.01,
                          RspnsThres = 0.01)
    param = GRMMParam(Param1D = [TempParam1D, TempParam1D])

    a = GR.GRMM(param)
    a.learn([1,2],[3,4])
    mean, var, dmean, dvar = a.predict([1,2])
    print(mean, var, dmean, dvar)

