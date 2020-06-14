# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:53:44 2020

@author: Wakasugi Kazuyuki

pyqubo == 0.4.0
openjij == 0.0.11
dwave-ocean-sdk == 2.2.0
cvxpy == 1.1.1

"""

import matplotlib.pyplot as plt
import numpy as np
from pyqubo import Array, Constraint
import openjij as oj
import re
import pandas as pd

#乱数シード
np.random.seed(0)

#荷物の候補の数
n = 5
#荷物の価値
values = np.random.randint(1, n, n)
#荷物の重さ
weights = np.random.randint(1, n, n)
#重量制限
weight_limit = n * 2
#マージン
margin = 2
#ペナルティ
alpha = 1

#荷物の入れる／入れない
x = Array.create('x', n, 'BINARY')
#スラック変数
s = Array.create('s', margin, 'BINARY')

#価値
total_value = sum(x_i * v_1 for x_i, v_1 in zip(x, values))
#重さ
total_weight = sum(x_i * w_1 for x_i, w_1 in zip(x, weights))
#ハミルトニアン
H = -1 * total_value + alpha * Constraint((total_weight - weight_limit + sum(s_i for s_i in s))**2, label='weight_limit')
#quboモデル
model = H.compile()
qubo, offset = model.to_qubo()

# OpenJijで解く
#--------------------------------------
sampler = oj.SASampler()
response = sampler.sample_qubo(qubo, num_reads=1000)
plt.hist(-(response.energies + offset))
## エネルギーが一番低い状態を取り出します。
dict_solution = response.first.sample
## デコードします。
decoded_solution, broken, energy = model.decode_solution(dict_solution, vartype="BINARY")
#--------------------------------------

# Cmosで解く
#--------------------------------------
cmos_fpga = oj.CMOSAnnealer(token="", machine_type="FPGA")
qubo_cmos = {}
for key in qubo.keys():
    
    def geti(x):
        s1, s2, _ = re.split("[][]", x)
        if s1 == 'x':
            i = int(s2)
        else:
            i = n + int(s2) 
        return i
            
    x, y = key
    i = geti(x)
    j = geti(y)
        
    qubo_cmos[(i, j+80)] = int(qubo[key])
    
#response = cmos_fpga.sample_qubo(Q=qubo_cmos)
response = cmos_fpga.sample_qubo(Q={(0, 81): 320, (1, 82): 320,})
#--------------------------------------

#Dwaveで解く
#--------------------------------------
# 接続情報をオプションとして渡す場合は以下のようにします。
endpoint = 'https://cloud.dwavesys.com/sapi'
token = ''
solver = 'DW_2000Q_6' #hybrid_v1
# DWaveSamplerを用います。
dw = DWaveSampler(endpoint=endpoint, token=token, solver=solver)
# キメラグラフに埋め込みを行います。
sampler = EmbeddingComposite(dw)
# QUBOの場合は以下を使用します。
response = sampler.sample_qubo(qubo, num_reads=3000)
#response = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token)).sample_qubo(qubo, num_reads=3000)

df_result = pd.DataFrame()
k = 0
for sample, energy, num_occurrences, chain_break_fraction in list(response.data()):
    #print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
    df_tmp = pd.DataFrame(dict(sample), index=[k])
    df_tmp['Energy'] = -(energy+offset)
    df_tmp['Occurrences'] = num_occurrences
    df_result = df_result.append(df_tmp)
    k+=1

result = df_result.pivot_table(index=df_result.columns[:n+margin].tolist()+['Energy'], values=['Occurrences'], aggfunc='sum').sort_values('Energy', ascending=False)
print(result)
#--------------------------------------

#全探索で解く
#--------------------------------------
case = np.array(np.meshgrid(*list(itertools.repeat([0, 1], n)))).T.reshape(-1, n)
case_value = (case * values).sum(axis=1).reshape(-1, 1)
case_weight = (case * weights).sum(axis=1).reshape(-1, 1)
df = pd.DataFrame(columns=list(range(n))+['value', 'weight'], data=np.concatenate([case, case_value, case_weight], axis=1))
tdf = df.loc[df['weight'] <= weight_limit].sort_values('value', ascending=False)
#--------------------------------------
