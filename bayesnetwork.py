from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np

#Preprocessing
hd = pd.read_csv('heart.csv')
hd = hd.replace('?', np.nan)

#Model and fit
model = BayesianModel([('age','target'),('sex','target'),('trestbps','target'),('fbs','target'),('target','chol'),('target','restecg')])
model.fit(hd, estimator = MaximumLikelihoodEstimator)

#Draw
import networkx as nx
nx.draw(model, with_labels = True)

#Inference
infer = VariableElimination(model)
q1 = infer.query(variables=['target'],evidence = {'sex':0})
print(q1)
q2 = infer.query(variables=['target'], evidence = {'fbs':1})
print(q2)

for i in model.get_cpds():
    print(i)