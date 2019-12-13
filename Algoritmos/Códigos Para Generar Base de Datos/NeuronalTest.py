# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:58:19 2019

@author: DULCE
"""

from sklearn.neural_network import MLPClassifier
import pandas as pd

class NeuralNetwork(MLPClassifier):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


df = pd.read_csv(r"C:\Users\DULCE\Desktop\Archivos algoritmos\Baseseleccion.csv")

x = df[["Norma","Correlacion"]]
y = df["Comando"]

clf = NeuralNetwork(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,5,6), random_state=1)

clf.fit(x, y)

predicted = clf.predict(x)
total_matches = len([True for i in predicted == y if i is True ])
total_tc = len(predicted)


print(f"Performance: {total_matches} out of {total_tc}. {(total_matches/total_tc)*100}% Correct classifications")