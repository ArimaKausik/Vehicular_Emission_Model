#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np # Necessary libaries are imported
from numpy.random import randint
import pandas as pd
from numpy.random import random as rnd
from random import gauss, randrange
from numpy import cumsum
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from sklearn import preprocessing
import pandas as pd
df = pd.read_csv('UpdatedDataset.csv', sep='\s*,\s*') # Reading the dataset from csv file
df.as_matrix() # converting the dataset as matrix for applying mathematical operations
X = df.iloc[:, :4] # Separating the input parameters and output variables
Y = df.iloc[:, 4:]
# Creating a hardcoded list of minimum and maximum of input parameters
# (These min and max values are used in mutation operation)
maxlist = []
minlist = []
maxlist.append(100) # Diesel blend %
maxlist.append(4) # Exhaust gas recirculation (EGR)
maxlist.append(180) # Air-fuel ratio (AFR)
maxlist.append(2500) # Injection pressure (Pinj)
minlist.append(0) # Diesel blend %
minlist.append(0.5) # Exhaust gas recirculation (EGR)
minlist.append(35) # Air-fuel ratio (AFR)
minlist.append(500) # Injection pressure (Pinj)
# preprocessing the input parameters and output parameters, by scaling the values with respect to the original dataset
scaler = preprocessing.StandardScaler().fit(X)
scalery = preprocessing.StandardScaler().fit(Y)
mutationPerc = 0.01 # This is the mutation percentage 1%
import random
import random as rd

def oneGeneration(initialGen): # This is the logic applied for each and every generation
	parent = selection(initialGen) # Selection is performed from the given generation
	embryo = mating(parent) # Mating is performed from the selected parents and embryo is formed
	children = mutation(embryo) # Mutuation is applied to some of the embryo,based on mutation percentage
	livingGen = children # After all these operations next generation is created
	return livingGen

def nGenerations(n, initialGen):
	gen = initialGen # With the given population as initial generation, process starts.
	for i in range(n): # for 'n' number of times the OneGeneration method is applied ,and 'n' number of generations is produced.
		nextGen = oneGeneration(gen)
		gen = nextGen
		# The below lines were performed in order to produce an excel(csv) file for each generations' population as well as their emissions
	np.savetxt('gen' + str(i + 1) + '.csv', gen, delimiter=',')
	pops = scaler.transform(gen)
	model = load_model('newmod1.h5')
	yhat = model.predict(pops)
	y_pred = scalery.inverse_transform(yhat)
	pops = scaler.inverse_transform(pops)
	np.savetxt('emissions ' + str(i + 1) + '.csv', pops,delimiter=',')
	# The code for excel(csv) file creation ends here
	return gen

def selection(gen): # Roulette wheel selection is implemented in this selection method
	fitness = []
	Sum = float(0)
	sumlist = []
	fitness = Fitness(gen) # fitness of the given generation is found from 'Fitness' method
	totsum = sum(fitness) # Total of the fitness values is found
	fitness = fitness / totsum # Each of the fitness value is divided from the totalsum
	for i in range(len(fitness)): # A list of cumulative sum is created from this for loop
		Sum += fitness[i]
		sumlist.append(Sum)
	parent1 = []
	parent2 = []
	indi = []
	for j in range(len(gen)): # Parents are selected
		reqFit = randrange(0, 100) / 100 # Roulette wheel selection is performed in order to give more chance for the fitter individuals to be selected for mating
		i = 0
		while reqFit > sumlist[i]:
			i += 1
		parent1.append(gen[i])
		ind = gen[i]
		reqFit = random.randrange(0, 100) / 100
		i = 0
		while reqFit > sumlist[i]:
			i += 1
		parent2.append(gen[i])
		parents = [parent1, parent2] # A pair of parents are selected
	return parents

def mating(pair): # Mating is performed ,where the embryos' genes are taken randomly from either of the parents
	offsprings = []
	for i in range(len(pair[0])):
		ind1 = pair[0][i]
	ind2 = pair[1][i]
	m1 = np.matrix(ind1)
	l1 = m1.tolist()
	m2 = np.matrix(ind2)
	l2 = m2.tolist()
	offs = []
	for j in range(4):
		choice = rd.random() < 0.5
		if choice:
			offs.append(l1[0][j])
		else:
			offs.append(l2[0][j])
		offsprings.append(offs)
	return offsprings

def mutation(gen): # Mutation is performed ,such that the genes lie in the range of min and max limits
	noofind = len(gen)
	genes = len(gen[0])
	for i in range(int(mutationPerc * noofind * genes)):
		x = randrange(0, genes)
		y = randrange(0, noofind)
		gen[y][x] = minlist[x] + randrange(0, 100) * ((maxlist[x]- minlist[x]) / 100)
	return gen

avgfit = []
maxfit = []
c1 = 1
c2 = 1
c3 = 1
c4 = 1
c5 = 0.02

def Fitness(pops):
	pops = scaler.transform(pops) # The given generation is scaled in order to be applied on the ANN model file
	model = load_model('newmod1.h5') # The pre prepared ANN model file is loaded
	yhat = model.predict(pops) # The model is used to produce the emissions' scaled values
	y_pred = scalery.inverse_transform(yhat) # An inversion of scaling is performed on the emissions' scaled values
	obj = []
	fit = []
	pops = scaler.inverse_transform(pops) # An inversion of scaling is performed on the input variables
	maxx = 0
	summ = 0
	for i in range(60): # Fitness function is calculated for all the individuals in the population
		IMEP = y_pred[i][0]
		THC = y_pred[i][1]
		NOx = y_pred[i][2]
		soot = y_pred[i][3]
		fit_val = 1 + c5 * ((IMEP - 2) / 8) - (c1 * (NOx - 250) / (2500- 250) + c2 * (soot - 0.0573) / (0.3946 - 0.0573) + c3* (THC - 25) / (180 - 25))
		if maxx < fit_val:
			maxx = fit_val
		summ += fit_val
		fit.append(fit_val)
		avgfit.append(summ * 1.0 / 60) # Average fitness of the generation is put into the avgfit list
		maxfit.append(maxx) # Maximum fitness of the generation is put into the maxfit list
	return fit

if __name__=='__main__':
	D = X['Diesel%']
	E = X['EGR']
	A = X['AFR']
	P = X['P_inj']
	gen = []
	for i in range(len(D)):
		ind = []
		ind.append(D[i])
		ind.append(E[i])
		ind.append(A[i])
		ind.append(P[i])
		gen.append(ind)
	Gen = np.matrix(gen)
# These below lines run for 10 generations and print the entire population of the last generation
# and also shows the 0-th individual and also its corresponding emissions and power produced
	i = 0
	n = 10 # The number of generations can be changed here
	LatestGen = nGenerations(n, Gen)
	print LatestGen
	print "================================================"
	print 'After ' + str(n) + ' generations'
	print 'For ' + str(i) + ' th individual'
	pops = LatestGen
	print pops[i]
	pops = scaler.transform(pops)
	model = load_model('newmod1.h5')
	yhat = model.predict(pops)
	y_pred = scalery.inverse_transform(yhat)
	print ('IMEP :', y_pred[i][0])
	print ('THC :', y_pred[i][1])
	print ('NOx :', y_pred[i][2])
	print ('soot :', y_pred[i][3])
	print "================================================"
	InAndOut = np.concatenate((scaler.inverse_transform(pops), y_pred), axis=1)
	asarr = np.asarray(Fitness(pops))
	asmat = []
	for i in range(60):
		asmat.append([asarr[i]])
		FinOut = np.concatenate((InAndOut, asmat), axis=1)
		np.savetxt('test.csv', FinOut, delimiter=',')

	df = pd.read_csv('test.csv', header=None)
	df.rename(columns={0: 'Diesel%',1: 'EGR',2: 'AFR',3: 'P_inj',4: 'IMEP',5: 'THC',6: 'NOx',7: 'soot',8: 'Fitness',}, inplace=True)
	df.to_csv('Gen.csv', index=False)
	i = 0
	n = 100
	LatestGen = nGenerations(n, Gen)
	print LatestGen
	print "================================================"
	print 'After ' + str(n) + ' generations'
	print 'For ' + str(i) + ' th individual'
	pops = LatestGen
	print pops[i]
	pops = scaler.transform(pops)
	model = load_model('newmod1.h5')
	yhat = model.predict(pops)
	y_pred = scalery.inverse_transform(yhat)
	print ('IMEP :', y_pred[i][0])
	print ('THC :', y_pred[i][1])
	print ('NOx :', y_pred[i][2])
	print ('soot :', y_pred[i][3])
	# this is the list of average fitness of a generation for 100 generations
	print avgfit
	np.savetxt('avgfit.csv', avgfit, delimiter=',')
	# this is the list of maximum fitness in a generation for 100 generations
	print maxfit