from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from numpy import diag, matrix, inf
from openopt import QP
import math
import copy



#---------------------------------------------------------------------------------------------------
eps=1.0
C=15.0

X=[]
Y=[]
tot_values=int(raw_input())
for i in range(tot_values):
	X.append(float(raw_input()))
	Y.append(float(raw_input()))



#------------------------------------------------------------------------------------------------------


M=20 #input dim after applying kernel

def transform(x):
	#using rbf kernel
	phi_x=[]
	for i in range(1,M+1):
		phi_x.append(math.exp(-((x-i)**2)/2))
	return phi_x






# calculating sig,m,A,beta

alpha=[1.0 for i in range(M)]
beta=1.0

iterations=0
difference=0
phi_x=[]
t=np.array([Y])
#A is diag(alpha)
A=np.matrix(np.diag(alpha))

for i in X:
	phi_x.append(transform(i))

phi_X=np.matrix(phi_x)

K=(phi_X.T)*(phi_X)

#mean and covariance of posterior distribution of weights
#initializing m,sigma

sigma=[]
sigma=(A+beta*K).I

m=(beta*sigma*(phi_X.T)*t.T)


while(iterations<1000):
	gamma=[float(1.0-A[i,i]*sigma[i,i]) for i in range(M)]
	alpha_1=[float(float(gamma[i])/m[i]**2) for i in range(M)]
	out_i=t.T-phi_X*m
	beta=(len(X)-sum(gamma))/(out_i.T*out_i)
	alpha=copy.copy(alpha_1)
	A=np.matrix(np.diag(alpha))
	sigma=(A+float(beta)*K).I
	m=(float(beta)*sigma*(phi_X.T)*t.T)
	iterations=iterations+1
	


output_X=[]
output_Y=[]
output_X.append(0)
support_vectors=[]
support_vectors_Y=[]
#support vectors are points where m tends to infinite. I took 100000 as infinity
for i in range(len(m)):
	if(abs(m[i])<100000):
		support_vectors.append(X[i])
		support_vectors_Y.append(Y[i])
		

for i in range(350):
	output_X.append(output_X[-1]+float(10)/350)


for i in output_X:
	# w is equal to mean of its posterior distribution
	output_Y.append((m.T)*((np.matrix(transform(i))).T))

plt.scatter(output_X,output_Y,marker='o')
plt.scatter(X,Y,marker='.')
plt.scatter(support_vectors,support_vectors_Y,marker='x')

plt.show()
print alpha

