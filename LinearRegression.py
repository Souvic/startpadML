import numpy as np

import matplotlib.pyplot as plt
#To visualize the result produced by linear regression algorithm
def vislinreg(x,labels,iter=0):
	if(iter==0):
		p=linreg(x,labels)
	else:
		p=linreggrad(x,labels,iter)
	print p
	plt.plot(x,labels,'r--',x,x*p[1]+p[0])
	plt.ylabel('y vs x and the poixs')
	plt.show()
	return p

#Actual algorithm: Here we directly derive the results mathematically, which may be time consuming in case of large datasets.

def linreg(x,labels):
	if(np.array(x.shape).shape==(1,)):
		x=np.array([x*0+1,x]).T
	else:
		x=np.append(x[:,0:1]*0+1,x,axis=1)
	return np.dot(np.dot(np.linalg.pinv(np.dot(x.T,x)),x.T),labels)

#The linear regression algorithm using gradient descent
#As a footnote I would have added: Its always better to initialize the gradient descent with the result of linear regression on a small randomly chosen subset of the whole dataset.
def linreggrad(x,labels,iter):
	x1=np.mean(x,axis=0)
	x11=np.sqrt(np.var(x,axis=0))
	x=(x-x1)/x11
	
	labels1=np.mean(labels)
	labels11=np.sqrt(np.var(labels))
	labels=(labels-labels1)/labels11



	if(np.array(x.shape).shape==(1,)):
		x=np.array([x*0+1,x])
		x=x.T
	else:
		x=np.append(x[:,0:1]*0+1,x,axis=1)
	w=np.array([np.zeros(x.shape[1])])
	w=w.T
	
	for i in range(0,iter):
		
		e=(np.dot(x,w)-labels)/1000
		u=(np.dot(x.T,e))
		w=w-u/10000
	oldw=w
	w=w.T
	w=w*labels11/np.append([[1]],x11)
	w[0,0]=w[0,0]+labels1-np.dot(np.append([[0]],x1)*labels11/np.append([[1]],x11),oldw)
	return w.T
