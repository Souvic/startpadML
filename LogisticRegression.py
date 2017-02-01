#x is the dependent variable columns joined into a matrix and labels is the column matrix of labels, iter is number of iterations you want the model to go through before it returns the output.
#The initialization of weights are fixed to a small number.. it could br small different random numbers also. But in this case it won't matter.
def logreg(x,labels,iter):
	x1=np.mean(x,axis=0)
	x11=np.sqrt(np.var(x,axis=0))
	x=(x-x1)/x11


	if(np.array(x.shape).shape==(1,)):
		x=np.array([x*0+1,x])
		x=x.T
	else:
		x=np.append(x[:,0:1]*0+1,x,axis=1)

	w=np.array([np.zeros(x.shape[1])])+0.1
	w=w.T
	w=w/np.sqrt(np.dot(w.T,w))
	for i in range(0,iter):

		f=1/(1+(np.exp(-np.dot(x,w))))
		loss=-2*(np.dot(np.log(f.T+0.0001),labels)+np.dot(np.log(1.0001-f.T),1-labels))
		print loss
		dloss=np.dot(x.T,(f-labels))/np.sum(x[:,0])*10000
		w=w-dloss
	oldw=w
	labels11=1
	labels1=0
	w=w.T
	w=w*labels11/np.append([[1]],x11)
	w[0,0]=w[0,0]+labels1-np.dot(np.append([[0]],x1)*labels11/np.append([[1]],x11),oldw)
	return w

