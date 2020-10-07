import numpy as np

class KNN:
	def __init__(self, k):
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		self.XTrain = X
		self.yTrain = y

	def predict(self, X):
		kValues=[]
		for i in range(0, X.shape[0]): # shape function lets the parsing be flexible for the testing fata of all the datasets.
			distances=[]
			minIndexes=[]
			for j in range(0, self.XTrain.shape[0]):  #checking the testing data with the training data for eucledian distances.
				distances.append(self.distance(self.XTrain[j], X[i]))
			k=self.k
			while k>0:
				minIndex=distances.index(min(distances))  #while k is any value greater than 0 (3 in this case) it will store the index of the values in minIndex
				minIndexes.append(self.yTrain[minIndex]) #appending the labels of the minimum values in the list minIndexes
				distances[minIndex] = float("inf") #setting the values of Indexes already checked to infinity so they are never considered while checking the minimum
				k=k-1
			sorted(minIndexes) #sorting to find the values that occur the most in the list
			maxValue = max(minIndexes, key=minIndexes.count) #finding the max values by count
			kValues.append(maxValue)
		return np.array(kValues)

class Perceptron:
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		self.XTrain = X
		self.yTrain = y
		for s in range(steps):     
			i = s%(self.yTrain.size)    #taking steps for all data points
			yNewTemp = self.XTrain[i].dot(self.w)+self.b    #calculationg the activation function for the previous weights if its less than 0 then pass 0 else pass 1
			if yNewTemp>0:
				yNew = 1
			else:
				yNew = 0
			if yNew != self.yTrain[i]:         # for all values where y is not equal to y* we calculate the difference in y and use the difference to calculate the new weigths
				diff = self.yTrain[i] - yNew
				self.w = self.w +self.lr* diff *self.XTrain[i]
    
	def predict(self, X):
		activation=[]
		sum = X.dot(self.w) + self.b  #calculating the sum of the values of weights with the testing data and the bias
		for i in range(0, sum.shape[0]):
			if sum[i]>0:     #is the sum is greater than 0 then we make the activation 1 otherwise 0 
				activation.append(1)
			else:
				activation.append(0)
		return np.array(activation)

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		self.input = input
		sum = input.dot(self.w)+self.b #forward just returns the sum of the input multiplied by the weights and the bias.
		return sum

	def backward(self, gradients):
		wNew = self.input.transpose().dot(gradients) #the algorithm in the slides where we take out the weights with the gradients and we then change the inputs and update the weights and biases
		XNew = gradients.dot(self.w.transpose())      
		self.w = self.w - self.lr * wNew
		self.b = self.b - self.lr * gradients
		return XNew

class Sigmoid:

	def __init__(self):
		None

	def sigmoidcalc(self,x):
		np.seterr(over='ignore')     #to supress the warning as told by the TA on forums
		return 1/(1+np.exp(-x))      # to calculate the sigmoid

	def forward(self, input):
		self.input = input
		return self.sigmoidcalc(self.input)  #inputs with respect to sigmoids

	def backward(self, gradients):
		return (1-self.sigmoidcalc(self.input)) * self.sigmoidcalc(self.input) * gradients #sigmoid gradient for backward.
