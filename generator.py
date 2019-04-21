"""Generate pulls from a linear contextual multi-armed bandit"""
import numpy as np 
import numpy.random as rand
import pandas as pd 

class LinearGeneratorParams(object):

	def __init__(self, alpha, beta, sigma_eps = 0.1, bounds = (0,1), r_max = 1, d = 2, k = 2, intercept = True):
		self.alpha = alpha
		self.beta = beta
		self.sigma_eps = sigma_eps
		self.bounds = bounds
		self.r_max = r_max
		self.d = d
		self.k = k
		self.intercept = intercept

class LinearGenerator(object):

	def __init__(self, params):
		self.params = params

	def context(self):
		val = rand.rand(self.params.d)*(self.params.bounds[1]-self.params.bounds[0]) + self.params.bounds[0]
		if self.params.intercept:
			val[0] = 1
		return val

	def pull(self, ctx, a):
		eta = rand.randn()*self.params.sigma_eps
		#print ctx, self.params.alpha[1,:]
		exp_rewards = [np.dot(ctx[1:],self.params.alpha[i,:]) + self.params.beta[i] for i in range(self.params.k)]
		exp_reward = exp_rewards[a]
		regret = max(exp_rewards) - exp_reward
		obs = exp_reward + eta
		return obs, regret
    
    
class DataGeneratorParams(object):

	def __init__(self, df, yInd, xInds=None, intercept = True):
		self.intercept = intercept
		self.df = df
		self.n = df.shape[0]
		self.yInd = yInd
		if intercept:
			#Append a vector of ones for the constant term
			ones = pd.DataFrame({'ones': np.ones(df.shape[0])})
			self.df = pd.concat([self.df, ones], axis=1)

		if xInds is not None:
			self.xInds = xInds
		else:
			#All the values that are not yInd
			self.xInds = [x for x in range(df.shape[1]) if x != yInd]
			if intercept:
				self.xInds.append(df.shape[1])
		self.d = len(self.xInds)
		self.k = df[df.columns[yInd]].unique().shape[0]
        #self._init_true_params()

	def _init_true_params(self):
		self.theta = np.zeros((self.d, self.k))
		#For each arm
		for idx in range(self.k):
			#Get the data
			data = self.df.loc[self.df['Class'] == idx+1].values
			#Solve the least squares


			#Save the params


class DataGenerator(object):

	def __init__(self, params):
		self.params = params
		self.count = -1

	def context(self):
		self.count += 1
		self.count = self.count % self.params.n
		val = self.params.df.iloc[self.count, self.params.xInds]
		return val

	def pull(self, ctx, a):
		taken_action = self.params.df.iloc[self.count,self.params.yInd]-1
		regret = 1 - (taken_action == a)
		obs = taken_action
		return (taken_action == a).astype(int), regret