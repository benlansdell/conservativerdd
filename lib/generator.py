"""Generate pulls from a linear contextual multi-armed bandit"""
import numpy as np 
import numpy.random as rand

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