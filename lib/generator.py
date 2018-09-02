"""Generate pulls from a linear contextual multi-armed bandit"""
import numpy as np 
import numpy.random as rand

class LinearGeneratorParams(object):

	def __init__(self, alpha, beta, sigma_eps = 0.1, bounds = (0,1), r_max = 1):
		self.alpha = alpha
		self.beta = beta
		self.sigma_eps = sigma_eps
		self.bounds = bounds
		self.r_max = r_max

class LinearGenerator(object):

	def __init__(self, params):
		self.params = params

	def context(self):
		return rand.rand()*(self.params.bounds[1]-self.params.bounds[0]) + self.params.bounds[0]

	def pull(self, ctx, a):
		eta = rand.randn()*self.params.sigma_eps
		exp_rewards = [ctx*self.params.alpha[i] + self.params.beta[i] for i in (0,1)]
		exp_reward = exp_rewards[a]
		regret = max(exp_rewards) - exp_reward
		obs = exp_reward + eta
		return obs, regret