""" Implements bandit algorithms 

* LinUCB
* ThresholdBandit 
"""
import numpy as np

class Bandit(object):
 	def __init__(self, generator, delta = 0.1, n_pulls = 10000):
		self.generator = generator
		self.n_pulls = n_pulls
		self.obs = np.zeros(n_pulls)
		self.contexts = np.zeros(n_pulls)
		self.arms = np.zeros(n_pulls)
		self.pull = 0
		self.delta = delta

	def choose_arm(self, ctx):
		raise NotImplementedError

	def update_bandit(self):
		raise NotImplementedError

	def run(self):
		if self.pull >= self.n_pulls:
			print("Bandit: exceeded max pulls, returning 0")
			return 0
		ctx = self.generator.context()
		a = self.choose_arm(ctx)
		obs, regret = self.generator.pull(ctx,a)
		self.contexts[self.pull] = ctx
		self.arms[self.pull] = a 
		self.obs[self.pull] = obs
		self.update_bandit()
		self.pull += 1
		return (ctx, a, obs, regret)

class LinUCB(Bandit):
	def choose_arm(self, ctx):
		return 1

	def update_bandit(self):
		return None

class ThresholdBandit(Bandit):
 	def __init__(self, generator, threshold = 0.5, delta = 0.1, n_pulls = 10000):
		self.threshold = threshold
		super(ThresholdBandit, self).__init__(generator, delta = delta, n_pulls = n_pulls)

	#Threshold policy
	def choose_arm(self, ctx):
		if ctx < self.threshold:
			return 0
		else:
			return 1

	#Update threshold
	def update_bandit(self):
		return None 
