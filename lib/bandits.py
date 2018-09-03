""" Implements bandit algorithms 

* LinUCB
* ThresholdBandit 
"""
import numpy as np

class BanditAlgorithm(object):
	def __init__(self, generator, delta = 0.1, n_pulls = 10000):
		self.generator = generator
		self.n_pulls = n_pulls
		self.obs = np.zeros(n_pulls)
		self.contexts = np.zeros(n_pulls)
		self.arms_idx = np.zeros(n_pulls)
		self.pull = 0
		self.delta = delta

	def choose_arm(self, ctx):
		raise NotImplementedError

	def update_bandit(self):
		raise NotImplementedError

	def step(self):
		if self.pull >= self.n_pulls:
			print("Bandit: exceeded max pulls, returning 0")
			return 0
		ctx = self.generator.context()
		arm_idx = self.choose_arm(ctx)
		obs, regret = self.generator.pull(ctx,arm_idx)
		self.contexts[self.pull] = ctx
		self.arms_idx[self.pull] = arm_idx
		self.obs[self.pull] = obs
		self.update_bandit()
		self.pull += 1
		return (ctx, arm_idx, obs, regret)

class LinUCB(BanditAlgorithm):
	def __init__(self, generator, beta = 2, delta = 0.1, n_pulls = 10000):
		self.beta = beta
		self.arms = lambda ctx, arm: (1, ctx, 0, 0) if arm == 1 else (0, 0, 1, ctx)
		self.V = np.identity(4)
		self.U = np.atleast_2d(np.zeros(4)).T
		super(LinUCB, self).__init__(generator, delta = delta, n_pulls = n_pulls)

	def choose_arm(self, ctx):
		theta = np.dot(np.linalg.inv(self.V), self.U)
		ucbs = []
		for arm in [self.arms(ctx, i) for i in range(2)]:
			arm = np.atleast_2d(arm).T
			ucb = np.dot(theta.T, arm)
			ucb += self.beta*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
			ucbs.append(ucb[0][0])
		return ucbs.index(max(ucbs))

	def update_bandit(self):
		ctx = self.contexts[self.pull]
		arm = self.arms(ctx, self.arms_idx[self.pull])
		arm = np.atleast_2d(arm).T
		obs = self.obs[self.pull]
		self.V += np.dot(arm, arm.T)
		self.U += obs*arm

class ThresholdBandit(LinUCB):
	def __init__(self, generator, threshold = 0.5, beta = 2, delta = 0.1, n_pulls = 10000):
		self.threshold = threshold
		super(ThresholdBandit, self).__init__(generator, beta = beta, delta = delta, n_pulls = n_pulls)

	#Threshold policy
	def choose_arm(self, ctx):
		if ctx < self.threshold:
			return 0
		else:
			return 1

	#Update threshold
	def update_bandit(self):
		arm = self.arms[self.arms[-1],:]
		arm = np.atleast_2d(arm).T
		obs = self.obs[-1]
		self.V += np.dot(arm, arm.T)
		self.U += obs*arm

		#
