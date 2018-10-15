""" Implements bandit algorithms 

* LinUCB
* ThresholdBandit 
"""
import numpy as np

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

class BanditAlgorithm(object):
	def __init__(self, generator, delta = 0.1, n_pulls = 10000):
		self.generator = generator
		self.n_pulls = n_pulls
		self.obs = np.zeros(n_pulls)
		self.contexts = np.zeros(n_pulls)
		self.arms_idx = np.zeros(n_pulls)
		self.pull = 0
		self.delta = delta

	def step(self):
		if self.pull >= self.n_pulls:
			print("Bandit: exceeded max pulls, returning 0")
			return 0
		ctx = self.generator.context()
		arm_idx = self._choose_arm(ctx)
		obs, regret = self.generator.pull(ctx,arm_idx)
		self.contexts[self.pull] = ctx
		self.arms_idx[self.pull] = arm_idx
		self.obs[self.pull] = obs
		self._update_bandit()
		self.pull += 1
		return (ctx, arm_idx, obs, regret)

	def _choose_arm(self, ctx):
		raise NotImplementedError

	def _update_bandit(self):
		raise NotImplementedError

	def predict(self, ctx, arm_idx):
		raise NotImplementedError

	def predict_upper(self, ctx, arm_idx):
		raise NotImplementedError

	def predict_lower(self, ctx, arm_idx):
		raise NotImplementedError
		
class LinUCB(BanditAlgorithm):
	def __init__(self, generator, beta = 2, delta = 0.1, n_pulls = 10000):
		self.beta = beta
		self.arms = lambda ctx, arm: (1, ctx, 0, 0) if arm == 1 else (0, 0, 1, ctx)
		self.V = np.identity(4)
		self.U = np.atleast_2d(np.zeros(4)).T
		super(LinUCB, self).__init__(generator, delta = delta, n_pulls = n_pulls)

	def _choose_arm(self, ctx):
		theta = np.dot(np.linalg.inv(self.V), self.U)
		ucbs = []
		for arm in [self.arms(ctx, i) for i in range(2)]:
			arm = np.atleast_2d(arm).T
			ucb = np.dot(theta.T, arm)
			ucb += self.beta*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
			ucbs.append(ucb[0][0])
		return ucbs.index(max(ucbs))

	def _update_bandit(self):
		ctx = self.contexts[self.pull]
		arm = self.arms(ctx, self.arms_idx[self.pull])
		arm = np.atleast_2d(arm).T
		obs = self.obs[self.pull]
		self.V += np.dot(arm, arm.T)
		self.U += obs*arm

	def predict(self, ctx, arm_idx):
		arm = self.arms(ctx, arm_idx)
		arm = np.atleast_2d(arm).T
		theta = np.dot(np.linalg.inv(self.V), self.U)
		pred = np.dot(theta.T, arm)
		return pred

	def predict_upper(self, ctx, arm_idx):
		arm = self.arms(ctx, arm_idx)
		arm = np.atleast_2d(arm).T
		theta = np.dot(np.linalg.inv(self.V), self.U)
		pred = np.dot(theta.T, arm)
		pred += self.beta*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred

	def predict_lower(self, ctx, arm_idx):
		arm = self.arms(ctx, arm_idx)
		arm = np.atleast_2d(arm).T
		theta = np.dot(np.linalg.inv(self.V), self.U)
		pred = np.dot(theta.T, arm)
		pred -= self.beta*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred

	def pred_arm(self, arm_idx, N = 100):
		xvals = np.linspace(0, 1, N)
		upper = np.zeros(N)
		lower = np.zeros(N)
		mean = np.zeros(N)
		for ctx in xvals:
			arm = self.arms(ctx, arm_idx)
			arm = np.atleast_2d(arm).T
			theta = np.dot(np.linalg.inv(self.V), self.U)
			pred = np.dot(theta.T, arm)
			lower = pred - self.beta*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
			upper = pred + self.beta*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred, lower, upper


class ThresholdBandit(LinUCB):
	def __init__(self, generator, threshold = 0.5, beta = 2, delta = 0.1, n_pulls = 10000):
		self.threshold = threshold
		N = 100
		self.xvals = np.linspace(*generator.params.bounds, num=N)
		super(ThresholdBandit, self).__init__(generator, beta = beta, delta = delta, n_pulls = n_pulls)

	def _choose_arm(self, ctx):
		if ctx < self.threshold:
			return 0
		else:
			return 1

	def _update_bandit(self):
		#Update ridge regression parameters
		super(ThresholdBandit, self)._update_bandit()
		minmax = np.zeros((self.xvals.shape[0], 2))

		self.lower_bound = 0
		self.upper_bound = 1

		#This is finding al0 = au1
		minmax[:, 0] = [self.predict_lower(i, 0) - self.predict_upper(i, 1) for i in self.xvals]
		#This is finding al1 = au0
		minmax[:, 1] = [self.predict_upper(i, 0) - self.predict_lower(i, 1) for i in self.xvals]

		#print minmax

		#Check if there's a change in sign. If there isn't, then lower bound stays at 0
		if not np.all(minmax[:,0] < 0) and not np.all(minmax[:,0] > 0):
			self.lower_bound = self.xvals[first_nonzero(minmax[:,0]<0, 0)]

		if not np.all(minmax[:,1] < 0) and not np.all(minmax[:,1] > 0):
			self.upper_bound = self.xvals[first_nonzero(minmax[:,1]<0, 0)]

		self.threshold = min(self.upper_bound, max(self.threshold, self.lower_bound))
