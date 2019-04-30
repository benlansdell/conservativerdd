""" Implements bandit algorithms 

* LinUCB
* ThresholdBandit 
* ThresholdConsBandit
* Greedy algorithm
* Conservative LinUCB algorithm (Kazerouni et al 2017)
"""

import numpy as np
import numpy.random as rand

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

#Compute expected regret for a current configuration and context distribution. A sampling approach
def expected_regret(bandit_alg, generator, N_pulls = 100000):
	ctxs = generator.contexts(N_pulls)
	actions = bandit_alg.choose_arms(ctxs)
	_, regret, _ = generator.pulls(ctxs, actions)
	return np.mean(regret)

#The exact quadrature implementation


class BanditAlgorithm(object):
	def __init__(self, generator, delta = 0.1, n_pulls = 10000):
		self.generator = generator
		self.d = generator.params.d #Dimension of context
		self.k = generator.params.k #Number of arms
		self.n_pulls = n_pulls
		self.obs = np.zeros(n_pulls)
		self.contexts = np.zeros((n_pulls, self.d))
		self.arms_idx = np.zeros(n_pulls)
		self.pull = 0
		self.delta = delta

	def step(self):
		if self.pull >= self.n_pulls:
			print("Bandit: exceeded max pulls, returning 0")
			return 0
		ctx = self.generator.context()
		arm_idx = self._choose_arm(ctx)
		obs, regret, _ = self.generator.pull(ctx,arm_idx)
		self.contexts[self.pull,:] = ctx
		self.arms_idx[self.pull] = arm_idx
		self.obs[self.pull] = obs
		self._update_bandit()
		self.pull += 1
		return (ctx, arm_idx, obs, regret)

	def choose_arms(self, ctxs):
		raise NotImplementedError

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

	def __init__(self, generator, delta = 0.1, n_pulls = 10000, lambd = 1e-4):
		L = 1
		self.lamd = lambd*n_pulls
		self.beta = lambda v: np.sqrt(self.lamd)*L + np.sqrt(np.log(np.linalg.det(v))-self.d*np.log(self.lamd)-2*np.log(delta))
		self.V = self.lamd*np.identity(generator.params.k*generator.params.d)
		self.U = np.atleast_2d(np.ones(generator.params.k*generator.params.d)).T
		super(LinUCB, self).__init__(generator, delta = delta, n_pulls = n_pulls)

	def theta(self):
		return np.dot(np.linalg.inv(self.V), self.U)

	def arms(self, ctx, arm):
		a = np.zeros(self.d*self.k)
		a[int(arm)*self.d:((int(arm)+1)*self.d)] = ctx
		return a

	def _choose_arm(self, ctx):
		theta = np.dot(np.linalg.inv(self.V), self.U)
		ucbs = []
		#print(np.linalg.det(self.V))
		for arm in [self.arms(ctx, i) for i in range(self.k)]:
			arm = np.atleast_2d(arm).T
			ucb = np.dot(theta.T, arm)
			ucb += self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
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
		pred += self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred

	def predict_lower(self, ctx, arm_idx):
		arm = self.arms(ctx, arm_idx)
		arm = np.atleast_2d(arm).T
		theta = np.dot(np.linalg.inv(self.V), self.U)
		pred = np.dot(theta.T, arm)
		pred -= self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
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
			lower = pred - self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
			upper = pred + self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred, lower, upper

class GreedyBandit(LinUCB):

	def _choose_arm(self, ctx):
		theta = np.dot(np.linalg.inv(self.V), self.U)
		vals = []
		for arm in [self.arms(ctx, i) for i in range(self.k)]:
			arm = np.atleast_2d(arm).T
			val = np.dot(theta.T, arm)
			vals.append(val[0][0])
		return vals.index(max(vals))

class ThresholdBandit(LinUCB):
	def __init__(self, generator, delta = 0.1, n_pulls = 10000, lambd = 1e-4):
#		alpha = generator.params.alpha
		self.update_theta = np.zeros(n_pulls)
		#How to start the arms?????
		#self.theta_tilde = np.ones((alpha.shape[0], alpha.shape[1]+1))
#		self.theta_tilde = np.random.rand(alpha.shape[0], alpha.shape[1]+1)
		self.theta_tilde = np.random.rand(generator.params.k, generator.params.d)
		super(ThresholdBandit, self).__init__(generator, delta = delta, n_pulls = n_pulls, lambd = lambd)

	def choose_arms(self, ctxs):
		N = ctxs.shape[0]
		vals = np.zeros((N, self.k))
		for idx in range(self.k):
			theta_k = self.theta_tilde[idx,:]
			val = np.dot(ctxs,theta_k)
			vals[:,idx] = val
		return np.argmax(vals, axis = 1)

	def _choose_arm(self, ctx):
		vals = []
		for idx in range(self.k):
			theta_k = self.theta_tilde[idx,:]
			val = np.dot(theta_k.T, ctx)
			vals.append(val)
		return vals.index(max(vals))

	def _update_bandit(self):

		#Update ridge regression parameters
		super(ThresholdBandit, self)._update_bandit()

		#Then project each theta_tilde parameter onto the confidence set for each arm
		k = self.k
		d = self.d
		for idx in range(k):
			#Get arm k's V and U
			V_k = self.V[idx*d:((idx+1)*d),idx*d:((idx+1)*d)]
			U_k = self.U[idx*d:((idx+1)*d)]
			theta_k = np.atleast_2d(self.theta_tilde[idx,:]).T
			theta_hat_k = np.dot(np.linalg.inv(V_k), U_k)
			diff_k = theta_k - theta_hat_k
			beta_k = self.beta(V_k)
			norm_k = np.dot(diff_k.T, np.dot(V_k, diff_k))
			#If theta_k is not in confidence region then update
			if norm_k > beta_k:
				self.update_theta[self.pull] = 1
				#Greedy update
				theta_k = theta_hat_k
				#Conservative update
				#theta_k = theta_hat_k + beta_k*diff_k/norm_k
			self.theta_tilde[idx,:] = np.squeeze(theta_k)

class ThresholdBaselineBandit(ThresholdBandit):
	def __init__(self, generator, delta = 0.1, n_pulls = 10000, lambd = 1e-4):
		self.update_theta = np.zeros(n_pulls)
		self.theta_tilde = np.random.rand(generator.params.k, generator.params.d)
		super(ThresholdBandit, self).__init__(generator, delta = delta, n_pulls = n_pulls, lambd = lambd)

	#def choose_arms(self, ctxs):
	#	N = ctxs.shape[0]
	#	vals = np.zeros((N, self.k))
	#	for idx in range(self.k):
	#		theta_k = self.theta_tilde[idx,:]
	#		val = np.dot(ctxs,theta_k)
	#		vals[:,idx] = val
	#	return np.argmax(vals, axis = 1)

	def _choose_arm(self, ctx):
		vals = np.zeros(self.k)
		preds_low = np.zeros(self.k)
		preds_high = np.zeros(self.k)
		viable = np.ones(self.k)
		for idx in range(self.k):
			arm = self.arms(ctx, idx)
			arm = np.atleast_2d(arm).T
			theta = np.dot(np.linalg.inv(self.V), self.U)
			pred = np.dot(theta.T, arm)
			pm = self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
			pred_high = pred + pm
			pred_low = pred - pm
			theta_k = self.theta_tilde[idx,:]
			val = np.dot(theta_k.T, ctx)
			vals[idx] = val
			preds_high[idx] = pred_high
			preds_low[idx] = pred_low

		#Determine which arms aren't viable
		for idx in range(self.k):
			if preds_high[idx] < np.max(preds_low):
				viable[idx] = 0

		#Play according to theta_zero for the viable arms
		return vals.tolist().index(max(vals[viable.astype(bool)]))

	def _update_bandit(self):
		#Don't update theta_tilde here....
		pass

class ThresholdConsBandit(ThresholdBandit):

	def _update_bandit(self):

		#Update ridge regression parameters
		super(ThresholdBandit, self)._update_bandit()

		#Then project each theta_tilde parameter onto the confidence set for each arm
		k = self.k
		d = self.d
		for idx in range(k):
			#Get arm k's V and U
			V_k = self.V[idx*d:((idx+1)*d),idx*d:((idx+1)*d)]
			U_k = self.U[idx*d:((idx+1)*d)]
			theta_k = np.atleast_2d(self.theta_tilde[idx,:]).T
			theta_hat_k = np.dot(np.linalg.inv(V_k), U_k)
			diff_k = theta_k - theta_hat_k
			beta_k = self.beta(V_k)
			norm_k = np.dot(diff_k.T, np.dot(V_k, diff_k))
			#If theta_k is not in confidence region then update
			if norm_k > beta_k:
				self.update_theta[self.pull] = 1
				#print "Updating theta_tilde"
				#Greedy update
				#theta_k = theta_hat_k
				#Conservative update
				theta_k = theta_hat_k + beta_k*diff_k/norm_k
			self.theta_tilde[idx,:] = np.squeeze(theta_k)

class ThresholdMaxConsBandit(ThresholdBandit):

	def _update_bandit(self):

		#Update ridge regression parameters
		super(ThresholdBandit, self)._update_bandit()

		#Now we only update parameters if outside the decision boundary.

		#This involves minimizing the cosine distance between the decision lines...

		#Solve with ADMM? 

		#Then project each theta_tilde parameter onto the confidence set for each arm
		k = self.k
		d = self.d
		for idx in range(k):
			#Get arm k's V and U
			V_k = self.V[idx*d:((idx+1)*d),idx*d:((idx+1)*d)]
			U_k = self.U[idx*d:((idx+1)*d)]
			theta_k = np.atleast_2d(self.theta_tilde[idx,:]).T
			theta_hat_k = np.dot(np.linalg.inv(V_k), U_k)
			diff_k = theta_k - theta_hat_k
			beta_k = self.beta(V_k)
			norm_k = np.dot(diff_k.T, np.dot(V_k, diff_k))
			#If theta_k is not in confidence region then update
			if norm_k > beta_k:
				self.update_theta[self.pull] = 1
				#print "Updating theta_tilde"
				#Greedy update
				#theta_k = theta_hat_k
				#Conservative update
				theta_k = theta_hat_k + beta_k*diff_k/norm_k
			self.theta_tilde[idx,:] = np.squeeze(theta_k)

class ConsLinUCB(BanditAlgorithm):
	"""
	Based on: Conservative contextual linear bandits

	Kazerouni et al 2017

	https://arxiv.org/pdf/1611.06426.pdf
	"""

	def __init__(self, generator, baseline_params, delta = 0.001, n_pulls = 10000, lambd = 1e-4, alpha = 0.1):
		self.L = 1
		self.alpha = alpha
		self.baseline_alpha = baseline_params[0]
		self.baseline_beta = baseline_params[1]
		self.lamd = lambd*n_pulls
		self.beta = lambda v: np.sqrt(self.lamd)*self.L + np.sqrt(np.log(np.linalg.det(v))-self.d*np.log(self.lamd)-2*np.log(delta))
		self.V = self.lamd*np.identity(generator.params.k*generator.params.d)
		self.U = np.atleast_2d(np.ones(generator.params.k*generator.params.d)).T
		self.reward_baseline_cons = 0
		self.reward_baseline_all = 0
		super(ConsLinUCB, self).__init__(generator, delta = delta, n_pulls = n_pulls)

	def theta(self):
		return np.dot(np.linalg.inv(self.V), self.U)

	def step(self):
		if self.pull >= self.n_pulls:
			print("Bandit: exceeded max pulls, returning Nones")
			return [None]*4
		ctx = self.generator.context()
		opt_arm_idx = self._choose_opt_arm(ctx)
		obs, regret, baseline_reward = self._play_baseline_arm(ctx)
		self.reward_baseline_all += baseline_reward
		L = self._compute_L(opt_arm_idx, ctx) + self.reward_baseline_cons
		#Decide to play optimistic or baseline action
		#print("LHS: %f, RHS: %f"%(L, (1-self.alpha)*self.reward_baseline_all))
		if L >= (1-self.alpha)*self.reward_baseline_all:
		#if 2 > 1:
			arm_idx = opt_arm_idx
			obs, regret, _ = self.generator.pull(ctx,arm_idx)
			self.contexts[self.pull,:] = ctx
			self.arms_idx[self.pull] = arm_idx
			self.obs[self.pull] = obs
			self._update_bandit()
			#print("Not playing baseline!")
		else:
			arm_idx = -1
			self.reward_baseline_cons += baseline_reward
			self.contexts[self.pull,:] = ctx
			self.arms_idx[self.pull] = arm_idx
			self.obs[self.pull] = obs
		self.pull += 1
		return (ctx, arm_idx, obs, regret)

	def _compute_L(self, opt_arm_idx, ctx):
		L = 0
		for idx in range(self.k):
			#Compute z_t (the sum of all previous contexts where action k was played, plus the current context)
			zt = np.sum(self.contexts[self.arms_idx == idx,:], axis = 0)
			if idx == opt_arm_idx:
				zt += ctx
			L += self.predict_lower(zt, idx)
		#print("L is %f"%L)
		return L

	def arms(self, ctx, arm):
		a = np.zeros(self.d*self.k)
		a[int(arm)*self.d:((int(arm)+1)*self.d)] = ctx
		return a

	def _play_baseline_arm(self, ctx):
		eta = rand.randn()*self.generator.params.sigma_eps
		expt_reward = np.dot(ctx[1:],self.baseline_alpha) + self.baseline_beta
		obs = expt_reward + eta
		#Compare w optimistic arms to get regret
		ucbs = []
		theta = np.dot(np.linalg.inv(self.V), self.U)
		for arm in [self.arms(ctx, i) for i in range(self.k)]:
			arm = np.atleast_2d(arm).T
			ucb = np.dot(theta.T, arm)
			ucbs.append(ucb[0][0])
		ucbs.append(expt_reward)
		regret = max(ucbs) - expt_reward
		return obs, regret, expt_reward

	def _choose_opt_arm(self, ctx):
		theta = np.dot(np.linalg.inv(self.V), self.U)
		ucbs = []
		for arm in [self.arms(ctx, i) for i in range(self.k)]:
			arm = np.atleast_2d(arm).T
			ucb = np.dot(theta.T, arm)
			ucb += self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
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
		pred += self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred

	def predict_lower(self, ctx, idx):
		arm_idxs = self.arms_idx[0:self.pull]
		arm_idxs = arm_idxs[arm_idxs == idx]
		if len(arm_idxs) == 0:
			pred = 0
		else:
			d = self.d
			V_k = self.V[idx*d:((idx+1)*d),idx*d:((idx+1)*d)]
			U_k = self.U[idx*d:((idx+1)*d)]
			theta = np.dot(np.linalg.inv(V_k), U_k)
			pred = np.dot(theta.T, ctx)
			pred -= self.beta(V_k)*np.sqrt(np.dot(ctx.T, np.dot(np.linalg.inv(V_k), ctx)))
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
			lower = pred - self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
			upper = pred + self.beta(self.V)*np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.V), arm)))
		return pred, lower, upper
