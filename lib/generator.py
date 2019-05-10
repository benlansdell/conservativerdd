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

	def contexts(self, N_pulls):
		vals = rand.rand(N_pulls, self.params.d)*(self.params.bounds[1]-self.params.bounds[0]) + self.params.bounds[0]
		if self.params.intercept:
			vals[:,0] = 1
		return vals

	def pull(self, ctx, a):
		eta = rand.randn()*self.params.sigma_eps
		#print ctx, self.params.alpha[1,:]
		exp_rewards = [np.dot(ctx[1:],self.params.alpha[i,:]) + self.params.beta[i] for i in range(self.params.k)]
		exp_reward = exp_rewards[a]
		regret = max(exp_rewards) - exp_reward
		obs = exp_reward + eta
		return obs, regret, exp_reward

	def pulls(self, ctxs, actions):
		n = ctxs.shape[0]
		etas = rand.randn(n)*self.params.sigma_eps
		#print ctx, self.params.alpha[1,:]
		exp_rewards = np.array([np.dot(ctxs[:,1:],self.params.alpha[i,:]) + self.params.beta[i] for i in range(self.params.k)])

		#Gotta be a quicker way to do this....
		#exp_reward = np.array([exp_rewards[actions.astype(int)[i],i] for i in range(n)])
		exp_reward = exp_rewards[actions.astype(int), np.arange(n)]

		regret = np.max(exp_rewards,axis=0) - exp_reward
		obs = exp_reward + etas
		return obs, regret, exp_reward    

class IHDPParams(object):
	def __init__(self, fn_in, batch = 0, shuffle = True):
		data = np.load(fn_in)
		n, b = data['mu1'].shape
		d = data['x'].shape[1]
		t = data['t']
		self.batch = batch
		self.ctx = data['x']
		self.expt_reward = np.zeros((n, 2, b))
		self.expt_reward[:,0,:] = data['mu0']
		self.expt_reward[:,1,:] = data['mu1']
		self.y = np.zeros((n, 2, b))
		self.y[:,0,:] = np.where((t == 0), data['yf'], data['ycf'])
		self.y[:,1,:] = np.where((t == 1), data['yf'], data['ycf'])
		self.pull = 0
		self.n = n
		self.d = d
		self.k = 2
		if shuffle:
			sh = np.arange(n)
			rand.shuffle(sh)
			self.ctx = np.squeeze(self.ctx[sh,:,:])
			self.expt_reward = np.squeeze(self.expt_reward[sh,:,:])
			self.y = np.squeeze(self.y[sh,:,:])

class IHDPGenerator(object):
	def __init__(self, params):
		self.params = params
		self.pull_idx = 0

	def context(self):
		ctx = self.params.ctx[self.pull_idx,:,self.params.batch]
		self.pull_idx += 1
		self.pull_idx = self.pull_idx % self.params.n
		return ctx

	def pull(self, ctx, a):
		exp_rewards = self.params.expt_reward[self.pull_idx,:,self.params.batch]
		exp_reward_a = exp_rewards[a]
		regret = max(exp_rewards) - exp_reward_a
		obs = self.params.y[self.pull_idx,a,self.params.batch]
		return obs, regret, exp_reward_a

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
