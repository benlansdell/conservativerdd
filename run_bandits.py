from lib.bandits import LinUCB, ThresholdBandit, GreedyBandit 
from lib.generator import LinearGeneratorParams, LinearGenerator
import numpy as np 
from scipy.stats import truncnorm

M = 30
N = 10000
alg = 'greedy'

max_alpha = 2
max_beta = 2

#Generate slopes
alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,2))
#Generate intercepts
betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,2))

regret = np.zeros((M, N))
arm_pulls = np.zeros((M, N, 2))

print("Running %s algorithm"%alg)
fn_out = './runs/bandit_%s_M_%d_N_%d.npz'%(alg, M, N)

for m in range(M):
	params = LinearGeneratorParams(alphas[m,:], betas[m,:])
	generator = LinearGenerator(params)
	if alg == 'greedy':
		bandit = GreedyBandit(generator)
	elif alg == 'linucb':
		bandit = LinUCB(generator)
	elif alg == 'threshold':
		bandit = ThresholdBandit(generator)
	
	print("Run: %d/%d"%(m+1,M))
	for i in range(N):
		(ctx, arm_idx, obs, r) = bandit.step()
		#print((ctx, arm_idx, obs, regret))
		regret[m,i] = r
		arm_pulls[m,i,arm_idx] = 1

np.savez(fn_out, regret = regret, arm_pulls = arm_pulls)