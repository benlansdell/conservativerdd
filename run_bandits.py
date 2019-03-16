from lib.bandits import LinUCB, ThresholdBandit, GreedyBandit 
from lib.generator import LinearGeneratorParams, LinearGenerator
import numpy as np 
from scipy.stats import truncnorm

M = 30    #number of runs
N = 10000 #number of timesteps
#alg = 'greedy'
#alg = 'linucb'
alg = 'threshold'

save = True

max_alpha = 2
max_beta = 2

k = 4    #Number of arms
d = 10   #Dimension of context (includes one dim for intercept term)
intercept = True

if alg == 'greedy':
	BanditAlg = GreedyBandit
elif alg == 'linucb':
	BanditAlg = LinUCB
elif alg == 'threshold':
	BanditAlg = ThresholdBandit

#Generate slopes and intercepts
alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,k,d-1))
betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))

regret = np.zeros((M, N))
arm_pulls = np.zeros((M, N, k))
n_changes = np.zeros((M, N))

print("Running %s algorithm"%alg)
fn_out = './runs/bandit_%s_M_%d_N_%d.npz'%(alg, M, N)

for m in range(M):
	params = LinearGeneratorParams(np.atleast_2d(alphas[m,:,:]), betas[m,:], d = d, k = k, intercept = intercept)
	generator = LinearGenerator(params)
	bandit = BanditAlg(generator)
	
	print("Run: %d/%d"%(m+1,M))
	for i in range(N):
		(ctx, arm_idx, obs, r) = bandit.step()
		#print((arm_idx, obs, r))
		regret[m,i] = r
		arm_pulls[m,i,arm_idx] = 1

if save:
	np.savez(fn_out, regret = regret, arm_pulls = arm_pulls)