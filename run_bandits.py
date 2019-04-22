from lib.bandits import LinUCB, ThresholdBandit, ThresholdConsBandit, GreedyBandit, ConsLinUCB, expected_regret
from lib.generator import LinearGeneratorParams, LinearGenerator
import numpy as np 
from scipy.stats import truncnorm

#alg = 'greedy'
#alg = 'linucb'
#alg = 'threshold'
#alg = 'thresholdcons'
alg = 'conslinucb'

M = 30    #number of runs
N = 10000 #number of timesteps
save = False
max_alpha = 2
max_beta = 2
k = 4    #Number of arms
d = 5   #Dimension of context (includes one dim for intercept term)
intercept = True
evaluate_every = 100

if alg == 'greedy':
	BanditAlg = GreedyBandit
elif alg == 'linucb':
	BanditAlg = LinUCB
elif alg == 'threshold':
	BanditAlg = ThresholdBandit
elif alg == 'thresholdcons':
	BanditAlg = ThresholdConsBandit
elif alg == 'conslinucb':
	BanditAlg = ConsLinUCB
else:
	print "Select a valid algorithm"

#Generate slopes and intercepts
alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,k,d-1))
betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))

baseline_alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,1,d-1))
baseline_betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,1))

regret = np.zeros((M, N))
expt_regret = np.zeros((M, N))
arm_pulls = np.zeros((M, N, k+1))
n_changes = np.zeros((M, N))
update_pol = np.zeros((M, N))

print("Running %s algorithm"%alg)
fn_out = './runs/bandit_%s_M_%d_N_%d.npz'%(alg, M, N)

for m in range(M):
	params = LinearGeneratorParams(np.atleast_2d(alphas[m,:,:]), betas[m,:], d = d, k = k, intercept = intercept)
	generator = LinearGenerator(params)
	bandit = BanditAlg(generator, (np.squeeze(baseline_alphas[m,:]), baseline_betas[m,0]))
	print("Run: %d/%d"%(m+1,M))
	for i in range(N):
		(ctx, arm_idx, obs, r) = bandit.step()
		#print((arm_idx, obs, r))
		regret[m,i] = r
		if arm_idx >= 0:
			arm_pulls[m,i,arm_idx] = 1
		else:
			arm_pulls[m,i,k] = 1			
		#if i % evaluate_every == 1:
		#	expt_reg[m,i] = expected_regret(bandit, generator)
		#if bandit.update_theta[bandit.pull-1] == 1:
		#	ereg = expected_regret(bandit, generator)
		#	print("Expected regret is now: %f"%ereg)
		#	expt_regret[m,i] = ereg
	#update_pol[m,:] = bandit.update_theta

if save:
	np.savez(fn_out, regret = regret, arm_pulls = arm_pulls, update_pol = update_pol, expt_reg = expt_reg)