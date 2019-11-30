#!/usr/bin/env python
from lib.bandits import LinUCB, ThresholdBandit, ThresholdConsBandit, GreedyBandit, ThresholdMaxConsBandit, ThresholdMaxConsGreedyBandit
from lib.generator import LinearGeneratorParams, LinearGenerator
import numpy as np 
from scipy.stats import truncnorm
import argparse

def get_args():
	argparser = argparse.ArgumentParser(description=__doc__)
	argparser.add_argument(
		'-a', '--algorithm',
		metavar='A',
		default='None',
		help='Algorithm')
	args = argparser.parse_args()
	return args

def main():

	args = get_args()
	alg = args.algorithm

	M = 30    #number of runs
	N = 10000 #number of timesteps
	
	save = True
	
	max_alpha = 2
	max_beta = 2
	
	O = 4
	del_angle = np.logspace(-4, -1, O)

	ks = [2]         #Number of arms
	ds = [4]   #Dimension of context (includes one dim for intercept term)
	intercept = True
	
	if alg == 'greedy':
		BanditAlg = GreedyBandit
	elif alg == 'linucb':
		BanditAlg = LinUCB
	elif alg == 'threshold':
		BanditAlg = ThresholdBandit
	elif alg == 'thresholdcons':
		BanditAlg = ThresholdConsBandit
	elif alg == 'thresholdmaxcons':
		BanditAlg = ThresholdMaxConsBandit
	elif alg == 'thresholdmaxconsgreedy':
		BanditAlg = ThresholdMaxConsGreedyBandit
	else:
		print "Select a valid algorithm"
	
	for i, k in enumerate(ks):
		for j, d in enumerate(ds):
			for idx_d, delta in enumerate(del_angle):
				#Generate slopes and intercepts
				alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,k,d-1))
				betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))
				
				regret = np.zeros((M, N))
				arm_pulls = np.zeros((M, N, k))
				n_changes = np.zeros((M, N))
				update_pol = np.zeros((M, N))
				
				print("Running %s algorithm with d=%d, k=%d"%(alg, d, k))
				fn_out = './runs/bandit_%s_d_%d_k_%d_M_%d_N_%d_delta_%f.npz'%(alg, d, k, M, N, delta)
				
				for m in range(M):
					params = LinearGeneratorParams(np.atleast_2d(alphas[m,:,:]), betas[m,:], d = d, k = k, intercept = intercept)
					generator = LinearGenerator(params)
					bandit = BanditAlg(generator, Delta_angle = delta)
					
					print("Run: %d/%d"%(m+1,M))
					for n in range(N):
						(ctx, arm_idx, obs, r) = bandit.step()
						#print((arm_idx, obs, r))
						regret[m,n] = r
						arm_pulls[m,n,arm_idx] = 1
					if hasattr(bandit, 'update_theta'):
						update_pol[m,:] = bandit.update_theta
				
				if save:
					np.savez(fn_out, regret = regret, arm_pulls = arm_pulls, update_pol = update_pol)

if __name__ == "__main__":
	main()
