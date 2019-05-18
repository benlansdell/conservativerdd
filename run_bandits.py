from lib.bandits import LinUCB, ThresholdBandit, ThresholdConsBandit, ThresholdMaxConsBandit, ThresholdMaxConsGreedyBandit, \
				GreedyBandit, ConsLinUCB, RarelySwitchingLinUCB, ThresholdBaselineBandit, expected_regret, expected_regret_per_arm
from lib.generator import LinearGeneratorParams, LinearGenerator
import numpy as np
from scipy.stats import truncnorm

algs = ['greedy','linucb','threshold','thresholdcons','rarelyswitching','conslinucb', 'thresholdmaxcons', 'thresholdmaxconsgreedy']

for alg in algs:	
	M = 50    #number of runs
	N = 10000 #number of timesteps

	delta = 1./N
	#Test number of rounds
	#N = 100

	save = True
	max_alpha = 2
	max_beta = 2
	k = 4    #Number of arms
	#d = 5   #Dimension of context (includes one dim for intercept term)
	d = 2   #Dimension of context (includes one dim for intercept term)
	intercept = True
	evaluate_every = 100
	baseline_idx = 1
	
	N_expt_pulls_others = 500000
	N_expt_pulls_CLUCB = 10000
	N_expt_pulls = N_expt_pulls_CLUCB if alg == 'conslinucb' else N_expt_pulls_others
	
	random_evals = ['greedy', 'linucb', 'conslinucb']
	random_eval = alg in random_evals
	
	if alg == 'greedy':
		BanditAlg = GreedyBandit
	elif alg == 'linucb':
		BanditAlg = LinUCB
	elif alg == 'threshold': #The greedy version
		BanditAlg = ThresholdBandit
	elif alg == 'thresholdcons':
		BanditAlg = ThresholdConsBandit
	elif alg == 'thresholdmaxcons':
		BanditAlg = ThresholdMaxConsBandit
	elif alg == 'thresholdmaxconsgreedy':
		BanditAlg = ThresholdMaxConsGreedyBandit
	#elif alg == 'thresholdbaseline':
	#	BanditAlg = ThresholdBaselineBandit
	elif alg == 'rarelyswitching':
		BanditAlg = RarelySwitchingLinUCB
	elif alg == 'conslinucb':
		BanditAlg = ConsLinUCB
	else:
		print("Select a valid algorithm")
	
	#Generate slopes and intercepts
	alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,k,d-1))
	if alg == 'conslinucb':
		betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))+4
	else:
		betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))
	
	#baseline_alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,1,d-1))
	#baseline_betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,1))
	#baseline_betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,1))+2
	
	#if alg == 'conslinucb':
	#	#Generate slopes and intercepts
	#	alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,k,d-1))
	#	betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))+4
	#	baseline_alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,1,d-1))
	#	baseline_betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,1))+2
	#else:
	#	#How to compare baseline alphas and betas?
	#	alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,k,d-1))
	#	betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,k))+4
	#	baseline_alphas = truncnorm.rvs(-max_alpha, max_alpha, scale = 1, size=(M,1,d-1))
	#	baseline_betas = truncnorm.rvs(-max_beta, max_beta, scale = 1, size=(M,1))+2
	
	regret = np.zeros((M, N))
	expt_rewards = np.zeros((M, N))
	expt_regret = np.zeros((M, N))
	expt_rewards_baseline = np.zeros((M, N))
	arm_pulls = np.zeros((M, N, k+1))
	n_changes = np.zeros((M, N))
	update_pol = np.zeros((M, N))
	
	print("Running %s algorithm"%alg)
	fn_out = './runs/bandit_%s_M_%d_N_%d_with_d_%d_k_%d.npz'%(alg, M, N, d, k)
	
	for m in range(M):
		params = LinearGeneratorParams(np.atleast_2d(alphas[m,:,:]), betas[m,:], d = d, k = k, intercept = intercept)
		generator = LinearGenerator(params)
		#Choose the baseline arm as the worst arm....
		means = expected_regret_per_arm(generator)
	    sorted_means = np.sort(means)
    	m_idx = np.where(means == sorted_means[baseline_idx])[0][0]
		base_alpha = alphas[m,m_idx,:]
		base_beta = betas[m,m_idx]
	
		if alg == 'conslinucb':
			#bandit = BanditAlg(generator, (np.squeeze(baseline_alphas[m,:]), baseline_betas[m,0]), alpha = 0.1)
			bandit = BanditAlg(generator, (base_alpha, base_beta), alpha = 0.1, delta = delta)
		else:
			bandit = BanditAlg(generator, delta = delta)
		print("Run: %d/%d"%(m+1,M))
		for i in range(N):
			(ctx, arm_idx, obs, r) = bandit.step()
			regret[m,i] = r
			if arm_idx >= 0:
				arm_pulls[m,i,arm_idx] = 1
			else:
				arm_pulls[m,i,k] = 1		
				arm_idx  = m_idx
			if not random_eval:
				if bandit.update_theta[bandit.pull-1] == 1:
					#Either the policy updates rarely, and only compute expt regret for each update, 
					#Or the policy updates all the time, and we compare consecutive pairs of expt regret
					if not alg == 'conslinucb':
						ereg = expected_regret(bandit, generator, N_pulls = N_expt_pulls)
						print("Expected regret is now: %f"%ereg)
						expt_regret[m,i] = ereg
	
			#if (random_eval and i % evaluate_every == 1) or \
			#		(random_eval and i % evaluate_every == 0):
			#	#Either the policy updates rarely, and only compute expt regret for each update, 
			#	#Or the policy updates all the time, and we compare consecutive pairs of expt regret
			#	if not alg == 'conslinucb':
			#		ereg = expected_regret(bandit, generator, N_pulls = N_expt_pulls)
			#		print("Expected regret is now: %f"%ereg)
			#		expt_regret[m,i] = ereg
	
			#Compute baseline performance for comparison
			#expt_rewards_baseline[m,i] = np.dot(ctx[1:],np.squeeze(baseline_alphas[m,:])) + baseline_betas[m,0]
			expt_rewards_baseline[m,i] = np.dot(ctx[1:],base_alpha) + base_beta
			expt_rewards[m,i] = np.dot(ctx[1:], alphas[m,arm_idx,:]) + betas[m,arm_idx]
			#Can use this in analysis to decide if the algorithms are violating performance constraints, as investigated in the CLUCB paper
	
		if hasattr(bandit, 'update_theta'):
			update_pol[m,:] = bandit.update_theta[0:N]
	
	if save:
		np.savez(fn_out, regret = regret, arm_pulls = arm_pulls, update_pol = update_pol, \
			expt_regret = expt_regret, reward = expt_rewards, reward_baseline = expt_rewards_baseline)