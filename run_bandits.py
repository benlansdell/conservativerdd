from lib.bandits import LinUCB, ThresholdBandit 
from lib.generator import LinearGeneratorParams, LinearGenerator
import numpy as np 

alphas = (0, 2)
betas = (1, 0)
threshold = 0.5

params = LinearGeneratorParams(alphas, betas)
generator = LinearGenerator(params)
linucb = LinUCB(generator)
#thresholdbandit = ThresholdBandit(generator, threshold)

#Step through a bunch of times
for i in range(2000):
	(ctx, arm_idx, obs, regret) = linucb.step()
	print((ctx, arm_idx, obs, regret))

theta = np.dot(np.linalg.inv(linucb.V), linucb.U)
print(theta)