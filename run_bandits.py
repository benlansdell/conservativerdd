from lib.bandits import LinUCB, ThresholdBandit 
from lib.generator import LinearGeneratorParams, LinearGenerator

alphas = (0, 2)
betas = (1, 0)
threshold = 0.5

params = LinearGeneratorParams(alphas, betas)
generator = LinearGenerator(params)
linucb = LinUCB(generator)
thresholdbandit = ThresholdBandit(generator, threshold)