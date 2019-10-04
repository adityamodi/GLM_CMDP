import numpy as np

class link_fxn(object):
	def __init__(self, sDim, xDim):
		self.xDim = xDim
		self.sDim = sDim

	def sample_state(self, x, W):
		return np.random.choice(self.sDim, p=self.prob(x,W))

	def prob(self, x, W):
		return 0

class multi_logit(link_fxn):
	def prob(self, x, W):
		z = np.dot(W,x)
		ex = np.exp(z - np.max(z))
		return ex/ex.sum(axis=0)

class linear_prob(link_fxn):
	def prob(self, x, W):
		assert np.min(x) >= 0 and np.max(x) <= 1 and np.sum(x) == 1
		assert np.min(W) >= 0 and np.max(W) <= 1
		assert np.max(W.sum(axis=0)) == 1 and np.min(W.sum(axis=0)) == 1
		return np.dot(W,x)

class rewardFxn(object):
	def __init__(self, rtype, xDim):
		self.xDim = xDim
		self.rtype = rtype

	def sample_reward(self, x, w):
		return a

	def mean(self, x, w):
		if self.rtype == 'bernoulli':
			mu = np.dot(x,w)
			return np.random.bernoulli(1,mu)
		elif self.rtype == 'continuous':
			mu = np.dot(x,w)
			return mu + (2*np.random.bernoulli(1,0.5)-1)*np.random.rand()*min(mu,1-mu)