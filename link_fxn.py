import numpy as np
import cvxpy as cp

class link_fxn(object):
	def __init__(self, sDim, xDim, B=0):
		self.xDim = xDim
		self.sDim = sDim
		self.B = B

	def sample_state(self, x, W):
		return np.random.choice(self.sDim, p=self.prob(x,W))

	def prob(self, x, W):
		return 0

class multi_logit(link_fxn):
	def prob(self, x, W):
		z = np.dot(W,x)
		ex = np.exp(z - np.max(z))
		return ex/ex.sum(axis=0)

	def update(self, Wt, x, y, Z, eta):
		'''
		Set up the cvxopt problem for ONS step
		'''
		W = cp.Variable((self.sDim, self.xDim))
		constraints = [cp.norm(W, "fro") <= self.B]
		p = self.prob(x, W)
		a = (p-y).reshape(self.sDim,1)
		x = x.reshape(self.xDim,1)
		prob = cp.Problem(cp.Minimize(0.5*cp.trace((W-Wt)@Z@(W.T-Wt.T)) + eta*cp.trace((a@x.T).T@(W-Wt))), constraints)
		_ = prob.solve()
		return W.value


class linear_prob(link_fxn):
	def prob(self, x, W):
		assert np.min(x) >= 0 and np.max(x) <= 1 and np.sum(x) == 1
		assert np.min(W) >= 0 and np.max(W) <= 1
		assert np.max(W.sum(axis=0)) == 1 and np.min(W.sum(axis=0)) == 1
		return np.dot(W,x)

	def update(self, Wt, x, y, Z, eta):
		'''
		Set up the cvxopt problem for ONS step
		'''
		W = cp.Variable((self.sDim, self.xDim))
		constraints = []
		for j in range(self.xDim):
			constraints += [cp.sum(W[:,j]) == 1.0]
			for i in range(self.sDim):
				constraints += [W[i,j] >= 0]
		p = self.prob(x, W)
		a = (p-y).reshape(self.sDim,1)
		x = x.reshape(self.xDim,1)
		prob = cp.Problem(cp.Minimize(0.5*cp.trace((W-Wt)@Z@(W.T-Wt.T)) + eta*cp.trace((a@x.T).T@(W-Wt))), constraints)
		_ = prob.solve()
		return W.value

class rewardFxn(object):
	def __init__(self, xDim, rtype = "bernoulli"):
		self.xDim = xDim
		self.rtype = rtype

	def sample_reward(self, x, w):
		if self.rtype == 'bernoulli':
			mu = self.mean(x,w)
			return np.random.bernoulli(1,mu)
		elif self.rtype == 'continuous':
			mu = self.mean(x,w)
			return mu + (2*np.random.bernoulli(1,0.5)-1)*np.random.rand()*min(mu,1-mu)

	def mean(self, x, w):
		return np.dot(x,w)

	def update(self, wt, x, r, Z, eta):
		'''
		Set up the cvxopt problem for ONS step
		'''
		W = cp.Variable(self.xDim)
		constraints = [cp.norm(W) <= 1]
		p = self.prob(x, W)
		x = x.reshape(self.xDim,1)
		prob = cp.Problem(cp.Minimize(0.5*cp.trace((W-Wt)@Z@(W.T-Wt.T)) + eta*cp.trace((a@x.T).T@(W-Wt))), constraints)
		_ = prob.solve()
		return W.value