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
	def __init__(self, sDim, xDim, B=0):
		self.xDim = xDim
		self.sDim = sDim
		self.B = B
		self.beta = 1
		self.alpha = 1.0/(self.sDim**2)

	def prob(self, x, W):
		z = np.dot(W,x)
		ex = np.exp(z - np.max(z))
		return ex/ex.sum(axis=0)

	def beta(self):
		return 1

	def alpha(self):
		return 1/(self.sDim**2)

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
	def __init__(self, sDim, xDim, B=0):
		self.xDim = xDim
		self.sDim = sDim
		self.B = B
		self.beta = 1
		self.alpha = 1.0

	def prob(self, x, W):
		# print(W)
		# print(np.max(W.sum(axis=0)))
		# print(np.min(W.sum(axis=0)))
		# print(np.sum(x))
		# assert np.min(x) >= 0 and np.max(x) <= 1.0 and np.sum(x) == 1.0
		# assert np.min(W) >= 0 and np.max(W) <= 1
		# assert np.max(W.sum(axis=0)) == 1.0 and np.min(W.sum(axis=0)) == 1.0
		return np.dot(W,x)

	def update(self, Wt, x, y, Z, eta):
		'''
		Set up the cvxopt problem for ONS step
		'''
		# print(Wt.shape,x.shape,y.shape,Z.shape)
		W = cp.Variable((self.sDim, self.xDim))
		constraints = []
		for j in range(self.xDim):
			constraints += [cp.sum(W[:,j]) == 1.0]
			for i in range(self.sDim):
				constraints += [W[i,j] >= 0]
		p = self.prob(x, Wt)
		a = (p-y).reshape(self.sDim,1)
		x = x.reshape(self.xDim,1)
		# Zinv = np.linalg.inv(Z)
		# print(np.linalg.eigvals(Zinv))
		cost  = 0
		for s in range(self.sDim):
			cost += cp.quad_form(W[s,:]-Wt[s,:], Z)
		prob = cp.Problem(cp.Minimize(0.5*cost + eta*cp.trace((a@x.T).T@(W-Wt))), constraints)
		# prob = cp.Problem(cp.Minimize(0.5*cp.trace(((W-Wt)@Zinv)@((W-Wt).T)) + eta*cp.trace((a@x.T).T@(W-Wt))), constraints)
		_ = prob.solve()
		return W.value

class rewardFxn(object):
	def __init__(self, xDim, rtype = "continuous"):
		self.xDim = xDim
		self.rtype = rtype

	def sample_reward(self, x, w):
		if self.rtype == 'bernoulli':
			mu = self.mean(x,w)
			return np.random.binomial(1,mu)
		elif self.rtype == 'continuous':
			mu = self.mean(x,w)
			return mu + (2*np.random.binomial(1,0.5)-1)*np.random.rand()*min(mu,1-mu)

	def mean(self, x, w):
		return np.dot(x,w)

	def update(self, wt, x, r, Z, eta):
		'''
		Set up the cvxopt problem for ONS step
		'''
		w = cp.Variable(self.xDim)
		constraints = []
		for d in range(self.xDim):
			constraints += [w[d] <= 1]
			constraints += [w[d] >= 0]
		p = self.mean(x, wt)
		x = x.reshape(self.xDim,1)
		prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(w-wt, Z) + eta*(p-r)*x.T*(w-wt)), constraints)
		_ = prob.solve()
		return w.value