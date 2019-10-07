import numpy as np

class GLORL(object):
	def __init__(self, nState, nAction, horizon, xDim, pFxn, rFxn, plr=0.01, rlr=0.01, lbda=0.1, pScale=0.1):
		self.nState = nState
		self.nAction = nAction
		self.horizon = horizon
		self.xDim = xDim
		self.pFxn = pFxn
		self.rFxn = rFxn
		self.plr = plr
		self.rlr = rlr
		self.lbda = lbda
		self.pScale = pScale
		self.wr = {}
		self.Wp = {}
		self.t = 0
		for s in range(nState):
			for a in range(nAction):
				self.Wp[s,a] = np.zeros((nState, xDim))
				self.wr[s,a] = np.zeros((xDim))
		self.policy = {}
		for h in range(horizon):
			for s in range(nState):
				self.policy[s,h] = np.random.randint(nAction)
		self.Z_inv = {}
		self.Z = {}
		self.idx = {}
		self.tot_potential = {}
		for s in range(nState):
			for a in range(nAction):
				self.Z_inv[s,a]  = 1.0/self.lbda * np.eye(xDim)
				self.Z[s,a] = self.lbda * np.eye(xDim)
				self.idx[s,a] = 0
				self.tot_potential[s,a] = 0


	def update_obs(self, ctxt, s, a, s_nxt, r):
		'''
		Updates the parameters for the given state-action pair with transition and reward
			- Update Z for (s,a)
			- Update Wp[s,a]
			- Update wr[s,a]
		'''
		self.Wp[s,a] = A
		self.wr[s,a] = b
		self.idx[s,a] += 1
		# Update Z[s,a] and Z_inv[s,a]
		mult = self.plr * self.pFxn.alpha() / 2.0
		x = ctxt.reshape(self.xDim, 1)
		self.Z[s,a] = self.Z[s,a] + mult * np.dot(x, x.T)
		self.Z_inv[s,a] = self.Z_inv[s,a] - np.dot(np.dot(self.Z_inv[s,a], mult*x), np.dot(x.T, self.Z_inv[s,a]))/ (1+np.sqrt(np.dot(mult*x.T, np.dot(self.Z_inv[s,a], x))))
		# Update the parameter with ONS step
		y = np.zeros(self.nState)
		y[s_nxt] = 1
		self.pFxn.update(self.Wp[s,a], ctxt, y, self.Z_inv[s,a], self.plr)
		self.rFxn.update(self.wr[s,a], ctxt, r, self.Z_inv[s,a], self.rlr)
		# Update tot_potential
		self.tot_potential[s,a] += np.dot(ctxt, np.dot(self.Z_inv[s,a], ctxt))

	def opt_plan(self, ctxt):
		pFactor = self.pFxn.beta() * self.pScale
		pBonus = {}
		rBonus = {}
		for s in range(self.nState):
			for a in range(self.nAction):
				gamma = self.lbda*10 + 4*self.plr*10*np.log(40*np.log(self.idx[s,a])*self.idx[s,a]*self.idx[s,a])/self.pFxn.alpha() + \
				8*self.tot_potential[s,a]/self.pFxn.alpha()
				potential = np.sqrt(np.dot(ctxt, np.dot(self.Z_inv[s,a], ctxt)))
				pBonus[s,a] = pFactor * np.sqrt(gamma) * potential
		qVal = {}
		qMax = {}

		qMax[self.horizon] = np.zeros(self.nState)

		for i in range(self.horizon):
			j = self.horizon - i - 1
			qMax[j] = np.zeros(self.nState)
			for s in range(self.nState):
				qVal[s,j] = np.zeros(self.nAction)
				for a in range(self.nAction):
					est = self.rFxn.mean(self.X, self.wr[s,a]) + np.dot(self.pFxn.prob(self.X, self.Wp[s,a]), qMax[j+1])
					qVal[s,j][a] = np.max(0, np.max(est+bonus[s,a], self.horizon - i))
				qMax[j][s] = np.max(qVal[s,j])
				self.policy[s,j] = np.argmax(qVal[s,j])

	def reset(self):
		self.t = 0

	def __call__(self, ctxt, state):
		if self.t == 0:
			self.opt_plan(ctxt)
		a = self.policy[state, self.t]
		self.t += 1
		return a