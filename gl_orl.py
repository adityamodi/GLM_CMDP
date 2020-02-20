import numpy as np

class GLORL(object):
	def __init__(self, nState, nAction, horizon, xDim, pFxn, rFxn, plr=1, rlr=1, lbda=0.1, pScale=0.1):
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
				for d in range(xDim):
					self.Wp[s,a][:,d] = np.random.dirichlet(0.4*np.ones(nState))
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
				self.idx[s,a] = 1
				self.tot_potential[s,a] = 0


	def update_obs(self, ctxt, s, a, s_nxt, r):
		'''
		Updates the parameters for the given state-action pair with transition and reward
			- Update Z for (s,a)
			- Update Wp[s,a]
			- Update wr[s,a]
		'''
		self.idx[s,a] += 1
		# Update Z[s,a] and Z_inv[s,a]
		mult = self.plr * self.pFxn.alpha / 2.0
		x = ctxt.reshape(self.xDim, 1)
		self.Z[s,a] = self.Z[s,a] + mult * np.dot(x, x.T)
		nZ_inv = self.Z_inv[s,a] - np.dot(np.dot(self.Z_inv[s,a], mult*x), np.dot(x.T, self.Z_inv[s,a]))/ (1+\
			np.sqrt(np.dot(mult*x.T, np.dot(self.Z_inv[s,a], x))))
		self.Z_inv[s,a] = nZ_inv
		# self.Z_inv[s,a] = np.linalg.inv(self.Z[s,a])
		# Update the parameter with ONS step
		y = np.zeros(self.nState)
		y[s_nxt] = 1
		self.Wp[s,a] = self.pFxn.update(self.Wp[s,a], ctxt, y, self.Z[s,a], self.plr)
		self.wr[s,a] = self.rFxn.update(self.wr[s,a], ctxt, r, self.Z[s,a], self.rlr)
		# Update tot_potential
		self.tot_potential[s,a] = np.log(np.linalg.det(self.Z[s,a])) - self.xDim*np.log(self.lbda)

	def opt_plan(self, ctxt):
		pFactor = self.pFxn.beta * np.sqrt(self.nState)
		pBonus = {}
		rBonus = {}
		for s in range(self.nState):
			for a in range(self.nAction):
				t = self.idx[s,a]
				temp = np.log(t*t*np.log(self.nState*t)) + 4*self.tot_potential[s,a]
				# temp = 4*self.tot_potential[s,a]
				# gamma = self.lbda*self.xDim + 8*self.plr + 2*self.plr*temp
				gamma = 2*self.plr*temp
				potential = np.sqrt(np.dot(ctxt, np.dot(self.Z_inv[s,a], ctxt)))
				pBonus[s,a] = pFactor * np.sqrt(gamma) * potential
				rBonus[s,a] = np.sqrt(self.tot_potential[s,a])*potential
		qVal = {}
		qMax = {}

		qMax[self.horizon] = np.zeros(self.nState)

		for i in range(self.horizon):
			j = self.horizon - i - 1
			qMax[j] = np.zeros(self.nState)
			for s in range(self.nState):
				qVal[s,j] = np.zeros(self.nAction)
				# print(qVal[s,j])
				for a in range(self.nAction):
					est = self.rFxn.mean(ctxt, self.wr[s,a]) + np.dot(self.pFxn.prob(ctxt, self.Wp[s,a]), qMax[j+1])
					opt_est = est+(pBonus[s,a]*np.max(qMax[j+1])+rBonus[s,a]) * self.pScale
					qVal[s,j][a] = max(0, min(opt_est, self.horizon - j))
				qMax[j][s] = np.max(qVal[s,j])
				self.policy[s,j] = np.argmax(qVal[s,j])

	def reset(self):
		self.t = 0

	def __call__(self, state):
		# if self.t == 0:
			# self.opt_plan(ctxt)
		a = self.policy[state, self.t]
		self.t += 1
		return a