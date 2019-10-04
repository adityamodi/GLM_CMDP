import numpy as np

class GLORL(object):
	def __init__(self, nState, nAction, horizon, xDim, pFxn, rFxn, plr, rlr, lbda):
		self.nState = nState
		self.nAction = nAction
		self.horizon = horizon
		self.xDim = xDim
		self.pFxn = pFxn
		self.rFxn = rFxn
		self.plr = plr
		self.rlr = rlr
		self.lbda = lbda
		self.wr = {}
		self.Wp = {}
		for s in range(nState):
			for a in range(nAction):
				self.Wp[s,a] = np.zeros((nState, xDim))
				self.wr[s,a] = np.zeros((xDim))
		self.policy = {}
		for h in range(horizon):
			for s in range(nState):
				self.policy[s,h] = np.random.randint(nAction)
		self.Z = {}
		for s in range(nState):
			for a in range(nAction):
				self.Z[s,a]  = self.lbda * np.eye(xDim)


	def update_obs(self, ctxt, s, a, s_nxt, r):
		'''
		Updates the parameters for the given state-action pair with transition and reward
			- Update Z for (s,a)
			- Update Wp[s,a]
			- Update wr[s,a]
		'''
		self.Wp[s,a] = A
		self.wr[s,a] = b

	def opt_plan(self, ctxt):
		qVal = {}
		qMax = {}

		qMax[self.horizon] = np.zeros(self.nState)

		for i in range(self.horizon):
			j = self.horizon - i - 1
			qMax[j] = np.zeros(self.nState)
			for s in range(self.nState):
				qVal[s,j] = np.zeros(self.nAction)
				for a in range(self.nAction):
					qVal[s,j][a] = self.rFxn.mean(self.X, self.wr[s,a]) + np.dot(self.pFxn.prob(self.X, self.Wp[s,a]), qMax[j+1])
				qMax[j][s] = np.max(qVal[s,j])
		return qVal, qMax