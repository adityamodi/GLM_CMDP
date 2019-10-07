import numpy as np

class ContextualMDP(object):
	'''Uses a contextual mapping to define a parametrized MDP'''

	def __init__(self, nState, nAction, horizon, xDim, pFxn, rFxn):
		'''
		Initialize the contextual tabular MDP

		Args:
			nState 	: 	number of states
			nAction : 	number of actions
			horizon	:	Episode Length
			xDim	:	Dimension of context
			pFxn	:	Link function for next state distributions
			rFxn	:	Reward Fxn
		'''

		self.nState = nState
		self.nAction = nAction
		self.horizon = horizon
		self.xDim = xDim
		self.pFxn = pFxn
		self.rFxn = rFxn
		self.t = 0
		self.state = 0

		# MDP parameters
		self.Wp = {}
		self.wr = {}
		for s in range(nState):
			for a in range(nAction):
				self.Wp[s,a] = np.zeros((nState, xDim))
				self.wr[s,a] = np.zeros((xDim))

	def setP(self, s, a, W):
		assert W.shape == (nState, xDim)
		self.Wp[s,a] = W

	def setR(self, s, a, w):
		assert w.shape == (xDim,)
		self.wr[s,a] = w

	def reset(self):
		self.t = 0
		self.state = 0

	def step(self, a):
		'''
		Move the environment one step ahead using the given action
		'''

		reward = self.rFxn.sample_reward(self.X, self.wr[self.state, a])
		newState = self.pFxn.sample_state(self.X, self.Wp[self.state, a])

		self.state = newState
		self.t += 1

		if self.t == self.horizon:
			pContinue = 0
			self.reset()
		else:
			pContinue = 1

		return newState, reward, pContinue

	def compute_Opt(self, ctxt):
		'''
		Computes the Q-values for each state action pair for the current context and the optimal policy as well
		'''
		qVal = {}
		qMax = {}

		qMax[self.horizon] = np.zeros(self.nState)

		for i in range(self.horizon):
			j = self.horizon - i - 1
			qMax[j] = np.zeros(self.nState)
			for s in range(self.nState):
				qVal[s,j] = np.zeros(self.nAction)
				for a in range(self.nAction):
					qVal[s,j][a] = self.rFxn.mean(ctxt, self.wr[s,a]) + np.dot(self.pFxn.prob(ctxt, self.Wp[s,a]), qMax[j+1])
				qMax[j][s] = np.max(qVal[s,j])
		return qVal, qMax

	def eval_policy(self, ctxt, policy):
		'''
		Computes the Q-values for each state action pair for the current context with given policy
		'''
		qVal = {}
		Vfx = {}

		Vfx[self.horizon] = np.zeros(self.nState)

		for i in range(self.horizon):
			j = self.horizon - i - 1
			Vfx[j] = np.zeros(self.nState)
			for s in range(self.nState):
				qVal[s,j] = np.zeros(self.nAction)
				for a in range(self.nAction):
					qVal[s,j][a] = self.rFxn.mean(ctxt, self.wr[s,a]) + np.dot(self.pFxn.prob(ctxt, self.Wp[s,a]), Vfx[j+1])
				Vfx[j][s] = qVal[s,j][policy[s,j]]
		return qVal, Vfx







