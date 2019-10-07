from cmdp import ContextualMDP as CMDP
import cvxpy as cp
import numpy as np
import link_fxn as lf

def ctxt_riverswim():
	'''Context of dimension 2
		Three actions intead of 2
		Good action and bad action
		There are two paths which give +1 (states 1-4) and -1 (states 5-8)
		Bad action takes you to path
		Good action takes to to good path
		Two MDPs with the index of the two actions switched
		0. is the back action
		1 is good action
		2 is good action
	'''
	nState = 9
	nAction = 3
	horizon = 20

	Wp = {}
	wr = {}

	for s in range(nState):
		wr[s,0] = np.zeros(2)
		Wp[s,0] = np.zeros((9,2))
		Wp[max(s-1,0),:] = np.ones(2)
		if s == 0:
			wr[s,0] = 0.005*np.ones(2)
			wr[s,1] = np.zeros(2)
			wr[s,2] = np.zeros(2)
			Wp[s,1][0,]
		elif s == 4:
			wr[s,1] = np.ones(2)
			wr[s,2] = np.ones(2)
		elif s == 8:
			wr[s,1] = -1*np.ones(2)
			wr[s,2] = -1*np.ones(2)
		else:
			wr[s,1] = np.zeros(2)
			wr[s,2] = np.zeros(2)


def random_cmdp(nState = 10, nAction = 5, xDim = 5, horizon = 10):
	norm_bnd = {}
	norm_bnd[10,5] = 5
	norm_bnd[10,3] = 4
	norm_bnd[20,5] = 6.5
	norm_bnd[20,3] = 5
	Wp = {}
	wr = {}

	for s in range(nState):
		for a in range(nAction):
			Wp[s,a] = 10*(2*np.random.rand(nState, xDim) - np.ones((nState, xDim)))/3
			wr[s,a] = np.random.rand(xDim)
			if cp.norm(wr[s,a]).value >= 1.0:
				wr[s,a] = wr[s,a]/cp.norm(wr[s,a]).value

	pFxn = lf.multi_logit(nState, xDim, 10)
	rFxn = lf.rewardFxn(xDim)
	cmdp = CMDP(nState, nAction, horizon, xDim, pFxn, rFxn)
	cmdp.Wp = Wp
	cmdp.wr = wr
	return cmdp
