import numpy as np
import cvxpy as cp

import envs
import cmdp
import gl_orl
import gl_rlsvi
import link_fxn as lf

ns = 5
dim = 5
W = np.zeros((ns,dim))
for d in range(dim):
	W[:,d] = np.random.dirichlet(0.4*np.ones(ns))

phi = lf.linear_prob(ns,dim)

Wt = np.zeros((ns,dim))
for d in range(dim):
	Wt[:,d] = np.random.dirichlet(0.4*np.ones(ns))

Z = 0.1*np.eye(dim)
eta = 1

for i in range(50000):
	x = np.random.dirichlet(0.35*np.ones(dim))
	true_p = phi.prob(x, W)
	# true_p /= np.sum(true_p)
	# print(ns, true_p, np.sum(true_p))
	est_p = phi.prob(x, Wt)
	Z = Z + 0.5*eta*np.dot(x,x.T)
	snxt = np.random.choice(int(ns), p=true_p)
	y = np.zeros(ns)
	y[snxt] = 1.0
	Zinv = np.linalg.inv(Z)
	Wt = phi.update(Wt, x, y, Z, eta)
	potential = np.sqrt(np.dot(x, np.dot(Zinv, x)))
	print("Round: ", i, "\tCurrent distance: ", np.sum(np.abs(true_p-est_p)), '\tPotential: ', potential)