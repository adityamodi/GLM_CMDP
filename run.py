import numpy as np
import cvxpy as cp

import envs
import cmdp
import gl_orl

curr = envs.random_cmdp(10,5,5,10)

x = np.random.rand(5)/np.sqrt(5)
# if cp.norm(x).value >= 1.0:
	# x = x/cp.norm(x).value
# x[0] = 1
# for s in range(4):
# 	for a in range(2):
# 		p = curr.pFxn.prob(x,curr.Wp[s,a])
# 		print(p)
# 		print(curr.pFxn.sample_state(x, curr.Wp[s,a]))
# 		r = curr.rFxn.mean(x, curr.wr[s,a])
# 		print(r)

policy = {}
for h in range(10):
	for s in range(10):
		policy[s,h] = np.random.choice(5)

q, qmax = curr.compute_Opt(x)
q_rand, v_rand = curr.eval_policy(x, policy)
for h in range(10):
	for s in range(10):
		print(q[s,h])

print("Random policy:")

for h in range(10):
	for s in range(10):
		print(q_rand[s,h])