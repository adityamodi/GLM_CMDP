import numpy as np
import cvxpy as cp

import envs
import cmdp
import gl_orl
import gl_rlsvi
import link_fxn as lf


d = 5
ac = 5
ns = 10
hz = 6
env = envs.random_lincmdp(ns,ac,d,hz)


def rand_policy(nState, horizon, nAction):
	pol = {}
	for h in range(horizon):
		for s in range(nState):
			pol[s,h] = np.random.randint(nAction)
	return pol

# x1 = np.random.dirichlet(0.35*np.ones(d))
# _, _, pol1 = env.compute_Opt(x1)
# x2 = np.random.dirichlet(0.35*np.ones(d))
# _, _, pol2 = env.compute_Opt(x2)
# x3 = np.random.dirichlet(0.35*np.ones(d))
# _, _, pol3 = env.compute_Opt(x3)
# np.random.seed.reseed(12345)
queue = []
avg_regret = 0
learner = gl_orl.GLORL(ns, ac, hz, d, lf.linear_prob(ns,d), lf.rewardFxn(d))
# learner = gl_rlsvi.GL_RLSVI(ns, ac, hz, d, lf.linear_prob(ns,d), lf.rewardFxn(d))
for i in range(5000000):
	env.reset()
	x = np.random.dirichlet(0.35*np.ones(d))
	_, qmax, _ = env.compute_Opt(x)
	# _, vfx1 = env.eval_policy(x, pol1)
	# _, vfx2 = env.eval_policy(x, pol2)
	# _, vfx3 = env.eval_policy(x, pol3)
	learner.opt_plan(x)
	_, vfx = env.eval_policy(x, learner.policy)
	for h in range(hz):
		curr_s = env.state
		curr_a = learner.policy[curr_s,h]
		next_s, r, _ = env.step(x, curr_a)
		learner.update_obs(x, curr_s, curr_a, next_s, r)
	if len(queue) > 500:
		queue.pop(0)
		queue.append(qmax[0][0]-vfx[0][0])
	else:
		queue.append(qmax[0][0]-vfx[0][0])
	avg_regret = np.average(queue)
	if i%200 == 0:
		print("Round: ",i , " Opt_value: ", qmax[0][0], " Policy_values: ", vfx[0][0], " Avg. regret: ", avg_regret)
	# print("Opt_value: ", qmax[0][0], "\tPolicy_values: ", vfx1[0][0], '\t', vfx2[0][0], '\t', vfx3[0][0])


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

# for i in range(15):
# 	print("Iteration", i)
# 	# x = np.random.rand(d)/np.sqrt(d)
# 	x = np.random.dirichlet(np.ones(d))
# 	policy = {}
# 	for h in range(10):
# 		for s in range(ns):
# 			policy[s,h] = np.random.choice(a)

# 	q, qmax = curr.compute_Opt(x)
# 	q_rand, v_rand = curr.eval_policy(x, policy)
# 	print(q[s,0])
# 	print(q_rand[s,0])
# pFxn = lf.multi_logit(ns, d)
# rFxn = lf.rewardFxn(d)
# agent = gl_orl.GLORL(ns, ac, hz, d, pFxn, rFxn)
# val = []
# ag_val = []
# for i in range(15000):
# 	agent.reset()
# 	x = np.random.dirichlet(np.ones(d))
# 	agent.opt_plan(x)
# 	tot_r = 0
# 	q, qmax = curr.compute_Opt(x)
# 	val.append(q[0,0])
# 	for h in range(hz):
# 		s = curr.state
# 		a = agent(s)
# 		next_s, r, _ = curr.step(x, a)
# 		tot_r += r
# 		agent.update_obs(x,s,a,next_s, r)
# 	ag_val.append(tot_r)
# 	print(val[-1], ag_val[-1])

# print(np.sum(val), np.sum(ag_val))