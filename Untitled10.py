#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import scipy
import random

import matplotlib.pyplot as plt
    


def generate_reward_sample():
    #rewards for no-op are gamma distributed 
    r_noop = []
    locs = 1/2
    scales = [20, 40, 80,190]
    for i in range(4):
        r_noop.append(-np.random.gamma(locs, scales[i], 1)[0])
    r_noop = np.array(r_noop)
    
    #rewards for repair are -N(100,1) for all but last state where it is -N(130,20)
    r_repair = -100 + -1 * np.random.randn(4)

    return np.concatenate((r_noop, r_repair))


def generate_posterior_samples(num_samples):
    
    all_samples = []
    for i in range(num_samples):
        r_sample = generate_reward_sample()
        all_samples.append(r_sample)


    print("mean of posterior from samples")
    print(np.mean(all_samples, axis=0))


    posterior = np.array(all_samples)

    return posterior.transpose()  #each column is a reward sample


if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    num_states = 4
    num_samples = 2000

    gamma = 0.95
    alpha = 0.99
    lamda = 0.9

    posterior = generate_posterior_samples(num_samples)
    r_sa = np.mean(posterior, axis=1)

    init_distribution = np.ones(num_states)/num_states  #uniform distribution
    mdp_env = MachineReplacementMDP(num_states, r_sa, gamma, init_distribution)
    print("---MDP solution for expectation---")
    print("mean MDP reward", r_sa)

    u_sa = solve_mdp_lp(mdp_env, debug=True)
    print("mean policy from posterior")
    print_stochastic_policy_action_probs(u_sa, mdp_env)
    print("MAP/Mean policy from posterior")
    print_policy_from_occupancies(u_sa, mdp_env) 
    print("rewards")
    print(mdp_env.r_sa)
    print("expected value = ", np.dot(u_sa, r_sa))
    stoch_pi = get_optimal_policy_from_usa(u_sa, mdp_env)
    print("expected return", get_policy_expected_return(stoch_pi, mdp_env))
    print("values", get_state_values(u_sa, mdp_env))
    print('q-values', get_q_values(u_sa, mdp_env))

    
  
    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    
    # print("solving for CVaR optimal policy")
    posterior_probs = np.ones(num_samples) / num_samples  #uniform dist since samples from MCMC
    
    #generate efficient frontier
    lambda_range = [0.0, 0.3, 0.5, 0.75, 0.95,0.99, 1.0]

    #generate_efficient_frontier.calc_frontier(mdp_env, u_expert, posterior, posterior_probs, lambda_range, alpha, debug=False)
    alpha = 0.99
    
    print("calculating optimal policy for alpha = {} over lambda = {}".format(alpha, lambda_range))
    cvar_rets = calc_frontier(mdp_env, u_expert, posterior, posterior_probs, lambda_range, alpha, debug=False)
    
    cvar_rets_array = np.array(cvar_rets)
    plt.figure()
    plt.plot(cvar_rets_array[:,0], cvar_rets_array[:,1], '-o')

    #go through and label the points in the figure with the corresponding lambda values
    unique_pts_lambdas = []
    unique_pts = []
    for i,pt in enumerate(cvar_rets_array):
        unique = True
        for upt in unique_pts:
            if np.linalg.norm(upt - pt) < 0.00001:
                unique = False
                break
        if unique:
            unique_pts_lambdas.append((pt[0], pt[1], lambda_range[i]))
            unique_pts.append(np.array(pt))


    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18) 
    plt.xlabel("Robustness (CVaR)", fontsize=20)
    plt.ylabel("Expected Return", fontsize=20)
    
    plt.tight_layout()
    

    plt.show()


# In[1]:


import numpy as np
from scipy.optimize import linprog
from interface import implements, Interface
import sys

#acts as abstract class
class MDP(Interface):
    def get_num_actions(self):
        pass

    def get_reward_dimensionality(self):
        pass

    def set_reward_fn(self, new_reward):
        pass

    def get_transition_prob(self, s1,a,s2):
        pass

    def get_num_states(self):
        pass

    def get_readable_actions(self, action_num):
        pass

    def get_state_action_rewards(self):
        pass

    def uses_linear_approximation(self):
        pass

    def transform_to_R_sa(self, reward_weights):
        #mainly used for BIRL to take hypothesis reward and transform it
        #take in representation of reward weights and return vectorized version of R_sa
        #R_sa = [R(s0,a0), .., R(sn,a0), ...R(s0,am),..., R(sn,am)]
        pass

    def get_transition_prob_matrices(self):
        #return a list of transition matrices for each action a_0 through a_m
        pass

        


class ChainMDP(implements(MDP)):
    #basic MDP class that has two actions (left, right), no terminal states and is a chain mdp with deterministic transitions
    def __init__(self, num_states, r_sa, gamma, init_dist):
        self.num_actions = 2
        self.num_rows = 1
        self.num_cols = num_states
        self.num_states =  num_states
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = []
       
        self.r_sa = r_sa

        self.init_states = []
        for s in range(self.num_states):
            if self.init_dist[s] > 0:
                self.init_states.append(s)


        self.P_left = self.get_transitions(policy="left")
        #print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        #print("P_right\n",self.P_right)
        self.Ps = [self.P_left, self.P_right]

    def get_transition_prob_matrices(self):
        return self.Ps

    def get_num_actions(self):
        return self.num_actions

    def transform_to_R_sa(self, reward_weights):
        #Don't do anything, reward_weights should be r_sa 
        assert(len(reward_weights) == len(self.r_sa))
        return reward_weights

    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "<"
        elif action_num == 1:
            return ">"
        else:
            print("error, only two possible actions")
            sys.exit()

    def get_num_states(self):
        return self.num_states

    def get_reward_dimensionality(self):
        return len(self.r_sa)
    
    def set_reward_fn(self, new_reward):
        self.r_sa = new_reward

    def get_state_action_rewards(self):
        return self.r_sa

    def get_transition_prob(self, s1,a,s2):
        return self.Ps[a][s1][s2]

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":  #action 0
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if c > 0:
                        P_pi[cnt, cnt - 1] = 1.0
                    else:
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "right":  #action 1
            #always transition one to right unless already at right border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if c < self.num_cols - 1:
                        #transition to next state to right
                        P_pi[cnt, cnt + 1] = 1.0
                    else:
                        #self transition
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi
        

class MachineReplacementMDP(ChainMDP):
    #basic MDP class that has two actions (left, right), no terminal states and is a chain mdp with deterministic transitions
    def __init__(self, num_states, r_sa, gamma, init_dist):
        self.num_actions = 2
        self.num_rows = 1
        self.num_cols = num_states
        self.num_states =  num_states
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = []
       
        self.r_sa = r_sa


        self.P_noop = self.get_transitions(policy="noop")
        #print("P_left\n",self.P_left)
        self.P_repair = self.get_transitions(policy="repair")
        #print("P_right\n",self.P_right)
        self.Ps = [self.P_noop, self.P_repair]


    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "noop" #no-op
        elif action_num == 1:
            return "repair" #repair
        else:
            print("error, only two possible actions")
            sys.exit()

    
    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "noop":  #action 0
            #always transition to one state farther in chain unless at the last state where you go to the beginning
            for c in range(self.num_cols):
                if c < self.num_cols - 1:
                    #continue to the right
                    P_pi[c, c + 1] = 1.0
                else:
                    #go back to the beginning
                    P_pi[c,0] = 1.0
            
        elif policy == "repair":  #action 1
            #always transition back to the first state
            for c in range(self.num_cols):
                P_pi[c,0] = 1.0
                
        return P_pi


# In[2]:


get_ipython().system('pip install python-interface')


# In[14]:


import numpy as np
import random


def get_optimal_policy_from_usa(u_sa, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    opt_stoch_pi = np.zeros((num_states, num_actions))
    for s in range(num_states):
        #compute the total occupancy for that state across all actions
        s_tot_occupancy = np.sum(u_sa[s::num_states])
        for a in range(num_actions):
            opt_stoch_pi[s][a] = u_sa[s+a*num_states] / max(s_tot_occupancy, 0.00000001)
    return opt_stoch_pi


def print_stochastic_policy_action_probs(u_sa, mdp_env):
    opt_stoch = get_optimal_policy_from_usa(u_sa, mdp_env)
    print_stoch_policy(opt_stoch, mdp_env)



def print_stoch_policy(stoch_pi, mdp_env):
    for s in range(mdp_env.get_num_states()):
        action_prob_str = "state {}: ".format(s)
        for a in range(mdp_env.get_num_actions()):
            action_prob_str += "{} = {:.3f}, ".format(mdp_env.get_readable_actions(a), stoch_pi[s,a])
        print(action_prob_str)


def print_table_row(vals):
    row_str = ""
    for i in range(len(vals) - 1):
        row_str += "{:0.2} & ".format(vals[i])
    row_str += "{:0.2} \\\\".format(vals[-1])
    return row_str


def print_policy_from_occupancies(proposal_occupancies, mdp_env):
    policy = get_optimal_policy_from_usa(proposal_occupancies, mdp_env)
    cnt = 0
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            if cnt not in mdp_env.terminals:
                row_str += mdp_env.get_readable_actions(np.argmax(policy[cnt])) + "\t"
            else:
                row_str += ".\t"  #denote terminal with .
            cnt += 1
        print(row_str)

def get_policy_string_from_occupancies(u_sa, mdp_env):
    #get stochastic policy
    opt_stoch = get_optimal_policy_from_usa(u_sa, mdp_env)
    cnt = 0
    policy_string_list = []
    for s in range(mdp_env.num_states):
        if s in mdp_env.terminals:
            policy_string_list.append(".")
        else:
            action_str = ""
            for a in range(mdp_env.num_actions):
                if opt_stoch[s,a] > 0.001:
                    action_str += mdp_env.get_readable_actions(a)
            policy_string_list.append(action_str)
    return policy_string_list


def get_stoch_policy_string_dictionary_from_occupancies(u_sa, mdp_env):
    #get stochastic policy
    opt_stoch = get_optimal_policy_from_usa(u_sa, mdp_env)
    cnt = 0
    policy_string_dictionary_list = []
    for s in range(mdp_env.num_states):
        if s in mdp_env.terminals:
            policy_string_dictionary_list.append({".":1.0})
        else:
            action_prob_dict = {}
            for a in range(mdp_env.num_actions):
                action_prob_dict[mdp_env.get_readable_actions(a)] = opt_stoch[s,a]
            policy_string_dictionary_list.append(action_prob_dict)
    return policy_string_dictionary_list


def print_policy_from_occupancies(u_sa,mdp_env):
    cnt = 0
    policy = get_optimal_policy_from_usa(u_sa, mdp_env)
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            if cnt not in mdp_env.terminals:
                row_str += mdp_env.get_readable_actions(np.argmax(policy[cnt])) + "\t"
            else:
                row_str += ".\t"  #denote terminal with .
            cnt += 1
        print(row_str)






# In[13]:


def solve_mdp_lp(mdp_env, reward_sa=None, debug=False):


    I_s = np.eye(mdp_env.num_states)
    gamma = mdp_env.gamma

    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)

    
    b_eq = mdp_env.init_dist
    if reward_sa is not None:
        c = -1.0 * reward_sa 
    else:
        c = -1.0 * mdp_env.r_sa  

    sol = linprog(c, A_eq=A_eq, b_eq = b_eq)

    u_sa = sol['x'] 

    print("expected value dot product", np.dot(u_sa, mdp_env.r_sa))
    
    return u_sa


# In[5]:


def get_state_values(occupancy_frequencies, mdp_env):
    num_states, gamma = mdp_env.num_states, mdp_env.gamma
    r_sa = mdp_env.get_state_action_rewards()
    #get optimal stochastic policy
    stochastic_policy = get_optimal_policy_from_usa(occupancy_frequencies, mdp_env)
    
    reward_policy = get_policy_rewards(stochastic_policy, r_sa)
    transitions_policy = get_policy_transitions(stochastic_policy, mdp_env)
    A = np.eye(num_states) - gamma * transitions_policy 
    b = reward_policy
    #solve for value function
    state_values = np.linalg.solve(A, b)

    return state_values
    

def get_q_values(occupancy_frequencies, mdp_env):
    num_actions, gamma = mdp_env.num_actions, mdp_env.gamma
    r_sa = mdp_env.get_state_action_rewards()
    #get state values
    state_values = get_state_values(occupancy_frequencies, mdp_env)
    #get state-action values
    Ps = tuple(mdp_env.Ps[i] for i in range(num_actions))
    P_column = np.concatenate(Ps, axis=0)
    #print(P_column)
    q_values = r_sa + gamma * np.dot(P_column, state_values)
    return q_values


# In[6]:


def get_policy_transitions(stoch_pi, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    P_pi = np.zeros((num_states, num_states))
    #calculate expectations
    for s1 in range(num_states):
        for s2 in range(num_states):
            cum_prob = 0.0
            for a in range(num_actions):
                cum_prob += stoch_pi[s1,a] * mdp_env.get_transition_prob(s1,a,s2)
            P_pi[s1,s2] =  cum_prob
    return P_pi

def get_policy_state_occupancy_frequencies(stoch_policy, mdp_env):
    P_pi = get_policy_transitions(stoch_policy, mdp_env)
    A = np.eye(mdp_env.get_num_states()) - mdp_env.gamma * P_pi.transpose()
    return np.linalg.solve(A, mdp_env.init_dist)
def get_policy_expected_return(stoch_policy, mdp_env):
    u_pi = get_policy_state_occupancy_frequencies(stoch_policy, mdp_env)
    R_pi = get_policy_rewards(stoch_policy, mdp_env.r_sa)
    return np.dot(u_pi, R_pi)


# In[7]:


def get_policy_rewards(stoch_pi, rewards_sa):
    num_states, num_actions = stoch_pi.shape
    policy_rewards = np.zeros(num_states)
    for s, a_probs in enumerate(stoch_pi):
        expected_reward = 0.0
        for a, prob in enumerate(a_probs):
            index = s + num_states * a
            expected_reward += prob * rewards_sa[index]
        policy_rewards[s] = expected_reward
    return policy_rewards


# In[8]:


def calc_frontier(mdp_env, u_expert, reward_posterior, posterior_probs, lambda_range, alpha, debug=False):

    
    cvar_exprews = []

    for lamda in lambda_range:
        cvar_opt_usa, cvar_value, exp_ret = solve_max_cvar_policy(mdp_env, u_expert, reward_posterior, posterior_probs, alpha, debug, lamda)
        
        print("Policy for lambda={} and alpha={}".format(lamda, alpha))
        print_policy_from_occupancies(cvar_opt_usa, mdp_env)
        print("stochastic policy")
        print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env)
        print("CVaR of policy = {}".format(cvar_value))
        print("Expected return of policy = {}".format(exp_ret))
        cvar_exprews.append((cvar_value, exp_ret))
    return cvar_exprews


# In[9]:


def solve_max_cvar_policy(mdp_env, u_expert, posterior_rewards, p_R, alpha, debug=False, lamda = 0.0):


    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim!
    k = mdp_env.num_states * mdp_env.num_actions

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()

    posterior_probs = p_R
    #new objective is 
    #max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

    #so the decision variables are (in the following order) all the u(s,a) and sigma, and all the z's.

    #we want to maximize so take the negative of this vector and minimize via scipy 
    u_coeff = np.dot(R, posterior_probs)
    c_cvar = -1. * np.concatenate((lamda * u_coeff, #for the u(s,a)'s (if lamda = 0 then no in objective, this is the lambda * p^T R^T u)
                        (1-lamda) * np.ones(1),                 #for sigma
                        (1-lamda) * -1.0/(1.0 - alpha) * posterior_probs))  #for the auxiliary variables z

    #constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

    #create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
    # and n rows (one for each z variable)
    auxiliary_constraints = np.zeros((n, k + 1 + n))
    for i in range(n):
        z_part = np.zeros(n)
        z_part[i] = -1.0 #make the part for the auxiliary variable >= the part in the relu
        z_row = np.concatenate((-R[:,i],  #-R_i(s,a)'s
                                np.ones(1),    #sigma
                                z_part))
        auxiliary_constraints[i,:] = z_row

    #add the upper bounds for these constraints:
    #check to see if we have mu or u
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        auxiliary_b = -1. * np.dot(posterior_rewards.transpose(), u_expert)
    else:
        auxiliary_b = -1. * np.dot(R.transpose(), u_expert)

    #add the non-negativitity constraints for the vars u(s,a) and z(R). 
    #mu's greater than or equal to zero
    auxiliary_u_geq0 = -np.eye(k, M=k+1+n)  #negative since constraint needs to be Ax<=b
    auxiliary_bu_geq0 = np.zeros(k)

    auxiliary_z_geq0 = np.concatenate((np.zeros((n, k+1)), -np.eye(n)), axis=1)
    auxiliary_bz_geq0 = np.zeros(n)

    #don't forget the normal MDP constraints over the mu(s,a) terms
    I_s = np.eye(num_states)
    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)


    b_eq = mdp_env.init_dist
    A_eq_plus = np.concatenate((A_eq, np.zeros((mdp_env.num_states,1+n))), axis=1)  #add zeros for sigma and the auxiliary z's

    A_cvar = np.concatenate((auxiliary_constraints,
                            auxiliary_u_geq0,
                            auxiliary_z_geq0), axis=0)
    b_cvar = np.concatenate((auxiliary_b, auxiliary_bu_geq0, auxiliary_bz_geq0))

    #solve the LP
    sol = linprog(c_cvar, A_eq=A_eq_plus, b_eq = b_eq, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    if debug: print("solution to optimizing CVaR")
    if debug: print(sol)
    
    if sol['success'] is False:
        #print(sol)
        print("didn't solve correctly!")
        input("Continue?")
    #the solution of the LP corresponds to the CVaR
    var_sigma = sol['x'][k] #get sigma (this is VaR (at least close))
    cvar_opt_usa = sol['x'][:k]

    #calculate the CVaR of the solution
    if k != len(u_expert):
        relu_part = var_sigma * np.ones(n) - np.dot(np.transpose(R), cvar_opt_usa) + np.dot(np.transpose(posterior_rewards), u_expert)
    else:
        relu_part = var_sigma * np.ones(n) - np.dot(np.transpose(R), cvar_opt_usa) + np.dot(np.transpose(R), u_expert)
    #take max with zero
    relu_part[relu_part < 0] = 0.0
    cvar = var_sigma - 1.0/(1 - alpha) * np.dot(posterior_probs, relu_part)

    #calculate expected return of optimized policy
    if k != len(u_expert):
 
        assert len(u_expert) == weight_dim
        exp_baseline_perf = np.dot(posterior_probs, np.dot(posterior_rewards.transpose(), u_expert))
    
    else:
        exp_baseline_perf = np.dot(np.dot(R, posterior_probs), u_expert)


    cvar_exp_ret = np.dot( np.dot(R, posterior_probs), cvar_opt_usa) - exp_baseline_perf

    if debug: print("CVaR = ", cvar)
    if debug: print("policy u(s,a) = ", cvar_opt_usa)
    cvar_opt_stoch_pi = get_optimal_policy_from_usa(cvar_opt_usa, mdp_env)
    if debug: print("CVaR opt stochastic policy")
    if debug: print(cvar_opt_stoch_pi)

    if debug:
        if k != len(u_expert):
            policy_losses = np.dot(R.transpose(), cvar_opt_usa)  - np.dot(posterior_rewards.transpose(), u_expert)
        else:
            policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)
        print("policy losses:", policy_losses)
    if debug: 
        if k != len(u_expert):
            print("expert returns:", np.dot(posterior_rewards.transpose(), u_expert))
        else:
            print("expert returns:", np.dot(R.transpose(), u_expert))
    if debug: print("my returns:", np.dot(R.transpose(), cvar_opt_usa))

    return cvar_opt_usa, cvar, cvar_exp_ret


# # grid world

# In[10]:


class BasicGridMDP(implements(MDP)):
    
    def __init__(self, num_rows, num_cols, r_s, gamma, init_dist, terminals = [], debug=False):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = terminals
        self.debug = debug
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.r_s)
        print("transformed R(s,a)", self.r_sa)

        self.init_states = []
        for s in range(self.num_states):
            if self.init_dist[s] > 0:
                self.init_states.append(s)


        self.P_left = self.get_transitions(policy="left")
        if self.debug: print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        if self.debug: print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        if self.debug: print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        if self.debug: print("P_down\n",self.P_down)
        self.Ps = [self.P_left, self.P_right, self.P_up, self.P_down] #actions:0,1,2,3


    def get_transition_prob_matrices(self):
        return self.Ps

    def get_num_actions(self):
        return self.num_actions

    def get_num_states(self):
        return self.num_states

   
    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "<"
        elif action_num == 1:
            return ">"
        elif action_num == 2:
            return "^"
        elif action_num == 3:
            return "v"
        else:
            print("error, only four possible actions")
            sys.exit()


    def get_transition_prob(self, s1,a,s2):
        return self.Ps[a][s1][s2]

    #Note that I'm using r_s as the reward dim not r_sa!
    def get_reward_dimensionality(self):
        return len(self.r_s)

    #NOTE: the dimensionality still needs to be checked.
    def uses_linear_approximation(self):
        return False

    def get_state_action_rewards(self):
        return self.r_sa

    #assume new reward is of the form r_s
    def set_reward_fn(self, new_reward):
        self.r_s = new_reward
        #also update r_sa
        self.r_sa = self.transform_to_R_sa(self.r_s)



   
    def transform_to_R_sa(self, reward_weights):     
        assert(len(reward_weights) == self.num_states)
        return np.tile(reward_weights, self.num_actions)

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":  #action 0 
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if c > 0:
                            P_pi[cnt, cnt - 1] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "right":  #action 1
            #always transition one to right unless already at right border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if c < self.num_cols - 1:
                            #transition to next state to right
                            P_pi[cnt, cnt + 1] = 1.0
                        else:
                            #self transition
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "up": #action 2
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if r > 0:
                            P_pi[cnt, cnt - self.num_cols] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "down":  #action 3
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if r < self.num_rows - 1:
                            P_pi[cnt, cnt + self.num_cols] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi


class FeaturizedGridMDP(BasicGridMDP):


    def __init__(self,num_rows, num_cols, state_feature_matrix, feature_weights, gamma, init_dist, terminals = [], debug=False):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = terminals
        self.debug = debug

        self.init_states = []
        for s in range(self.num_states):
            if self.init_dist[s] > 0:
                self.init_states.append(s)

        
        self.P_left = self.get_transitions(policy="left")
        if self.debug: print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        if self.debug: print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        if self.debug: print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        if self.debug: print("P_down\n",self.P_down)
        self.Ps = [self.P_left, self.P_right, self.P_up, self.P_down] #actions:0,1,2,3

        #figure out reward function
        self.state_features = state_feature_matrix
        self.feature_weights = feature_weights
        r_s = np.dot(self.state_features, self.feature_weights)
        print("r_s", r_s)
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.feature_weights)
        print("transformed R(s,a)", self.r_sa)


    def get_reward_dimensionality(self):
        return len(self.feature_weights)

    def uses_linear_approximation(self):
        return True

    def set_reward_fn(self, new_reward):
        #input is the new_reward weights
        assert(len(new_reward) == len(self.feature_weights))
        #update feature weights
        self.feature_weights = new_reward.copy()
        #update r_s
        self.r_s = np.dot(self.state_features, new_reward)
        #update r_sa
        self.r_sa = np.tile(self.r_s, self.num_actions)


    def transform_to_R_sa(self, reward_weights):
        #assumes that inputs are the reward feature weights or state rewards
        #returns the vectorized R_sa 
        
        #first get R_s
        if len(reward_weights) == self.get_reward_dimensionality():
            R_s = np.dot(self.state_features, reward_weights)
        elif len(reward_weights) == self.num_states:
            R_s = reward_weights
        else:
            print("Error, reward weights should be features or state rewards")
            sys.exit()
        return np.tile(R_s, self.num_actions)




# In[ ]:




