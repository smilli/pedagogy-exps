#!/usr/bin/env python

import numpy as np

from mdp_lib import MDP as MDPClass
from mdp_lib import Policy, GridWorld
from mdp_lib.util import calc_softmax_policy

'''
An observer belief MDP takes a set of MDPs and tracks a belief state over them
that can be used by a showing agent.
'''

class ObserverBeliefMDP(MDPClass):
    def __init__(self, init_ground_state=None,
                 mdp_params=None,
                 mdp_codes=None,
                 MDP=None,
                 discount_rate=None,

                 base_softmax_temp=1,
                 base_policy_type='softmax',
                 true_mdp_i=None,
                 belief_reward_isterminal=True,
                 true_belief_reward=5,
                 update_includes_intention=True):
        '''
        Handles observer belief-MDP
        '''
        #set self parameters
        self.init_ground_state = init_ground_state
        self.base_softmax_temp = base_softmax_temp
        self.base_policy_type = base_policy_type
        self.true_mdp_i = true_mdp_i
        self.true_belief_reward = true_belief_reward
        self.belief_reward_isterminal = belief_reward_isterminal
        self.update_includes_intention = update_includes_intention
        if MDP is None:
            MDP = GridWorld
        if mdp_codes is None:
            mdp_codes = [str(i) for i in range(len(mdp_params))]
        self.mdp_codes = tuple(mdp_codes)
        self.MDP = MDP
        self.discount_rate = discount_rate

        #initialize mdp space
        self.mdps = []
        for p in mdp_params:
            self.mdps.append(MDP(include_intermediate_terminal=True, **p))

        for mdp in self.mdps:
            mdp.solve()



        self.softmax_policies = []
        for mdp in self.mdps:
            av = mdp.action_value_function
            smp = calc_softmax_policy(av, temp=self.base_softmax_temp)
            self.softmax_policies.append(smp)

        self.transition_functions = []
        for mdp in self.mdps:
            self.transition_functions.append(mdp.gen_transition_dict())

        self.true_mdp = self.mdps[self.true_mdp_i]
        self.rmax = np.max([true_belief_reward, self.true_mdp.rmax])
        self.n_actions = self.true_mdp.n_actions
        self.terminal_state_reward = self.true_mdp.terminal_state_reward

        self.transition_dist_cache = {}
    
    def get_init_state(self):
        b = tuple(np.ones(len(self.mdps))/len(self.mdps))
        w = self.init_ground_state
        return (b, w)
    
    def transition(self, s, a):
        '''
        Calculates change in observer's belief state as a result of seeing action.
        s is a tuple of (belief, world-state)
        a is an action in the world mdp
        '''
        tdist = self.transition_dist(s, a)
        nss, ps = zip(*tdist.iteritems())
        return nss[np.random.choice(range(len(nss)), p=ps)]

    def transition_dist(self, s, a):
        try:
            return self.transition_dist_cache[(s, a)]
        except KeyError:
            pass
        b, w = s
        nw_p = self.mdps[self.true_mdp_i].transition_dist(w, a)
        nb_a = np.array([smp[w][a] for smp in self.softmax_policies])
        t_dist = {}
        for nw, p in nw_p.iteritems():
            if p == 0:
                continue
            nb_nw = np.array([tf[w][a].get(nw, 0) for tf in self.transition_functions])
            if self.update_includes_intention:
                nb = b*nb_a*nb_nw
            else:
                nb = b*nb_nw
            nb = tuple(nb / np.sum(nb))
            t_dist[(nb, nw)] = p

        self.transition_dist_cache[(s, a)] = t_dist
        return t_dist
    
    def get_adjacent_states(self, s):
        b, w = s
        actions = self.mdps[0].available_actions(w)
        adj_states = []
        for a in actions:
            ns = self.transition(s, a)
            adj_states.append(ns)
        return adj_states
    
    def get_belief_transition_likelihoods(self, s, a):
        b,w = s
        return [smp[w][a] for smp in self.softmax_policies]
    
    def reward(self, s=None, a=None, ns=None, only_belief_reward=False):
        b, w = s
        nb, nw = ns
        wr = self.true_mdp.reward(s=w, a=a, ns=nw)
        if self.belief_reward_isterminal:
            if w == self.true_mdp.intermediate_terminal:
                br = self.true_belief_reward*nb[self.true_mdp_i]
            else:
                br = 0
        else:
            bchange = nb[self.true_mdp_i] - b[self.true_mdp_i]
            br = bchange*self.true_belief_reward

        if only_belief_reward:
            return br
        return br + wr

    def is_terminal(self, state):
        b, w = state
        return self.mdps[self.true_mdp_i].is_terminal(w)

    def available_actions(self, s):
        b, w = s
        return self.mdps[self.true_mdp_i].available_actions(w)

    def get_true_mdp(self):
        return self.true_mdp

    def __hash__(self):
        #todo make differently ordered mdps equivalent
        myhash = (
            self.init_ground_state,
            self.base_softmax_temp,
            tuple(self.mdps),
            self.mdp_codes,
            self.base_policy_type,
            self.true_mdp_i,
            self.true_belief_reward,
            self.MDP.__name__,
            self.belief_reward_isterminal
        )
        return hash(myhash)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return False

class ObserverBeliefMDPGroundPolicyWrapper(Policy):
    def __init__(self, ground_policy):
        self.ground_policy = ground_policy

    def get_action(self, state):
        return self.ground_policy.get_action(state[1])


