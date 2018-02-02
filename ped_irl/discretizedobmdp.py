from __future__ import division
from itertools import combinations_with_replacement
import logging

import numpy as np
from sklearn.neighbors import BallTree

from obs_belief_mdp import ObserverBeliefMDP
from mdp_lib.util import calc_softmax_policy, calc_softmax_dist

logger = logging.getLogger(__name__)

class DiscretizedObserverBeliefMDPApproximation(ObserverBeliefMDP):
    def __init__(self,
                 n_probability_bins=5,
                 seed_beliefs=None,

                 discretized_tf=None,
                 **kwargs):
        super(DiscretizedObserverBeliefMDPApproximation,
              self).__init__(**kwargs)
        self.n_pbins = n_probability_bins

        binsize = 1/n_probability_bins
        self.pbins = np.linspace(binsize/2, 1-binsize/2, n_probability_bins)
        self.discretization = np.linspace(0, 1, n_probability_bins+1)

        # set up points for discretization
        #   This is a combination of general grid points and belief points
        # that are seeding the discretization
        if seed_beliefs is None:
            seed_beliefs = []
        self.seed_beliefs = seed_beliefs

        if discretized_tf is not None:
            logger.debug("Discretized Transition Function Provided")
            self.disc_tf = discretized_tf
            self.belief_points = list(set([b for b, _ in self.disc_tf.iterkeys()]))
            bp_nbrs = BallTree(self.belief_points, leaf_size=40)
            self.bp_nbrs = bp_nbrs
            self.create_start_state()
        else:
            logger.debug("Discretized Transition Function NOT Provided")

    def create_start_state(self):
        #set up start state
        start_state = super(DiscretizedObserverBeliefMDPApproximation,
                            self).get_init_state()
        start_state = self.discretize_b(s=start_state)
        self.start_state = start_state

    def add_seed_beliefs(self, new_beliefs):
        self.seed_beliefs.extend(new_beliefs)

    def build_discretized_tf(self):
        belief_grid_points = self._gen_belief_grid(self.n_pbins)
        if len(self.seed_beliefs) > 0:
            self.belief_points = np.vstack((belief_grid_points,
                                            self.seed_beliefs))
        else:
            self.belief_points = belief_grid_points
        bp_nbrs = BallTree(self.belief_points, leaf_size=40)
        self.bp_nbrs = bp_nbrs

        self.disc_tf = self._gen_discretized_tf(self.belief_points)

    def _gen_belief_grid(self, n_probability_bins):
        belief_grid_points = []
        for divs in combinations_with_replacement(
                range(n_probability_bins + 1),
                r=len(self.mdp_codes)-1):
            b = [0, ] + list(divs) + [n_probability_bins, ]
            b = np.ediff1d(b)
            b = b / np.sum(b)
            belief_grid_points.append(tuple(b))
        return np.array(belief_grid_points)

    def _gen_discretized_tf(self, belief_points):
        # map actual next states to discretized next states
        obmdp = super(DiscretizedObserverBeliefMDPApproximation, self)
        true_mdp = obmdp.get_true_mdp()

        logger.debug("Building Discretized Transition Function")
        logger.debug("Discrete Space Size (n=%d)" % len(belief_points))
        next_beliefs = []
        for b in belief_points:
            b = tuple(b)
            for w in true_mdp.get_states():
                s = (b, w)
                for a in obmdp.available_actions(s):
                    for ns in obmdp.transition_dist(s, a).keys():
                        nb, nw = ns
                        next_beliefs.append(nb)

        disc_b_ind = self.bp_nbrs.query(next_beliefs, return_distance=False)

        # make it so the next belief is a mixture of closest beliefs?
        # disc_b_dist, disc_b_ind = self.bp_nbrs.query(next_beliefs, k=1)
        # nb_to_disc_nb = {}
        # for nb_i, nb in enumerate(next_beliefs):
        #     nearest_idx = disc_b_ind[nb_i]
        #     neighbor = belief_points[nearest_idx]
        #     dists = disc_b_dist[nb_i]

        nb_to_disc_nb = dict(zip(next_beliefs,
                                 belief_points[disc_b_ind].squeeze()))
        disc_tf = {}
        for b in belief_points:
            b = tuple(b)
            for w in true_mdp.get_states():
                s = (b, w)
                disc_tf[s] = {}
                for a in obmdp.available_actions(s):
                    disc_tf[s][a] = {}
                    p_norm = 0
                    for ns, p in obmdp.transition_dist(s, a).iteritems():
                        nb, nw = ns
                        ns = (tuple(nb_to_disc_nb[nb]), nw)
                        disc_tf[s][a][ns] = disc_tf[s][a].get(ns, 0.0) + p
                        p_norm += p
                    tdist = {ns: p / p_norm for ns, p in disc_tf[s][a].items()}
                    disc_tf[s][a] = tdist
        return disc_tf

    def get_discretized_tf(self):
        return self.disc_tf

    def discretize_b(self, s=None, b=None):
        if b is not None:
            dist, ind = self.bp_nbrs.query([b])
            return tuple(self.belief_points[ind[0][0]])
        elif s is not None:
            b, w = s
            b = self.discretize_b(b=b)
            return (b, w)

    def transition_dist(self, s, a):
        '''
        Discretizes the belief

        :param s:
        :param a:
        :return:
        '''
        if s not in self.disc_tf:
            s = self.discretize_b(s=s)
        return self.disc_tf[s][a]

    def get_states(self):
        return self.disc_tf.iterkeys()

    def get_init_state(self):
        return self.start_state

    def gen_reward_dict(self):
        rf = {}
        for s, a_ns_p in self.disc_tf.iteritems():
            rf[s] = {}
            for a, ns_p in a_ns_p.iteritems():
                rf[s][a] = {}
                for ns in ns_p.iterkeys():
                    rf[s][a][ns] = self.reward(s=s, a=a, ns=ns)
        return rf

    def get_softmax_actionprobs(self, s, temp=1):
        return calc_softmax_dist(self.action_value_function[s], temp)

    def get_softmax_function(self, temp=1):
        return calc_softmax_policy(self.action_value_function, temp)

if __name__ == '__main__':
    from itertools import product

    from mdp_lib.domains.gridworld import GridWorld

    goal_reward = 50
    true_belief_reward = 50
    danger_reward = -10
    step_cost = 0
    wall_action = False
    wait_action = False
    init_ground_state = (0, 2)
    ground_goal_state = (5, 2)

    base_discount_rate = .99
    base_softmax_temp = 1
    obmdp_discount_rate = .99
    true_mdp_code = 'oox'

    belief_reward_isterminal = False

    # task parameters

    state_features = [
        '.oooo.',
        '.oppp.',
        '.opccy',
        '.oppc.',
        '.cccc.'
    ]

    # =============================#
    #   Build set of ground MDPs  #
    # =============================#
    mdp_params = []
    feature_rewards = [dict(zip('opc', rs)) for rs in
                       product([0, danger_reward],
                               repeat=3)]
    mdp_codes = []
    for frewards in feature_rewards:
        rfc = ['o' if frewards[f] == 0 else 'x' for f in 'opc']
        rfc = ''.join(rfc)
        mdp_codes.append(rfc)
        frewards['y'] = goal_reward
        frewards['.'] = 0

    for mdpc, frewards in zip(mdp_codes, feature_rewards):
        params = {
            'gridworld_array': state_features,
            'feature_rewards': frewards,
            'absorbing_states': [ground_goal_state, ],
            'init_state': init_ground_state,
            'wall_action': wall_action,
            'step_cost': step_cost,
            'wait_action': wait_action,
            'discount_rate': base_discount_rate
        }
        mdp_params.append(params)

    # ===========================================#
    #   Build Observer Belief MDP and support   #
    # ===========================================#
    ob_mdp = DiscretizedObserverBeliefMDPApproximation(**{
        'n_probability_bins': 2,

        'init_ground_state': init_ground_state,
        'mdp_params': mdp_params,
        'mdp_codes': mdp_codes,
        'MDP': GridWorld,
        'base_softmax_temp': base_softmax_temp,
        'true_belief_reward': true_belief_reward,
        'base_policy_type': 'softmax',
        'true_mdp_i': mdp_codes.index(true_mdp_code),
        'belief_reward_isterminal': False,
        'discount_rate': .99
    })