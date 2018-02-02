from __future__ import division
import logging, sys

import numpy as np

from mdp_lib.domains.gridworld import GridWorld
from ped_irl.discretizedobmdp import DiscretizedObserverBeliefMDPApproximation
from mdp_lib.util import sample_prob_dict

from itertools import product

logger = logging.getLogger(__name__)

class PlanningInObserverBeliefModel(object):
    def __init__(self,
                 goal_reward=50,
                 true_belief_reward = 50,
                 danger_reward = -10,
                 step_cost = 0,
                 wall_action = False,
                 wait_action = False,
                 init_ground_state=(0, 2),
                 ground_goal_state=(5, 2),

                 base_discount_rate = .99,
                 base_softmax_temp = 1,
                 obmdp_discount_rate = .99,
                 true_mdp_code='oox',

                 obmdp_softmax_temp=1,

                 belief_reward_isterminal = False,

                 n_probability_bins=5,
                 seed_trajs=None,
                 discretized_tf=None
    ):
        self.ground_goal_state = ground_goal_state
        self.obmdp_softmax_temp = obmdp_softmax_temp
        self.obmdp_discount_rate = obmdp_discount_rate
        #task parameters

        state_features = [
            '.oooo.',
            '.oppp.',
            '.opccy',
            '.oppc.',
            '.cccc.'
        ]

        #=============================#
        #   Build set of ground MDPs  #
        #=============================#
        mdp_params = []
        feature_rewards = [dict(zip('opc', rs)) for rs in product([0, danger_reward],
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

        #===========================================#
        #   Build Observer Belief MDP and support   #
        #===========================================#
        logger.debug("Building OBMDP")
        ob_mdp = DiscretizedObserverBeliefMDPApproximation(**{
            'n_probability_bins': n_probability_bins,

            'init_ground_state': init_ground_state,
            'mdp_params': mdp_params,
            'mdp_codes': mdp_codes,
            'MDP': GridWorld,
            'base_softmax_temp': base_softmax_temp,
            'true_belief_reward': true_belief_reward,
            'base_policy_type': 'softmax',
            'true_mdp_i': mdp_codes.index(true_mdp_code),
            'belief_reward_isterminal': False,

            'discount_rate': obmdp_discount_rate,

            'discretized_tf': discretized_tf
        })
        self.ob_mdp = ob_mdp

    def build(self):
        self.ob_mdp.build_discretized_tf()
        self.ob_mdp.create_start_state()

    def solve(self, **kwargs):
        logger.debug("Running Discretized OBMDP (%d states)" \
                     % len(self.ob_mdp.disc_tf))
        self.ob_mdp.solve(**kwargs)

    def fit_traj(self, traj, obmdp_softmax_temp=None, log=False):
        if obmdp_softmax_temp is None:
            obmdp_softmax_temp = self.obmdp_softmax_temp

        s = self.ob_mdp.get_init_state()
        prob = 1
        loglike = 0
        for ti, (w, a) in enumerate(traj):
            if a == '%':
                break
            smprobs = self.ob_mdp.get_softmax_actionprobs(s,
                                                          temp=obmdp_softmax_temp)
            try:
                prob *= smprobs[a]
            except FloatingPointError:
                prob = 0
            loglike += np.log(smprobs[a])
            ns_dict = self.ob_mdp.transition_dist(s=s, a=a)
            for ns in ns_dict.iterkeys():
                nb, nw = ns
                part_nw = traj[ti+1][0]
                if nw == part_nw:
                    break
            s = ns

        if log:
            return loglike
        return prob

    def get_belief_traj(self, traj):
        btraj = []
        s = self.ob_mdp.get_init_state()
        for ti in xrange(len(traj)):
            w, a = traj[ti]
            btraj.append((s, a))
            if a == '%':
                break
            ns_dist = self.ob_mdp.transition_dist(s=s, a=a)
            pnw, _ = traj[ti + 1]
            for nb, nw in ns_dist.iterkeys():
                if nw == pnw:
                    break
            s = (nb, nw)
        return btraj

    def seed_beliefs_with_trajs(self, trajs,
                                            branch_steps=0):
        '''

        :param trajs:
        :param branch_steps: Branch out this many steps from each state
         in each participant trajectory provided
        :return:
        '''
        true_obmdp = super(DiscretizedObserverBeliefMDPApproximation,
                           self.ob_mdp)
        visited_states = set([])
        beliefs = set([])
        for traj in trajs:
            s = true_obmdp.get_init_state()
            b, _ = s
            beliefs.add(b)
            for ti in xrange(len(traj)):
                _, a = traj[ti]
                if a == '%':
                    break
                pnw, _ = traj[ti + 1]
                for nb, nw in true_obmdp.transition_dist(s, a).iterkeys():
                    if nw == pnw:
                        break
                beliefs.add(nb)
                s = (nb, nw)
                visited_states.add(s)
        branched_states = self._branch_from_states(visited_states,
                                                    true_obmdp,
                                                    branch_steps)
        branched_beliefs = set([b for b, w in branched_states])
        for b in branched_beliefs:
            beliefs.add(b)
        self.ob_mdp.add_seed_beliefs(list(beliefs))

    def _branch_from_states(self, init_states, true_obmdp, branch_steps):
        def _branch(state, depth):
            if depth == 0:
                return [state,]
            branched_states = []
            next_states = set([])
            for a in true_obmdp.available_actions(state):
                for ns in true_obmdp.transition_dist(s=state, a=a).keys():
                    next_states.add(ns)
            for ns in next_states:
                branched_states.extend(_branch(ns, depth-1))
            return branched_states

        branched_states = []
        for s in init_states:
            branched_states.extend(_branch(s, branch_steps))
        return branched_states


    def _generate_model_traj(self, obmdp_softmax_temp):
        s = self.ob_mdp.get_init_state()
        traj = []
        for t_i in xrange(100):
            if self.ob_mdp.is_terminal(s):
                break
            smprobs = self.ob_mdp.get_softmax_actionprobs(
                s, temp=obmdp_softmax_temp)
            a = sample_prob_dict(smprobs)
            traj.append((s, a))
            ns_dist = self.ob_mdp.transition_dist(s, a)
            ns = sample_prob_dict(ns_dist)
            s = ns
        return traj

    def generate_model_trajs(self,
                             n_trajs=1000,
                             obmdp_softmax_temp=None):
        trajs = []
        for _ in xrange(n_trajs):
            trajs.append(self._generate_model_traj(obmdp_softmax_temp))
        return trajs

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        # filename=logfile,
                        level=logging.DEBUG,
                        format='%(asctime)s : %(name)s : %(message)s',
                        datefmt='%H:%M:%S')

    model = PlanningInObserverBeliefModel(
        base_softmax_temp = 1,
        obmdp_discount_rate = .99,
        belief_reward_isterminal = False,

        true_mdp_code='oox',

        n_probability_bins=3
    )
    model.seed_beliefs_with_trajs(
        [[((0, 2), '>'),
          ((1, 2), '>'),
          ((2, 2), '^'),
          ((2, 3), '>'),
          ((3, 3), '>'),
          ((4, 3), '>'),
          ((5, 3), 'v'),
          ((5, 2), '%')],
         [((0, 2), 'v'),
          ((0, 1), 'v'),
          ((0, 0), '>'),
          ((1, 0), '>'),
          ((2, 0), '>'),
          ((3, 0), '>'),
          ((4, 0), '>'),
          ((5, 0), '^'),
          ((5, 1), '^'),
          ((5, 2), '%')],
         [((0, 2), '>'),
          ((1, 2), '>'),
          ((2, 2), '>'),
          ((3, 2), '>'),
          ((4, 2), '>'),
          ((5, 2), '%')]]
    )
    model.build()
    model.solve()
    trajs = model.generate_model_trajs(obmdp_softmax_temp=.2)