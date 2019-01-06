import itertools
from collections import Counter

import numpy as np

from baselines.constraint.dfa import DFA

id_fn = lambda x: x


def quadrant(cos, sin):
    if cos >= 0:
        if sin >= 0:
            return 0
        else:
            return 3
    else:
        if sin >= 0:
            return 1
        else:
            return 2


def reacher_state_quadrants(obs):
    angles = obs[:4]
    q1 = quadrant(angles[0], angles[2])
    q2 = quadrant(angles[1], angles[3])
    val = q1 * 4 + q2
    return val  #hex(val)[2:]


def reacher_action_threshold(act):
    return np.linalg.norm(act) > 1.


class Constraint(DFA):
    def __init__(self,
                 name,
                 reg_ex,
                 violation_reward,
                 s_tl=id_fn,
                 a_tl=id_fn,
                 s_active=True,
                 a_active=True):
        super(Constraint, self).__init__(reg_ex)
        self.name = name
        self.violation_reward = violation_reward
        self.s_tl = s_tl
        self.a_tl = a_tl
        self.s_active = s_active
        self.a_active = a_active

    def step(self, obs, action, done):
        is_viol = False
        if self.s_active:
            is_viol = is_viol | super().step(self.s_tl(obs))
        if self.a_active:
            is_viol = is_viol | super().step(self.a_tl(action))
        rew_mod = self.violation_reward if is_viol else 0.
        if is_viol: print('violated')
        return is_viol, rew_mod

    def reset(self):
        return super().reset()


class CountingPotentialConstraint(Constraint):
    def __init__(self,
                 name,
                 reg_ex,
                 violation_reward,
                 gamma,
                 s_tl=id_fn,
                 a_tl=id_fn,
                 s_active=True,
                 a_active=True):
        super(CountingPotentialConstraint, self).__init__(
            name, reg_ex, violation_reward, s_tl, a_tl, s_active, a_active)
        self.episode_visit_count = Counter()
        self.visit_count = Counter(self.states())
        self.violation_count = Counter(self.accepting_states())
        self.gamma = gamma
        self.prev_state = self.current_state

    def get_state_potentials(self):
        potential = lambda s: self.violation_count[s] / self.visit_count[s]
        return {s: potential(s) for s in self.states()}

    def step(self, obs, action, done):
        is_viol, _ = super().step(obs, action, done)
        dfa_state = self.current_state
        self.episode_visit_count[dfa_state] += 1

        current_viol_propn = (
            self.violation_count[dfa_state] / self.visit_count[dfa_state])
        prev_viol_propn = (self.violation_count[self.prev_state] /
                           self.visit_count[self.prev_state])
        rew_mod = (self.gamma * current_viol_propn -
                   prev_viol_propn) * self.violation_reward
        if self.prev_state in self.accepting_states(): rew_mod = 0

        if is_viol:
            self.violation_count += self.episode_visit_count
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()
        if done:
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()

        self.prev_state = dfa_state
        return is_viol, rew_mod


ACTUATION1D_REGEX_k = lambda k: '2{k}|3{k}'.format(k=k)
DITHERING1D_REGEX_k = lambda k: '(23){k}|(32){k}|(0|1|2|3){k2}'.format(k=k, k2=2*k)
DITHERING2D_REGEX_4 = '((2|A)(2|A)(5|D)(5|D))|((2|A)(5|D)(2|A)(5|D))|((2|A)(5|D)(5|D)(2|A))|((5|D)(2|A)(2|A)(5|D))|((5|D)(2|A)(5|D)(2|A))|((5|D)(5|D)(2|A)(2|A))|((2|A)(2|A)(8|G)(9|H))|((2|A)(2|A)(9|H)(8|G))|((2|A)(8|G)(2|A)(9|H))|((2|A)(8|G)(9|H)(2|A))|((2|A)(9|H)(2|A)(8|G))|((2|A)(9|H)(8|G)(2|A))|((8|G)(2|A)(2|A)(9|H))|((8|G)(2|A)(9|H)(2|A))|((8|G)(9|H)(2|A)(2|A))|((9|H)(2|A)(2|A)(8|G))|((9|H)(2|A)(8|G)(2|A))|((9|H)(8|G)(2|A)(2|A))|((2|A)(3|B)(4|C)(5|D))|((2|A)(3|B)(5|D)(4|C))|((2|A)(4|C)(3|B)(5|D))|((2|A)(4|C)(5|D)(3|B))|((2|A)(5|D)(3|B)(4|C))|((2|A)(5|D)(4|C)(3|B))|((3|B)(2|A)(4|C)(5|D))|((3|B)(2|A)(5|D)(4|C))|((3|B)(4|C)(2|A)(5|D))|((3|B)(4|C)(5|D)(2|A))|((3|B)(5|D)(2|A)(4|C))|((3|B)(5|D)(4|C)(2|A))|((4|C)(2|A)(3|B)(5|D))|((4|C)(2|A)(5|D)(3|B))|((4|C)(3|B)(2|A)(5|D))|((4|C)(3|B)(5|D)(2|A))|((4|C)(5|D)(2|A)(3|B))|((4|C)(5|D)(3|B)(2|A))|((5|D)(2|A)(3|B)(4|C))|((5|D)(2|A)(4|C)(3|B))|((5|D)(3|B)(2|A)(4|C))|((5|D)(3|B)(4|C)(2|A))|((5|D)(4|C)(2|A)(3|B))|((5|D)(4|C)(3|B)(2|A))|((2|A)(3|B)(9|H))|((2|A)(3|B)(9|H))|((2|A)(9|H)(3|B))|((2|A)(9|H)(3|B))|((2|A)(3|B)(9|H))|((2|A)(9|H)(3|B))|((3|B)(2|A)(9|H))|((3|B)(2|A)(9|H))|((3|B)(9|H)(2|A))|((3|B)(9|H)(2|A))|((3|B)(2|A)(9|H))|((3|B)(9|H)(2|A))|((9|H)(2|A)(3|B))|((9|H)(2|A)(3|B))|((9|H)(3|B)(2|A))|((9|H)(3|B)(2|A))|((9|H)(2|A)(3|B))|((9|H)(3|B)(2|A))|((2|A)(3|B)(9|H))|((2|A)(9|H)(3|B))|((3|B)(2|A)(9|H))|((3|B)(9|H)(2|A))|((9|H)(2|A)(3|B))|((9|H)(3|B)(2|A))|((2|A)(4|C)(8|G))|((2|A)(4|C)(8|G))|((2|A)(8|G)(4|C))|((2|A)(8|G)(4|C))|((2|A)(4|C)(8|G))|((2|A)(8|G)(4|C))|((4|C)(2|A)(8|G))|((4|C)(2|A)(8|G))|((4|C)(8|G)(2|A))|((4|C)(8|G)(2|A))|((4|C)(2|A)(8|G))|((4|C)(8|G)(2|A))|((8|G)(2|A)(4|C))|((8|G)(2|A)(4|C))|((8|G)(4|C)(2|A))|((8|G)(4|C)(2|A))|((8|G)(2|A)(4|C))|((8|G)(4|C)(2|A))|((2|A)(4|C)(8|G))|((2|A)(8|G)(4|C))|((4|C)(2|A)(8|G))|((4|C)(8|G)(2|A))|((8|G)(2|A)(4|C))|((8|G)(4|C)(2|A))|((2|A)(5|D)(6|E)(9|H))|((2|A)(5|D)(9|H)(6|E))|((2|A)(6|E)(5|D)(9|H))|((2|A)(6|E)(9|H)(5|D))|((2|A)(9|H)(5|D)(6|E))|((2|A)(9|H)(6|E)(5|D))|((5|D)(2|A)(6|E)(9|H))|((5|D)(2|A)(9|H)(6|E))|((5|D)(6|E)(2|A)(9|H))|((5|D)(6|E)(9|H)(2|A))|((5|D)(9|H)(2|A)(6|E))|((5|D)(9|H)(6|E)(2|A))|((6|E)(2|A)(5|D)(9|H))|((6|E)(2|A)(9|H)(5|D))|((6|E)(5|D)(2|A)(9|H))|((6|E)(5|D)(9|H)(2|A))|((6|E)(9|H)(2|A)(5|D))|((6|E)(9|H)(5|D)(2|A))|((9|H)(2|A)(5|D)(6|E))|((9|H)(2|A)(6|E)(5|D))|((9|H)(5|D)(2|A)(6|E))|((9|H)(5|D)(6|E)(2|A))|((9|H)(6|E)(2|A)(5|D))|((9|H)(6|E)(5|D)(2|A))|((2|A)(5|D)(7|F)(8|G))|((2|A)(5|D)(8|G)(7|F))|((2|A)(7|F)(5|D)(8|G))|((2|A)(7|F)(8|G)(5|D))|((2|A)(8|G)(5|D)(7|F))|((2|A)(8|G)(7|F)(5|D))|((5|D)(2|A)(7|F)(8|G))|((5|D)(2|A)(8|G)(7|F))|((5|D)(7|F)(2|A)(8|G))|((5|D)(7|F)(8|G)(2|A))|((5|D)(8|G)(2|A)(7|F))|((5|D)(8|G)(7|F)(2|A))|((7|F)(2|A)(5|D)(8|G))|((7|F)(2|A)(8|G)(5|D))|((7|F)(5|D)(2|A)(8|G))|((7|F)(5|D)(8|G)(2|A))|((7|F)(8|G)(2|A)(5|D))|((7|F)(8|G)(5|D)(2|A))|((8|G)(2|A)(5|D)(7|F))|((8|G)(2|A)(7|F)(5|D))|((8|G)(5|D)(2|A)(7|F))|((8|G)(5|D)(7|F)(2|A))|((8|G)(7|F)(2|A)(5|D))|((8|G)(7|F)(5|D)(2|A))|((2|A)(5|D))|((2|A)(5|D))|((2|A)(5|D))|((5|D)(2|A))|((5|D)(2|A))|((5|D)(2|A))|((2|A)(5|D))|((2|A)(5|D))|((5|D)(2|A))|((5|D)(2|A))|((2|A)(5|D))|((5|D)(2|A))|((3|B)(3|B)(4|C)(4|C))|((3|B)(4|C)(3|B)(4|C))|((3|B)(4|C)(4|C)(3|B))|((4|C)(3|B)(3|B)(4|C))|((4|C)(3|B)(4|C)(3|B))|((4|C)(4|C)(3|B)(3|B))|((3|B)(3|B)(7|F)(9|H))|((3|B)(3|B)(9|H)(7|F))|((3|B)(7|F)(3|B)(9|H))|((3|B)(7|F)(9|H)(3|B))|((3|B)(9|H)(3|B)(7|F))|((3|B)(9|H)(7|F)(3|B))|((7|F)(3|B)(3|B)(9|H))|((7|F)(3|B)(9|H)(3|B))|((7|F)(9|H)(3|B)(3|B))|((9|H)(3|B)(3|B)(7|F))|((9|H)(3|B)(7|F)(3|B))|((9|H)(7|F)(3|B)(3|B))|((3|B)(4|C)(6|E)(9|H))|((3|B)(4|C)(9|H)(6|E))|((3|B)(6|E)(4|C)(9|H))|((3|B)(6|E)(9|H)(4|C))|((3|B)(9|H)(4|C)(6|E))|((3|B)(9|H)(6|E)(4|C))|((4|C)(3|B)(6|E)(9|H))|((4|C)(3|B)(9|H)(6|E))|((4|C)(6|E)(3|B)(9|H))|((4|C)(6|E)(9|H)(3|B))|((4|C)(9|H)(3|B)(6|E))|((4|C)(9|H)(6|E)(3|B))|((6|E)(3|B)(4|C)(9|H))|((6|E)(3|B)(9|H)(4|C))|((6|E)(4|C)(3|B)(9|H))|((6|E)(4|C)(9|H)(3|B))|((6|E)(9|H)(3|B)(4|C))|((6|E)(9|H)(4|C)(3|B))|((9|H)(3|B)(4|C)(6|E))|((9|H)(3|B)(6|E)(4|C))|((9|H)(4|C)(3|B)(6|E))|((9|H)(4|C)(6|E)(3|B))|((9|H)(6|E)(3|B)(4|C))|((9|H)(6|E)(4|C)(3|B))|((3|B)(4|C)(7|F)(8|G))|((3|B)(4|C)(8|G)(7|F))|((3|B)(7|F)(4|C)(8|G))|((3|B)(7|F)(8|G)(4|C))|((3|B)(8|G)(4|C)(7|F))|((3|B)(8|G)(7|F)(4|C))|((4|C)(3|B)(7|F)(8|G))|((4|C)(3|B)(8|G)(7|F))|((4|C)(7|F)(3|B)(8|G))|((4|C)(7|F)(8|G)(3|B))|((4|C)(8|G)(3|B)(7|F))|((4|C)(8|G)(7|F)(3|B))|((7|F)(3|B)(4|C)(8|G))|((7|F)(3|B)(8|G)(4|C))|((7|F)(4|C)(3|B)(8|G))|((7|F)(4|C)(8|G)(3|B))|((7|F)(8|G)(3|B)(4|C))|((7|F)(8|G)(4|C)(3|B))|((8|G)(3|B)(4|C)(7|F))|((8|G)(3|B)(7|F)(4|C))|((8|G)(4|C)(3|B)(7|F))|((8|G)(4|C)(7|F)(3|B))|((8|G)(7|F)(3|B)(4|C))|((8|G)(7|F)(4|C)(3|B))|((3|B)(4|C))|((3|B)(4|C))|((3|B)(4|C))|((4|C)(3|B))|((4|C)(3|B))|((4|C)(3|B))|((3|B)(4|C))|((3|B)(4|C))|((4|C)(3|B))|((4|C)(3|B))|((3|B)(4|C))|((4|C)(3|B))|((3|B)(5|D)(7|F))|((3|B)(5|D)(7|F))|((3|B)(7|F)(5|D))|((3|B)(7|F)(5|D))|((3|B)(5|D)(7|F))|((3|B)(7|F)(5|D))|((5|D)(3|B)(7|F))|((5|D)(3|B)(7|F))|((5|D)(7|F)(3|B))|((5|D)(7|F)(3|B))|((5|D)(3|B)(7|F))|((5|D)(7|F)(3|B))|((7|F)(3|B)(5|D))|((7|F)(3|B)(5|D))|((7|F)(5|D)(3|B))|((7|F)(5|D)(3|B))|((7|F)(3|B)(5|D))|((7|F)(5|D)(3|B))|((3|B)(5|D)(7|F))|((3|B)(7|F)(5|D))|((5|D)(3|B)(7|F))|((5|D)(7|F)(3|B))|((7|F)(3|B)(5|D))|((7|F)(5|D)(3|B))|((4|C)(4|C)(6|E)(8|G))|((4|C)(4|C)(8|G)(6|E))|((4|C)(6|E)(4|C)(8|G))|((4|C)(6|E)(8|G)(4|C))|((4|C)(8|G)(4|C)(6|E))|((4|C)(8|G)(6|E)(4|C))|((6|E)(4|C)(4|C)(8|G))|((6|E)(4|C)(8|G)(4|C))|((6|E)(8|G)(4|C)(4|C))|((8|G)(4|C)(4|C)(6|E))|((8|G)(4|C)(6|E)(4|C))|((8|G)(6|E)(4|C)(4|C))|((4|C)(5|D)(6|E))|((4|C)(5|D)(6|E))|((4|C)(6|E)(5|D))|((4|C)(6|E)(5|D))|((4|C)(5|D)(6|E))|((4|C)(6|E)(5|D))|((5|D)(4|C)(6|E))|((5|D)(4|C)(6|E))|((5|D)(6|E)(4|C))|((5|D)(6|E)(4|C))|((5|D)(4|C)(6|E))|((5|D)(6|E)(4|C))|((6|E)(4|C)(5|D))|((6|E)(4|C)(5|D))|((6|E)(5|D)(4|C))|((6|E)(5|D)(4|C))|((6|E)(4|C)(5|D))|((6|E)(5|D)(4|C))|((4|C)(5|D)(6|E))|((4|C)(6|E)(5|D))|((5|D)(4|C)(6|E))|((5|D)(6|E)(4|C))|((6|E)(4|C)(5|D))|((6|E)(5|D)(4|C))|((5|D)(5|D)(6|E)(7|F))|((5|D)(5|D)(7|F)(6|E))|((5|D)(6|E)(5|D)(7|F))|((5|D)(6|E)(7|F)(5|D))|((5|D)(7|F)(5|D)(6|E))|((5|D)(7|F)(6|E)(5|D))|((6|E)(5|D)(5|D)(7|F))|((6|E)(5|D)(7|F)(5|D))|((6|E)(7|F)(5|D)(5|D))|((7|F)(5|D)(5|D)(6|E))|((7|F)(5|D)(6|E)(5|D))|((7|F)(6|E)(5|D)(5|D))|((6|E)(6|E)(9|H)(9|H))|((6|E)(9|H)(6|E)(9|H))|((6|E)(9|H)(9|H)(6|E))|((9|H)(6|E)(6|E)(9|H))|((9|H)(6|E)(9|H)(6|E))|((9|H)(9|H)(6|E)(6|E))|((6|E)(7|F)(8|G)(9|H))|((6|E)(7|F)(9|H)(8|G))|((6|E)(8|G)(7|F)(9|H))|((6|E)(8|G)(9|H)(7|F))|((6|E)(9|H)(7|F)(8|G))|((6|E)(9|H)(8|G)(7|F))|((7|F)(6|E)(8|G)(9|H))|((7|F)(6|E)(9|H)(8|G))|((7|F)(8|G)(6|E)(9|H))|((7|F)(8|G)(9|H)(6|E))|((7|F)(9|H)(6|E)(8|G))|((7|F)(9|H)(8|G)(6|E))|((8|G)(6|E)(7|F)(9|H))|((8|G)(6|E)(9|H)(7|F))|((8|G)(7|F)(6|E)(9|H))|((8|G)(7|F)(9|H)(6|E))|((8|G)(9|H)(6|E)(7|F))|((8|G)(9|H)(7|F)(6|E))|((9|H)(6|E)(7|F)(8|G))|((9|H)(6|E)(8|G)(7|F))|((9|H)(7|F)(6|E)(8|G))|((9|H)(7|F)(8|G)(6|E))|((9|H)(8|G)(6|E)(7|F))|((9|H)(8|G)(7|F)(6|E))|((6|E)(9|H))|((6|E)(9|H))|((6|E)(9|H))|((9|H)(6|E))|((9|H)(6|E))|((9|H)(6|E))|((6|E)(9|H))|((6|E)(9|H))|((9|H)(6|E))|((9|H)(6|E))|((6|E)(9|H))|((9|H)(6|E))|((7|F)(7|F)(8|G)(8|G))|((7|F)(8|G)(7|F)(8|G))|((7|F)(8|G)(8|G)(7|F))|((8|G)(7|F)(7|F)(8|G))|((8|G)(7|F)(8|G)(7|F))|((8|G)(8|G)(7|F)(7|F))|((7|F)(8|G))|((7|F)(8|G))|((7|F)(8|G))|((8|G)(7|F))|((8|G)(7|F))|((8|G)(7|F))|((7|F)(8|G))|((7|F)(8|G))|((8|G)(7|F))|((8|G)(7|F))|((7|F)(8|G))|((8|G)(7|F))'
DITHERINGANY_9 = "{9}|".join(
    list(map(str, range(10))) + ['A', 'B', 'C', 'D', 'E', 'F']) + "{9}"
ENDURO_DITHERING = '(2|5|7){7}|(3|6|8){7}'
REACHER_REVISIT = '|'.join([
    '({0}{1}{0})'.format(a, b) for a, b in itertools.permutations(
        map(lambda x: hex(x)[2:], range(16)), 2)
])

CONSTRAINT_DICT = {'2d_dithering': lambda r: Constraint('2d_dithering', DITHERING2D_REGEX_4, r),
                 '2d_dithering_counting': lambda r: CountingPotentialConstraint('2d_dithering_counting', DITHERING2D_REGEX_4, r, 0.99),
                 '1d_dithering': lambda r: Constraint('1d_dithering', DITHERING1D_REGEX_k(2), r),
                 '1d_dithering_counting': lambda r: CountingPotentialConstraint('1d_dithering_counting', DITHERING1D_REGEX_k(2), r, 0.99),
                 '1d_actuation': lambda r: Constraint('1d_actuation', ACTUATION1D_REGEX_k(4), r),
                 'any_dithering_9': lambda r: Constraint('any_dithering_9', DITHERINGANY_9, r),
                 'enduro_dithering': lambda r: Constraint('enduro_dithering', ENDURO_DITHERING, r),
                 'reacher_revisit': lambda r: Constraint('reacher_revisit', REACHER_REVISIT, r, s_tl=reacher_state_quadrants, a_tl=reacher_action_threshold, a_active=False),
                 'reacher_revisit_counting': lambda r: CountingPotentialConstraint('reacher_revisit_counting', REACHER_REVISIT, r, 0.99, s_tl=reacher_state_quadrants, a_tl=reacher_action_threshold, a_active=False)}
