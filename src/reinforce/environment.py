from src.allocator import Allocator
from src.op import Operator

class Environment():
    def __init__(self):
        self.reset()

def reset(self):
    self.allocator = Allocator()
    self.operators = [Operator("Operator 1"), Operator("Operator 2")]
    state = [allocator.a, operators[0].get_request(), operators[1].get_request()]
    return state

def step(self, action):
    """
    Agent performs the action
    """
    def action_to_spectrum(action_idx):
        """
            Converts the action from the discrete action space to a continuous spectrum allocation
            for operator 1 and operator 2.  
        """
        p1 = [1, 0.75, 0.5, 0.25, 0] # Percent of spectrum allocated to operator 1
        p1 = p1[action_idx]
        p2 = 1 - p1  # Percent of spectrum allocated to operator 2
        s1 = p1 * self.spectrum_size 
        s2 = p2 * self.spectrum_size
        return [s1, s2] 

    s = action_to_spectrum(action_idx)
    reward = 0
    for i, op in enumerate(operators):
        op.spectrum_size += s[i]
        op.rr_schedule(t)
        reward += op.get_reward(t)
        op.spectrum_size -= s[i]

    next_state = self.get_state(operators)
    return next_state, reward
