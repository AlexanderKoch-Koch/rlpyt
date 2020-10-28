from rlpyt.distributions.gaussian import Gaussian
import torch
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.agents.pg.mujoco import MujocoMixin, AlternatingRecurrentGaussianPgAgent
from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
                               AlternatingRecurrentAgentMixin)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.distributions.gaussian import DistInfoStd

DistInfo = namedarraytuple("DistInfo", ["mean", 'std'])


class VmpoAgent(RecurrentAgentMixin, BaseAgent):
    """
    Base class for gaussian vmpo agents. This version uses a Gaussian with diagonal covariance matrix. It expects a
    mu and std vector from the agent. The agent should have applied a softmax to the std to ensure positive values.
    Exp on the std output seems to workd slightly worse.
    """

    def initialize(self, env_spaces, *args, **kwargs):
        """Extends base method to build Gaussian distribution."""
        super().initialize(env_spaces, *args, **kwargs)
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # squash=env_spaces.action.high[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state),
                                 device=self.device)
        mu, std, value, rnn_state = self.model(*model_inputs)
        dist_info, value = buffer_to((DistInfoStd(mean=mu, log_std=std), value), device="cpu")
        return dist_info, value, rnn_state

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        agent_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mu, std, value, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        dist_info = DistInfoStd(mean=mu, log_std=std)
        # action = self.distribution.sample(dist_info) if self._mode == 'sample' else mu
        dist = torch.distributions.normal.Normal(loc=mu, scale=std)
        action = dist.sample() if self._mode == 'sample' else mu
        if self.prev_rnn_state is None:
            prev_rnn_state = buffer_func(rnn_state, torch.zeros_like)
        else:
            prev_rnn_state = self.prev_rnn_state

        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
                                  prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)


class MujocoVmpoAgent(MujocoMixin, VmpoAgent):
    pass


class AlternatingVmpoAgent(AlternatingRecurrentAgentMixin, MujocoVmpoAgent):
    pass
