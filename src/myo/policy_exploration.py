"""Keep gSDE noise amplitude independent of the actor latent-vector norm."""
import torch.nn.functional as functional
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class UnitLatentNoise(StateDependentNoiseDistribution):
    def proba_distribution(self, mean_actions, log_std, latent_sde):
        # Parent stores these same features for sampling AND the likelihood.
        # The actor mean still uses its original, unnormalized latent vector.
        return super().proba_distribution(
            mean_actions, log_std, functional.normalize(latent_sde, dim=1))


class UnitLatentSDEPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        if not self.use_sde:
            raise ValueError("UnitLatentSDEPolicy requires use_sde=True")
        self.action_dist = UnitLatentNoise(self.action_space.shape[0], **self.dist_kwargs)
        super()._build(lr_schedule)
