from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class SACAgentHybridSingleArm(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    
    Compared to SACAgent (in sac.py), this agent has a hybrid policy, with the gripper actions
    learned using DQN. Use this agent for single arm setups.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )
    
    def forward_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="grasp_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_grasp_critic(
        self,
        observations: Data, 
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_grasp_critic(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy( # type: ignore              
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    @jax.jit
    def get_encoded_features(self, observations: Data) -> jax.Array:
        """Extract the actor's fused latent representation for causal masking."""
        return self.state.apply_fn(
            {"params": self.state.params},
            observations,
            name="actor",
            method="get_features",
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _align_causal_weights(self, causal_weights, action_dim: int):
        causal_weights = jnp.asarray(causal_weights).reshape(-1)
        if causal_weights.shape[0] > action_dim:
            causal_weights = causal_weights[:action_dim]
        elif causal_weights.shape[0] < action_dim:
            causal_weights = jnp.pad(
                causal_weights,
                (0, action_dim - causal_weights.shape[0]),
                constant_values=1.0,
            )

        # Keep entropy magnitude comparable to the unweighted sum.
        weight_sum = jnp.maximum(jnp.sum(causal_weights), 1e-6)
        causal_weights = causal_weights * (float(action_dim) / weight_sum)
        return causal_weights.reshape(1, -1)

    def _align_grasp_causal_vector(self, grasp_causal_bias):
        bias = jnp.asarray(grasp_causal_bias, dtype=jnp.float32).reshape(-1)
        if bias.shape[0] > 3:
            bias = bias[:3]
        elif bias.shape[0] < 3:
            bias = jnp.pad(bias, (0, 3 - bias.shape[0]), constant_values=0.0)
        bias = jnp.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
        return bias.reshape(1, 3)

    def _extract_grasp_causal_state(self, observations):
        if isinstance(observations, dict) or hasattr(observations, "items"):
            if "state" in observations:
                state = observations["state"]
            else:
                state = jnp.concatenate(
                    [jnp.asarray(value).reshape(jnp.asarray(value).shape[0], -1)
                     for value in observations.values()],
                    axis=-1,
                )
        else:
            state = observations
        state = jnp.asarray(state, dtype=jnp.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        elif state.ndim > 2:
            state = state.reshape(state.shape[0], -1)
        return state

    def _align_grasp_causal_matrix(self, matrix, state_dim: int):
        matrix = jnp.asarray(matrix, dtype=jnp.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(3, -1)
        if matrix.shape[0] > 3:
            matrix = matrix[:3]
        elif matrix.shape[0] < 3:
            matrix = jnp.pad(matrix, ((0, 3 - matrix.shape[0]), (0, 0)))
        if matrix.shape[1] > state_dim:
            matrix = matrix[:, :state_dim]
        elif matrix.shape[1] < state_dim:
            matrix = jnp.pad(matrix, ((0, 0), (0, state_dim - matrix.shape[1])))
        return jnp.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def _align_grasp_causal_state_vector(self, values, state_dim: int, fill_value: float):
        values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
        if values.shape[0] > state_dim:
            values = values[:state_dim]
        elif values.shape[0] < state_dim:
            values = jnp.pad(values, (0, state_dim - values.shape[0]), constant_values=fill_value)
        return jnp.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)

    def _compute_grasp_causal_bias(self, observations, grasp_causal_bias):
        if grasp_causal_bias is None:
            return None
        if not isinstance(grasp_causal_bias, dict):
            return self._align_grasp_causal_vector(grasp_causal_bias)

        state = self._extract_grasp_causal_state(observations)
        state_dim = state.shape[-1]
        matrix = self._align_grasp_causal_matrix(
            grasp_causal_bias.get("matrix", jnp.zeros((3, state_dim), dtype=jnp.float32)),
            state_dim,
        )
        state_mean = self._align_grasp_causal_state_vector(
            grasp_causal_bias.get("state_mean", jnp.zeros((state_dim,), dtype=jnp.float32)),
            state_dim,
            0.0,
        )
        state_std = self._align_grasp_causal_state_vector(
            grasp_causal_bias.get("state_std", jnp.ones((state_dim,), dtype=jnp.float32)),
            state_dim,
            1.0,
        )
        state_std = jnp.where(jnp.abs(state_std) > 1e-6, state_std, 1.0)
        state_z = (state - state_mean.reshape(1, -1)) / state_std.reshape(1, -1)
        state_bias = state_z @ matrix.T
        state_bias = state_bias - jnp.mean(state_bias, axis=-1, keepdims=True)
        state_bias = jnp.clip(
            jnp.nan_to_num(state_bias, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

        fallback_bias = self._align_grasp_causal_vector(
            grasp_causal_bias.get("fallback_bias", jnp.zeros((3,), dtype=jnp.float32))
        )
        active_matrix = jnp.asarray(
            grasp_causal_bias.get("active_matrix", jnp.asarray([1.0], dtype=jnp.float32)),
            dtype=jnp.float32,
        ).reshape(-1)[0]
        active_matrix = jnp.clip(jnp.nan_to_num(active_matrix, nan=0.0), 0.0, 1.0)
        return active_matrix * state_bias + (1.0 - active_matrix) * fallback_bias

    def _sample_and_log_prob(self, dist, seed, causal_weights=None):
        if causal_weights is None:
            return dist.sample_and_log_prob(seed=seed)

        if hasattr(dist, "bijector"):
            base_dist = dist.distribution
            x_t = base_dist.sample(seed=seed)
            y_t = dist.bijector.forward(x_t)

            unsummed_base = distrax.Normal(base_dist.loc, base_dist.scale_diag).log_prob(x_t)
            unsummed_ildj = distrax.Tanh().forward_log_det_jacobian(x_t)
            unsummed_log_prob = unsummed_base - unsummed_ildj
            causal_weights = self._align_causal_weights(causal_weights, unsummed_log_prob.shape[-1])
            return y_t, (unsummed_log_prob * causal_weights).sum(axis=-1)

        x_t = dist.sample(seed=seed)
        unsummed_log_prob = distrax.Normal(dist.loc, dist.scale_diag).log_prob(x_t)
        causal_weights = self._align_causal_weights(causal_weights, unsummed_log_prob.shape[-1])
        return x_t, (unsummed_log_prob * causal_weights).sum(axis=-1)

    def _compute_next_actions(self, batch, rng, causal_weights=None):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )

        next_actions, next_actions_log_probs = self._sample_and_log_prob(
            next_action_distributions,
            seed=rng,
            causal_weights=causal_weights,
        )
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey, causal_weights=None):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = batch["actions"][..., :-1]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key, causal_weights=causal_weights
        )

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info
    

    def grasp_critic_loss_fn(
        self,
        batch,
        params: Params,
        rng: PRNGKey,
        grasp_causal_bias=None,
        grasp_causal_beta=0.0,
    ):
        """classes that inherit this class can change this function"""

        batch_size = batch["rewards"].shape[0]
        grasp_action = jnp.round(batch["actions"][..., -1]).astype(jnp.int16) + 1 # Cast env action from [-1, 1] to {0, 1, 2}

         # Evaluate next grasp Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 3))

        # Select target next grasp Q based on the gripper action that maximizes the current grasp Q
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        grasp_bias_active = grasp_causal_bias is not None
        if grasp_bias_active:
            grasp_bias = self._compute_grasp_causal_bias(
                batch["next_observations"], grasp_causal_bias
            )
            next_grasp_qs = next_grasp_qs + jnp.asarray(
                grasp_causal_beta, dtype=next_grasp_qs.dtype
            ) * grasp_bias
        # For DQN, select actions using online network, evaluate with target network
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1) 
        chex.assert_shape(best_next_grasp_action, (batch_size,))
        
        target_next_grasp_q = target_next_grasp_qs[jnp.arange(batch_size), best_next_grasp_action]
        chex.assert_shape(target_next_grasp_q, (batch_size,))

        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        chex.assert_shape(target_grasp_q, (batch_size,))

        # Forward pass through the online grasp critic to get predicted Q-values
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"], 
            rng=rng, 
            grad_params=params
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 3))
        
        # Select the predicted Q-values for the taken grasp actions in the batch
        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))
        
        # Compute MSE loss between predicted and target Q-values
        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q])
        grasp_critic_loss = jnp.mean((predicted_grasp_q - target_grasp_q) ** 2)

        info = {
            "grasp_critic_loss": grasp_critic_loss,
            "predicted_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_grasp_qs": jnp.mean(target_grasp_q),
            "grasp_rewards": grasp_rewards.mean(),
            "grasp_causal_bias_active": jnp.asarray(grasp_bias_active, dtype=jnp.float32),
        }
        if grasp_bias_active:
            info.update(
                {
                    "grasp_causal_beta": jnp.asarray(grasp_causal_beta, dtype=jnp.float32),
                    "grasp_causal_bias_open": jnp.mean(grasp_bias[..., 0]),
                    "grasp_causal_bias_stay": jnp.mean(grasp_bias[..., 1]),
                    "grasp_causal_bias_close": jnp.mean(grasp_bias[..., 2]),
                    "grasp_causal_bias_std": jnp.std(grasp_bias),
                }
            )

        return grasp_critic_loss, info


    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey, causal_weights=None):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = self._sample_and_log_prob(
            action_distributions,
            seed=sample_rng,
            causal_weights=causal_weights,
        )

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)
        entropy = -log_probs.mean()
        unweighted_entropy = -action_distributions.log_prob(actions).mean()

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": entropy,
            "causal_entropy_delta": entropy - unweighted_entropy,
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey, causal_weights=None):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key, causal_weights=causal_weights
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {
            "temperature_loss": temperature_loss,
            "temperature_entropy": entropy,
        }
    
    def loss_fns(
        self,
        batch,
        causal_weights=None,
        grasp_causal_bias=None,
        grasp_causal_beta=0.0,
    ):
        return {
            "critic": partial(self.critic_loss_fn, batch, causal_weights=causal_weights),
            "grasp_critic": partial(
                self.grasp_critic_loss_fn,
                batch,
                grasp_causal_bias=grasp_causal_bias,
                grasp_causal_beta=grasp_causal_beta,
            ),
            "actor": partial(self.policy_loss_fn, batch, causal_weights=causal_weights),
            "temperature": partial(self.temperature_loss_fn, batch, causal_weights=causal_weights),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
        **kwargs
    ) -> Tuple["SACAgentHybridSingleArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, self.config["action_dim"]))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        grasp_causal_bias=None,
        grasp_causal_beta=0.0,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            ee_actions = dist.mode()
        else:
            ee_actions = dist.sample(seed=seed)
        
        seed, grasp_key = jax.random.split(seed, 2)
        grasp_q_values = self.forward_grasp_critic(observations, rng=grasp_key, train=False)
        if grasp_causal_bias is not None:
            grasp_bias = self._compute_grasp_causal_bias(observations, grasp_causal_bias)
            if grasp_q_values.ndim == 1 and grasp_bias.ndim == 2 and grasp_bias.shape[0] == 1:
                grasp_bias = grasp_bias.reshape(-1)
            grasp_q_values = grasp_q_values + jnp.asarray(
                grasp_causal_beta, dtype=grasp_q_values.dtype
            ) * grasp_bias
        
        # Select grasp actions based on the grasp Q-values
        grasp_action = grasp_q_values.argmax(axis=-1)
        grasp_action = grasp_action - 1 # Mapping back to {-1, 0, 1}

        return jnp.concatenate([ee_actions, grasp_action[..., None]], axis=-1)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "grasp_critic": grasp_critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)

        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions[..., :-1]],
            grasp_critic=[observations],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -(actions.shape[-1] - 1) / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                action_dim=actions.shape[-1],
                ee_action_dim=actions.shape[-1] - 1,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        
        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
        )(name="grasp_critic")
        
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1]-1,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
