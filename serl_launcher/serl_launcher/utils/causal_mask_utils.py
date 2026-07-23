from functools import partial
import math
from typing import Mapping, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class LocalCausalModel(nn.Module):
    """Gaussian dynamics model for P(z_next | z, action)."""

    hidden_dims: Tuple[int, ...] = (512, 512)
    state_dim: int = 0

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            
        mean = nn.Dense(self.state_dim)(x)
        logvar = nn.Dense(self.state_dim)(x)
        return mean, jnp.clip(logvar, -10.0, 2.0)


class CausalTrainState(train_state.TrainState):
    pass


def create_causal_state(rng, state_dim, action_dim, lr=1e-3):
    model = LocalCausalModel(state_dim=state_dim)
    params = model.init(
        rng,
        jnp.zeros((1, state_dim)),
        jnp.zeros((1, action_dim)),
    )["params"]
    return CausalTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
    )


@jax.jit
def causal_model_step(state: CausalTrainState, obs, actions, next_obs):
    def loss_fn(params):
        mean, logvar = state.apply_fn({"params": params}, obs, actions)
        var = jnp.exp(logvar)
        return 0.5 * jnp.mean((next_obs - mean) ** 2 / var + logvar)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def causal_model_metrics(state: CausalTrainState, obs, actions, next_obs):
    """Evaluate one-step latent prediction without updating model parameters."""
    mean, logvar = state.apply_fn({"params": state.params}, obs, actions)
    var = jnp.exp(logvar)
    return {
        "nll": 0.5 * jnp.mean((next_obs - mean) ** 2 / var + logvar),
        "mse": jnp.mean((next_obs - mean) ** 2),
    }


def evaluate_causal_model(
    state,
    buffer,
    encoder_fn=None,
    batch_size=256,
    sharding=None,
):
    """Evaluate the dynamics model on an independently sampled validation batch."""
    if len(buffer) == 0:
        return {}

    batch_size = min(batch_size, len(buffer))
    batch = buffer.sample(batch_size)
    if sharding is not None:
        batch = jax.device_put(batch, device=sharding)
    observations = batch["observations"]
    if encoder_fn is not None:
        obs = encoder_fn(observations)
        next_obs = encoder_fn(batch["next_observations"])
    elif isinstance(observations, Mapping) and "state" in observations:
        obs = observations["state"]
        next_obs = batch["next_observations"]["state"]
    else:
        obs = observations
        next_obs = batch["next_observations"]
    return causal_model_metrics(state, obs, batch["actions"], next_obs)


def train_causal_model(
    state,
    buffer,
    encoder_fn=None,
    num_steps=1000,
    batch_size=256,
    sharding=None,
    return_info=False,
):
    """Fit the local dynamics model from replay transitions."""
    if len(buffer) == 0 or num_steps <= 0:
        return (state, {}) if return_info else state

    batch_size = min(batch_size, len(buffer))
    iterator = buffer.get_iterator(
        sample_args={"batch_size": batch_size, "pack_obs_and_next_obs": False},
        device=sharding,
    )
    last_loss = None
    for _ in range(num_steps):
        batch = next(iterator)
        observations = batch["observations"]
        if encoder_fn is not None:
            obs = encoder_fn(observations)
            next_obs = encoder_fn(batch["next_observations"])
        elif isinstance(observations, Mapping) and "state" in observations:
            obs = observations["state"]
            next_obs = batch["next_observations"]["state"]
        else:
            obs = observations
            next_obs = batch["next_observations"]
        state, last_loss = causal_model_step(state, obs, batch["actions"], next_obs)
    if return_info:
        return state, {"loss": last_loss}
    return state


def train_causal_model_from_buffers(
    state,
    buffers,
    ratios,
    encoder_fn=None,
    num_steps=1000,
    batch_size=256,
    sharding=None,
    return_info=False,
):
    """Fit the local dynamics model from a ratio-controlled mixture of buffers."""
    active_sources = [
        (buffer, float(ratio))
        for buffer, ratio in zip(buffers, ratios)
        if buffer is not None and len(buffer) > 0 and float(ratio) > 0.0
    ]
    if not active_sources or num_steps <= 0:
        return (state, {}) if return_info else state

    ratio_sum = sum(ratio for _, ratio in active_sources)
    source_batch_sizes = []
    remaining = int(batch_size)
    for index, (_, ratio) in enumerate(active_sources):
        if index == len(active_sources) - 1:
            source_bs = remaining
        else:
            source_bs = int(round(batch_size * ratio / ratio_sum))
            source_bs = max(1, min(remaining - (len(active_sources) - index - 1), source_bs))
        source_batch_sizes.append(source_bs)
        remaining -= source_bs

    iterators = [
        buffer.get_iterator(
            sample_args={
                "batch_size": source_bs,
                "pack_obs_and_next_obs": False,
            },
            device=sharding,
        )
        for (buffer, _), source_bs in zip(active_sources, source_batch_sizes)
    ]

    last_loss = None
    for _ in range(num_steps):
        batches = [next(iterator) for iterator in iterators]
        batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batches)
        observations = batch["observations"]
        if encoder_fn is not None:
            obs = encoder_fn(observations)
            next_obs = encoder_fn(batch["next_observations"])
        elif isinstance(observations, Mapping) and "state" in observations:
            obs = observations["state"]
            next_obs = batch["next_observations"]["state"]
        else:
            obs = observations
            next_obs = batch["next_observations"]
        state, last_loss = causal_model_step(state, obs, batch["actions"], next_obs)

    if return_info:
        info = {
            "loss": last_loss,
            "active_sources": len(active_sources),
        }
        for index, ((buffer, ratio), source_bs) in enumerate(
            zip(active_sources, source_batch_sizes)
        ):
            info[f"source_{index}_size"] = len(buffer)
            info[f"source_{index}_ratio"] = ratio / ratio_sum
            info[f"source_{index}_batch_size"] = source_bs
        return state, info
    return state


@partial(
    jax.jit,
    static_argnames=("apply_fn", "action_dim", "n_action_samples", "return_scores"),
)
def compute_cmi_masks_jax(
    params,
    apply_fn,
    obs,
    action_dim,
    key,
    threshold=0.1,
    n_action_samples=64,
    return_scores=False,
):
    """Return a mask for latent dimensions that are weakly action-dependent."""
    keys = jax.random.split(key, obs.shape[0])

    def eval_state_actions(state, action_key):
        actions = jax.random.uniform(
            action_key,
            (n_action_samples, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
        repeated_state = jnp.repeat(state[None, :], n_action_samples, axis=0)
        mean, logvar = apply_fn({"params": params}, repeated_state, actions)
        return mean, jnp.exp(logvar)

    means, variances = jax.vmap(eval_state_actions)(obs, keys)
    marginal_means = jnp.mean(means, axis=1, keepdims=True)
    marginal_variances = jnp.mean(variances, axis=1, keepdims=True)
    kls = 0.5 * (
        variances / marginal_variances
        + (means - marginal_means) ** 2 / marginal_variances
        - 1
        + jnp.log(marginal_variances)
        - jnp.log(variances)
    )
    scores = jnp.mean(jnp.clip(kls, min=0.0), axis=1)
    masks = scores < threshold
    return (masks, scores) if return_scores else masks


def _replace(mapping, **updates):
    if hasattr(mapping, "copy"):
        try:
            return mapping.copy(add_or_replace=updates)
        except TypeError:
            pass
    return {**mapping, **updates}


def add_encoded_features(batch, encoded_features, next_encoded_features, *, use_encoded):
    """Attach latent overrides while preserving FrozenDict inputs."""
    batch_size = encoded_features.shape[0]
    observations = _replace(
        batch["observations"],
        encoded_features=encoded_features,
        is_encoded=jnp.full((batch_size,), use_encoded, dtype=bool),
    )
    next_observations = _replace(
        batch["next_observations"],
        encoded_features=next_encoded_features,
        is_encoded=jnp.full((batch_size,), use_encoded, dtype=bool),
    )
    return _replace(
        batch,
        observations=observations,
        next_observations=next_observations,
    )


def select_lowest_score_mask(scores, ratio):
    """Select the lowest-scoring ratio of latent dimensions per sample."""
    latent_dim = scores.shape[-1]
    selected_count = min(latent_dim, max(0, int(math.floor(latent_dim * ratio))))
    if selected_count <= 0:
        return jnp.zeros_like(scores, dtype=bool)
    if selected_count >= latent_dim:
        return jnp.ones_like(scores, dtype=bool)

    selected = jnp.argsort(scores, axis=-1)[..., :selected_count]
    mask = jnp.zeros_like(scores, dtype=bool)
    row_indices = jnp.arange(scores.shape[0])[:, None]
    return mask.at[row_indices, selected].set(True)


def augment_latent_features(
    base_z,
    selected_mask,
    *,
    mask_baseline="mean",
    mask_strength=0.2,
):
    """Apply causal masking to selected latent dimensions."""
    if mask_baseline == "mean":
        baseline = jnp.mean(base_z, axis=0, keepdims=True)
    elif mask_baseline == "zero":
        baseline = jnp.zeros_like(base_z)
    else:
        raise ValueError(f"Unsupported causal mask baseline: {mask_baseline}")
    augmented_z = base_z + mask_strength * (baseline - base_z)
    return jnp.where(selected_mask, augmented_z, base_z)


def causal_mask_counterfactuals(
    batch_demo,
    z_demo=None,
    z_next_demo=None,
    selected_latent_mask=None,
    mask_baseline="mean",
    mask_strength=0.2,
):
    """Build demo-labelled counterfactual transitions with causal masking in the latent space."""
    if z_demo is not None:
        return add_encoded_features(
            batch_demo,
            augment_latent_features(
                z_demo,
                selected_latent_mask,
                mask_baseline=mask_baseline,
                mask_strength=mask_strength,
            ),
            augment_latent_features(
                z_next_demo,
                selected_latent_mask,
                mask_baseline=mask_baseline,
                mask_strength=mask_strength,
            ),
            use_encoded=True,
        )
    raise ValueError("Causal mask augmentation requires encoded latent features")
