import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Setup path so causallearnmain from ACE can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
ace_dir = os.path.join(project_root, "ACE")

if project_root not in sys.path:
    sys.path.append(project_root)
if ace_dir not in sys.path:
    sys.path.append(ace_dir)

lingam_import_error = None
try:
    from causallearnmain.causallearn.search.FCMBased import lingam
except ImportError as exc:
    lingam_import_error = exc
    try:
        from ACE.causallearnmain.causallearn.search.FCMBased import lingam
    except ImportError as ace_exc:
        lingam_import_error = ace_exc


def _to_2d_features(array, name):
    """Convert replay buffer fields to (batch, features)."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


def _softmax_scaled(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = np.sum(exp_values)
    if not np.isfinite(denom) or denom <= 0:
        return np.ones_like(values, dtype=np.float32)
    return (exp_values / denom) * values.shape[0]


def _centered_unit(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    centered = values - np.mean(values)
    scale = np.max(np.abs(centered))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    return (centered / scale).astype(np.float32)


def _extract_sar(batch, action_indices=None):
    obs = batch["observations"]
    if isinstance(obs, dict) or hasattr(obs, 'items'):
        if "state" in obs:
            states = _to_2d_features(obs["state"], "state")
        else:
            states = np.concatenate(
                [_to_2d_features(value, key) for key, value in obs.items()],
                axis=-1,
            )
    else:
        states = _to_2d_features(obs, "observations")
    actions = _to_2d_features(batch["actions"], "actions")
    if action_indices is not None:
        actions = actions[:, np.asarray(action_indices, dtype=np.int64)]
    reward_key = "causal_rewards" if "causal_rewards" in batch else "rewards"
    rewards = _to_2d_features(batch[reward_key], reward_key)
    return states, actions, rewards


def _extract_states_actions(batch):
    obs = batch["observations"]
    if isinstance(obs, dict) or hasattr(obs, 'items'):
        if "state" in obs:
            states = _to_2d_features(obs["state"], "state")
        else:
            states = np.concatenate(
                [_to_2d_features(value, key) for key, value in obs.items()],
                axis=-1,
            )
    else:
        states = _to_2d_features(obs, "observations")
    actions = _to_2d_features(batch["actions"], "actions")
    return states, actions


def _variable_column_mask(values, eps=1e-6):
    """Select finite columns with enough variation for DirectLiNGAM."""
    std = np.std(np.asarray(values, dtype=np.float32), axis=0)
    return np.logical_and(np.isfinite(std), std > eps)


def _neutral_action_weights(action_dim, reason):
    print(f"Causal weights fallback to neutral values: {reason}")
    return np.ones(action_dim, dtype=np.float32), 0.0


def _compute_weight_from_sar(states, actions, rewards, sample_size, causal_method):
    actual_size = min(sample_size, states.shape[0], actions.shape[0], rewards.shape[0])

    states = states[:actual_size]
    actions = actions[:actual_size]
    rewards = rewards[:actual_size, :1]
    action_dim = actions.shape[1]

    raw_input = np.hstack((states, actions, rewards))
    if not np.all(np.isfinite(raw_input)):
        raise ValueError("causal input contains NaN or Inf values")
    if actual_size < 3:
        return _neutral_action_weights(action_dim, f"only {actual_size} samples are available")

    state_mask = _variable_column_mask(states)
    action_mask = _variable_column_mask(actions)
    reward_variable = bool(_variable_column_mask(rewards)[0])
    dropped_states = int(np.sum(~state_mask))
    dropped_actions = int(np.sum(~action_mask))
    if dropped_states or dropped_actions:
        print(
            "Causal preprocessing removed constant columns: "
            f"state={dropped_states}/{states.shape[1]}, "
            f"action={dropped_actions}/{action_dim}"
        )
    if not reward_variable:
        return _neutral_action_weights(action_dim, "reward has near-zero variance")
    if not np.any(action_mask):
        return _neutral_action_weights(action_dim, "all action dimensions have near-zero variance")

    filtered_states = states[:, state_mask]
    filtered_actions = actions[:, action_mask]
    X_ori = np.hstack((filtered_states, filtered_actions, rewards))
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))

    model_running_time = 0.0
    if causal_method == 'DirectLiNGAM':
        if "lingam" not in globals():
            raise ImportError(f"DirectLiNGAM import failed: {lingam_import_error}")
        start_time = time.time()
        model = lingam.DirectLiNGAM()
        with warnings.catch_warnings():
            # DirectLiNGAM may encounter zero residual variance for collinear
            # columns. Treat a non-finite graph as a neutral-weight fallback.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(X)
        end_time = time.time()
        model_running_time = end_time - start_time
        action_start = filtered_states.shape[1]
        active_weights = model.adjacency_matrix_[
            -1, action_start:(action_start + filtered_actions.shape[1])
        ]
        if not np.all(np.isfinite(active_weights)):
            return _neutral_action_weights(action_dim, "DirectLiNGAM returned non-finite action effects")
        weight_r = np.zeros(action_dim, dtype=np.float32)
        weight_r[action_mask] = active_weights
    else:
        raise ValueError(f"Unsupported causal_method: {causal_method}")

    weight = _softmax_scaled(weight_r)
    if not np.all(np.isfinite(weight)):
        return _neutral_action_weights(action_dim, "scaled action weights are non-finite")
    if np.std(weight_r) <= 1e-6:
        print(
            "Causal action effects are flat before scaling; "
            f"raw action effects={weight_r}. Returning neutral entropy weights.",
            flush=True,
        )
    print(f"Causal weights (scaled) for actions -> reward: {weight}, computed in {model_running_time:.4f} seconds")
    return weight, model_running_time


def get_sa2r_weight_from_batch(batch, causal_method='DirectLiNGAM', action_indices=None):
    """
    Computes causal weights from an already sampled/assembled batch.
    The batch only needs observations, actions, and rewards/causal_rewards.
    """
    states, actions, rewards = _extract_sar(batch, action_indices=action_indices)
    return _compute_weight_from_sar(
        states=states,
        actions=actions,
        rewards=rewards,
        sample_size=states.shape[0],
        causal_method=causal_method,
    )


def get_sa2r_weight(
    memory,
    demo_memory=None,
    sample_size=1000,
    causal_method='DirectLiNGAM',
    action_indices=None,
):
    """
    Computes causal weights using DirectLiNGAM on a given SERL/JaxRL memory replay buffer.
    Adapted from ACE/utilis/causal_weight.py.
    """
    if demo_memory is not None:
        batch_online = memory.sample(batch_size=sample_size // 2)
        batch_demo = demo_memory.sample(batch_size=sample_size - sample_size // 2)
        states_o, actions_o, rewards_o = _extract_sar(batch_online, action_indices=action_indices)
        states_d, actions_d, rewards_d = _extract_sar(batch_demo, action_indices=action_indices)
        
        states = np.concatenate([states_o, states_d], axis=0)
        actions = np.concatenate([actions_o, actions_d], axis=0)
        rewards = np.concatenate([rewards_o, rewards_d], axis=0)
    else:
        # In SERL, ReplayBuffer.sample(batch_size) returns a DatasetDict (dict-like)
        batch = memory.sample(batch_size=sample_size)
        states, actions, rewards = _extract_sar(batch, action_indices=action_indices)

    return _compute_weight_from_sar(
        states=states,
        actions=actions,
        rewards=rewards,
        sample_size=sample_size,
        causal_method=causal_method,
    )


def get_sa2r_weight_demo_only(
    demo_memory,
    sample_size=1000,
    causal_method='DirectLiNGAM',
    action_indices=None,
):
    """
    Computes causal weights ONLY from the demo/intervention buffer memory.
    This helps to find the causal matrix solely based on the expert/intervention data
    without being polluted by random online exploration noise.
    """
    # Reuse the original function by passing demo_memory as the sole buffer
    return get_sa2r_weight(
        memory=demo_memory,
        demo_memory=None,
        sample_size=sample_size,
        causal_method=causal_method,
        action_indices=action_indices,
    )


def get_obs2grasp_bias_demo_only(
    demo_memory,
    sample_size=1000,
    causal_method='DirectLiNGAM',
    grasp_action_index=-1,
):
    """
    Estimate a state-conditioned 3-way additive gripper-action bias from
    demo/intervention data.

    The returned payload is ordered as gripper classes {open, stay, close},
    matching the hybrid grasp critic's internal action indices {0, 1, 2}.
    Callers compute a per-observation score:
        bias(s) = center((state - mean) / std @ matrix.T)
        Q_eff = Q_grasp + beta * bias(s)

    If the learned linear graph is flat, the payload marks active_matrix=0 and
    includes a conservative frequency fallback for backward-compatible behavior.
    """
    batch = demo_memory.sample(batch_size=sample_size)
    states, actions = _extract_states_actions(batch)

    actual_size = min(sample_size, states.shape[0], actions.shape[0])
    states = states[:actual_size]
    actions = actions[:actual_size]
    if actions.shape[1] == 0:
        raise ValueError("grasp causal bias needs at least one action dimension")

    grasp_actions = np.rint(actions[:, grasp_action_index]).astype(np.int32)
    grasp_classes = np.clip(grasp_actions + 1, 0, 2)
    counts = np.bincount(grasp_classes, minlength=3).astype(np.float32)
    one_hot = np.eye(3, dtype=np.float32)[grasp_classes]

    state_mean = states.mean(axis=0).astype(np.float32)
    state_std = states.std(axis=0).astype(np.float32)
    state_std = np.where(state_std > 1e-6, state_std, 1.0).astype(np.float32)
    states_z = ((states - state_mean) / state_std).astype(np.float32)

    X_ori = np.hstack((states_z, one_hot))
    if not np.all(np.isfinite(X_ori)):
        raise ValueError("grasp causal input contains NaN or Inf values")

    model_running_time = 0.0
    raw_effect = np.zeros(3, dtype=np.float32)
    matrix = np.zeros((3, states.shape[1]), dtype=np.float32)
    mean_bias = np.zeros(3, dtype=np.float32)
    score_scale = 1.0
    active_matrix = 0.0
    method_used = causal_method
    try:
        if causal_method == 'DirectLiNGAM':
            if "lingam" not in globals():
                raise ImportError(f"DirectLiNGAM import failed: {lingam_import_error}")
            start_time = time.time()
            model = lingam.DirectLiNGAM()
            model.fit(pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1]))))
            model_running_time = time.time() - start_time
            state_dim = states.shape[1]
            grasp_rows = model.adjacency_matrix_[state_dim:(state_dim + 3), :state_dim]
            raw_effect = np.sum(np.abs(grasp_rows), axis=1).astype(np.float32)
            score_samples = states_z @ grasp_rows.T
            score_samples = score_samples - np.mean(score_samples, axis=1, keepdims=True)
            finite_scores = np.abs(score_samples[np.isfinite(score_samples)])
            if finite_scores.size > 0:
                score_scale = float(np.percentile(finite_scores, 95))
            if not np.isfinite(score_scale) or score_scale <= 1e-6:
                score_scale = 1.0
            matrix = (grasp_rows / score_scale).astype(np.float32)
            mean_bias = np.mean(
                np.clip(score_samples / score_scale, -1.0, 1.0),
                axis=0,
            ).astype(np.float32)
            active_matrix = float(np.any(np.abs(matrix) > 1e-6))
        else:
            raise ValueError(f"Unsupported causal_method: {causal_method}")
    except Exception as exc:
        method_used = f"frequency_fallback:{type(exc).__name__}"
        raw_effect = np.zeros(3, dtype=np.float32)
        matrix = np.zeros((3, states.shape[1]), dtype=np.float32)
        mean_bias = np.zeros(3, dtype=np.float32)
        active_matrix = 0.0

    fallback_bias = _centered_unit(raw_effect)
    if active_matrix <= 0.0 and not np.any(np.abs(fallback_bias) > 1e-6):
        # If the causal graph is uninformative, keep a conservative empirical prior.
        freq_prior = np.log((counts + 1.0) / (np.mean(counts + 1.0)))
        fallback_bias = _centered_unit(freq_prior)
        if method_used == causal_method:
            method_used = "frequency_fallback:flat_causal_effect"
        active_matrix = 0.0

    bias_payload = {
        "matrix": matrix.astype(np.float32),
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "fallback_bias": fallback_bias.astype(np.float32),
        "mean_bias": mean_bias.astype(np.float32),
        "active_matrix": np.asarray([active_matrix], dtype=np.float32),
    }

    info = {
        "compute_time_s": float(model_running_time),
        "sample_size": int(actual_size),
        "method": method_used,
        "active_matrix": float(active_matrix),
        "score_scale": float(score_scale),
        "mean_bias_open": float(mean_bias[0]),
        "mean_bias_stay": float(mean_bias[1]),
        "mean_bias_close": float(mean_bias[2]),
        "fallback_bias_open": float(fallback_bias[0]),
        "fallback_bias_stay": float(fallback_bias[1]),
        "fallback_bias_close": float(fallback_bias[2]),
        "raw_effect_open": float(raw_effect[0]),
        "raw_effect_stay": float(raw_effect[1]),
        "raw_effect_close": float(raw_effect[2]),
        "count_open": float(counts[0]),
        "count_stay": float(counts[1]),
        "count_close": float(counts[2]),
    }
    print(
        "State-conditioned grasp causal bias mean {open, stay, close}: "
        f"{mean_bias}, fallback={fallback_bias}, active_matrix={active_matrix}, "
        f"method={method_used}, computed in {model_running_time:.4f} seconds"
    )
    return bias_payload, info
