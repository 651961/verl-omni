# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 DiffusionART contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ART tensor kernels adapted from https://github.com/vanzll/DiffusionART.

Statistics and loss arithmetic use FP32; model GEMMs may use BF16.
"""

import torch


def _validate_scalar(value: float, name: str, *, positive: bool = False) -> float:
    """Validate the sign of a scalar hyperparameter."""
    value = float(value)
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _validate_batch(value: torch.Tensor, name: str) -> None:
    """Require a nonempty leading batch and at least one feature dimension."""
    if value.ndim < 2 or value.shape[0] == 0 or value.numel() == 0:
        raise ValueError(f"{name} must have nonempty batch and feature dimensions")


def _validate_shapes(reference: torch.Tensor, **values: torch.Tensor) -> None:
    """Validate prediction-space tensor shapes."""
    for name, value in values.items():
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}")


def _batch_vector(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Accept one value per item, including explicit singleton feature axes."""
    if value.ndim == 0 or value.numel() != reference.shape[0] or value.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} must contain one value per item")
    return value.detach().to(device=reference.device).reshape(-1)


def _expand_batch(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast per-item coefficients over feature dimensions."""
    return value.reshape((-1,) + (1,) * (reference.ndim - 1))


def schedule_coefficient(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Broadcast per-item or per-feature schedule coefficients without merging samples."""
    if value.ndim > 0 and value.numel() == reference.shape[0]:
        return _expand_batch(_batch_vector(value, reference, name), reference).float()
    if (
        value.ndim != reference.ndim
        or value.shape[0] != reference.shape[0]
        or any(size not in (1, target) for size, target in zip(value.shape[1:], reference.shape[1:], strict=True))
    ):
        raise ValueError(f"{name} must contain per-item or broadcastable per-feature coefficients")
    return value.detach().to(device=reference.device, dtype=torch.float32)


def art_loss(
    current_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_direction: torch.Tensor,
    guidance_magnitude: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
    ode_velocity_coefficient: torch.Tensor | None = None,
    current_ode_mean: torch.Tensor | None = None,
    old_ode_mean: torch.Tensor | None = None,
    loss_space: str = "velocity_linear_sigma",
    direction_normalization: str = "rms",
    mask_space: str = "velocity",
    velocity_mask_metric: str = "scheduler_reference_metric",
    corrective_mask_enabled: bool = True,
    movement_mask_threshold: float = 1e-5,
    noise_normalized_v_radius: float = 0.02,
    q2_reference_sde_noise_level: float = 0.8,
    q2_reference_kl_threshold: float = 1e-5,
    restore_anchor_mask: torch.Tensor | None = None,
    target_projection_radius: float = 0.0,
    reference_prediction: torch.Tensor | None = None,
    reference_mse_beta: float = 0.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Return per-item losses; all data except current predictions are frozen.

    Supports x0, uniform velocity, and sigma-weighted velocity MSE. The mask
    removes large outward moves but permits target-correcting moves. Inactive
    rows train only if explicitly requested as restoring anchors. Sigma and ODE
    coefficients may vary by feature for joint modalities with separate clocks;
    weighting and scheduler metrics are reduced after applying those coefficients.
    """
    _validate_batch(current_velocity, "current_velocity")
    _validate_shapes(current_velocity, old_velocity=old_velocity, guidance_direction=guidance_direction)
    if loss_space not in ("x0", "velocity_uniform", "velocity_linear_sigma"):
        raise ValueError("unsupported ART loss_space")
    if direction_normalization not in ("rms", "none"):
        raise ValueError("direction_normalization must be rms or none")
    if mask_space not in ("velocity", "ode_mean"):
        raise ValueError("mask_space must be velocity or ode_mean")
    metrics = ("fixed_mse", "noise_normalized_v", "q2_sde_kl", "scheduler_reference_metric")
    if velocity_mask_metric not in metrics:
        raise ValueError("unsupported velocity_mask_metric")
    if mask_space == "ode_mean" and velocity_mask_metric != "fixed_mse":
        raise ValueError("ode_mean masking requires fixed_mse")
    eps = _validate_scalar(eps, "eps", positive=True)
    threshold = _validate_scalar(movement_mask_threshold, "movement_mask_threshold")
    radius = _validate_scalar(noise_normalized_v_radius, "noise_normalized_v_radius", positive=True)
    noise = _validate_scalar(q2_reference_sde_noise_level, "q2_reference_sde_noise_level", positive=True)
    kl_threshold = _validate_scalar(q2_reference_kl_threshold, "q2_reference_kl_threshold")
    projection = _validate_scalar(target_projection_radius, "target_projection_radius")
    reference_beta = _validate_scalar(reference_mse_beta, "reference_mse_beta")
    if projection and loss_space != "x0":
        raise ValueError("target projection is supported only in x0 space")
    v = current_velocity.float()
    old = old_velocity.detach().to(v)
    s = schedule_coefficient(sigma, v, "sigma")
    magnitude = _batch_vector(guidance_magnitude, v, "guidance_magnitude").float()
    if ((s < 0) | (s > 1)).any():
        raise ValueError("sigma must be in [0, 1]")
    if (magnitude < 0).any():
        raise ValueError("guidance_magnitude must be nonnegative")
    raw = guidance_direction.detach().to(v)
    raw_rms = raw.square().flatten(1).mean(1).sqrt()
    direction = raw / _expand_batch(raw_rms.clamp_min(eps), v) if direction_normalization == "rms" else raw
    displacement = _expand_batch(magnitude, v) * direction
    weight = s if loss_space == "velocity_linear_sigma" else torch.ones_like(s)
    if loss_space == "x0":
        if state is None:
            raise ValueError("x0 loss requires state")
        _validate_shapes(v, state=state)
        x = state.detach().to(v)
        if projection:
            displacement = displacement * _expand_batch(
                (projection / displacement.square().flatten(1).mean(1).sqrt().clamp_min(eps)).clamp_max(1),
                v,
            )
        target = (x - s * old + displacement).detach()
        residual = target - (x - s * v)
        desired_step = -s * residual.detach()
    else:
        target = (old - displacement).detach()
        residual = target - v
        desired_step = residual.detach()
    unmasked = (weight * residual.square()).flatten(1).mean(1)
    movement = v - old
    energy = movement.square().flatten(1).mean(1)
    alignment = (movement * desired_step).flatten(1).mean(1)
    if mask_space == "ode_mean" or velocity_mask_metric in (
        "q2_sde_kl",
        "scheduler_reference_metric",
    ):
        if ode_velocity_coefficient is None:
            raise ValueError("this mask requires ode_velocity_coefficient")
        dt = schedule_coefficient(ode_velocity_coefficient, v, "ode_velocity_coefficient")
        if (dt >= 0).any():
            raise ValueError("ODE coefficients must be negative")
        if mask_space == "ode_mean":
            if current_ode_mean is None or old_ode_mean is None:
                raise ValueError("ode_mean masking requires current and old ODE means")
            _validate_shapes(v, current_ode_mean=current_ode_mean, old_ode_mean=old_ode_mean)
            mean_movement = current_ode_mean.float() - old_ode_mean.detach().to(v)
            energy = mean_movement.square().flatten(1).mean(1)
            alignment = (mean_movement * dt * desired_step).flatten(1).mean(1)
        else:
            initial = torch.isclose(s, torch.ones_like(s), atol=eps, rtol=0)
            std = torch.where(
                initial,
                torch.full_like(s, noise),
                (noise**2 * s / (1 - s).clamp_min(eps) * (-dt)).clamp_min(0).sqrt(),
            )
            coefficient = torch.where(initial, dt, dt * (1 + 0.5 * noise**2))
            energy = (
                (coefficient.square() * movement.square() / (2 * std.square().clamp_min(eps**2))).flatten(1).mean(1)
            )
            threshold = kl_threshold
    elif velocity_mask_metric == "noise_normalized_v":
        energy = torch.where(s > 0, movement.square() / torch.where(s > 0, s, 1.0), 0.0).flatten(1).mean(1)
        threshold = radius**2
    positive_sigma = (s > 0).flatten(1).all(1)
    active = (magnitude > 0) & (raw_rms > 0) & positive_sigma
    requested_anchor = (
        torch.zeros_like(active)
        if restore_anchor_mask is None
        else _batch_vector(restore_anchor_mask, v, "restore_anchor_mask").bool()
    )
    anchor = requested_anchor & ~active & positive_sigma
    masked = (active & corrective_mask_enabled & (energy >= threshold) & (alignment > 0)).detach()
    keep = (anchor | (active & ~masked)).detach()
    policy = torch.where(keep, unmasked, torch.zeros_like(unmasked))
    reference = torch.zeros_like(policy)
    if reference_prediction is not None:
        _validate_shapes(v, reference_prediction=reference_prediction)
        reference = (v - reference_prediction.detach().to(v)).square().flatten(1).mean(1)
    elif reference_beta:
        raise ValueError("nonzero reference_mse_beta requires reference_prediction")
    return {
        "per_item_loss": policy + reference_beta * reference,
        "per_item_policy_loss": policy,
        "per_item_reference_mse": reference,
        "per_item_unmasked_loss": unmasked,
        "target": target,
        "keep_mask": keep,
        "masked": masked,
        "movement_score": energy.detach(),
        "outward_alignment": alignment.detach(),
    }


def art_feature_group_loss(
    current_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_direction: torch.Tensor,
    guidance_magnitude: torch.Tensor,
    *,
    feature_group_sizes: torch.Tensor,
    **options,
) -> dict[str, torch.Tensor]:
    """Apply ART independently to feature partitions, returning summaries [batch, groups].

    Each group gets its own covariance magnitude, RMS, corrective mask and feature
    mean. Averaging these summaries weights groups equally, regardless of size.
    Padding beyond the declared groups never enters the objective.
    """
    _validate_batch(current_velocity, "current_velocity")
    _validate_shapes(current_velocity, old_velocity=old_velocity, guidance_direction=guidance_direction)
    sizes = feature_group_sizes.detach().cpu()
    if (
        sizes.ndim != 2
        or sizes.shape[0] != current_velocity.shape[0]
        or sizes.shape[1] == 0
        or sizes.shape != guidance_magnitude.shape
        or sizes.dtype not in (torch.int32, torch.int64)
        or not (sizes > 0).all()
        or (sizes.sum(1) > current_velocity.shape[1]).any()
    ):
        raise ValueError("ART feature group sizes and magnitudes must have shape [batch, groups] and valid sizes")
    sigma = schedule_coefficient(sigma, current_velocity, "sigma").expand_as(current_velocity)
    feature_options = {"state", "current_ode_mean", "old_ode_mean", "reference_prediction"}
    results = []
    for row, counts in enumerate(sizes.tolist()):
        offset = 0
        row_results = []
        for group, count in enumerate(counts):
            selection = (slice(row, row + 1), slice(offset, offset + count))
            kwargs = dict(options)
            for key in feature_options:
                if kwargs.get(key) is not None:
                    kwargs[key] = kwargs[key][selection]
            if kwargs.get("ode_velocity_coefficient") is not None:
                kwargs["ode_velocity_coefficient"] = schedule_coefficient(
                    kwargs["ode_velocity_coefficient"], current_velocity, "ode_velocity_coefficient"
                ).expand_as(current_velocity)[selection]
            if kwargs.get("restore_anchor_mask") is not None:
                kwargs["restore_anchor_mask"] = kwargs["restore_anchor_mask"][row : row + 1, group]
            result = art_loss(
                current_velocity[selection],
                old_velocity[selection],
                sigma[selection],
                guidance_direction[selection],
                guidance_magnitude[row : row + 1, group],
                **kwargs,
            )
            row_results.append(result)
            offset += count
        results.append(row_results)
    return {
        key: torch.stack([torch.cat([result[key] for result in row]) for row in results])
        for key in results[0][0]
        if key != "target"
    }


@torch.no_grad()
def art_scalar_covariance(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    row_group_indices: torch.Tensor,
    group_count: int,
    *,
    reduction: str = "unbiased",
    direction_mode: str = "historical",
    scale_mode: str = "unit_normalized_covariance",
    guidance_gain: float = 1.0,
    zero_signal_mode: str = "historical",
    zero_signal_atol: float = 0.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Complete-group covariance, never a microbatch estimator.

    Historical direction reduction divides covariance by reward MAD before
    RMS normalization with an epsilon floor. ``normalized_aggregate`` instead
    normalizes the aggregate covariance to exact unit RMS when active.
    ``exact`` clears roundoff from exactly constant rewards without a cutoff.
    ``threshold`` zero handling explicitly gates raw covariance and reward range;
    it is not silently applied to the historical fp32 computation.
    """
    _validate_batch(endpoints, "endpoints")
    eps = _validate_scalar(eps, "eps", positive=True)
    gain = _validate_scalar(guidance_gain, "guidance_gain")
    atol = _validate_scalar(zero_signal_atol, "zero_signal_atol")
    if reduction not in ("unbiased", "population"):
        raise ValueError("reduction must be unbiased or population")
    if direction_mode not in ("historical", "normalized_aggregate"):
        raise ValueError("unknown direction_mode")
    if zero_signal_mode not in ("historical", "exact", "threshold"):
        raise ValueError("unknown zero_signal_mode")
    if (zero_signal_mode in ("historical", "exact") and atol != 0) or (zero_signal_mode == "threshold" and atol <= 0):
        raise ValueError("threshold mode requires positive atol; historical/exact require zero")
    if scale_mode not in (
        "reward_mad",
        "raw_covariance",
        "reward_std_normalized_covariance",
        "unit_normalized_covariance",
    ):
        raise ValueError("unknown scale_mode")
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    if row_group_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("row_group_indices must be integer indices")
    x = endpoints.detach().float().flatten(1)
    r = rewards.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
    groups = row_group_indices.to(device=x.device, dtype=torch.long).reshape(-1)
    if r.numel() != len(x) or groups.numel() != len(x):
        raise ValueError("rewards and groups must match the endpoint count")
    if (groups < 0).any() or (groups >= group_count).any():
        raise ValueError("group index out of range")
    counts = torch.bincount(groups, minlength=group_count).float()
    sums = x.new_zeros(group_count).index_add_(0, groups, r)
    square_sums = x.new_zeros(group_count).index_add_(0, groups, r.square())
    means = sums / counts.clamp_min(1)
    stds = (square_sums / counts.clamp_min(1) - means.square()).clamp_min(0).sqrt()
    centered = r - means[groups]
    if zero_signal_mode == "exact":
        maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(0, groups, r, reduce="amax")
        minima = r.new_full((group_count,), torch.inf).scatter_reduce_(0, groups, r, reduce="amin")
        constant = maxima == minima
        # Summing identical FP32 rewards can round their mean away from the reward.
        centered = torch.where(constant[groups], 0.0, centered)
        means = torch.where(constant, maxima, means)
        stds = torch.where(constant, 0.0, stds)
    mad = x.new_zeros(group_count).index_add_(0, groups, centered.abs()) / counts.clamp_min(1)
    numerator = x.new_zeros(group_count, x.shape[1]).index_add_(0, groups, centered[:, None] * x)
    denominator = (counts - int(reduction == "unbiased")).clamp_min(1)
    covariance = numerator / denominator[:, None]
    covariance = torch.where((counts > 1)[:, None], covariance, 0.0)
    rms = covariance.square().mean(1).sqrt()
    magnitudes = {
        "reward_mad": mad,
        "raw_covariance": rms,
        "reward_std_normalized_covariance": rms / (stds + eps),
        "unit_normalized_covariance": torch.ones_like(rms),
    }[scale_mode] * gain
    signal = covariance / torch.where(mad > 0, mad, 1.0)[:, None]
    signal_rms = signal.square().mean(1).sqrt()
    active = (counts > 1) & (mad > 0) & (signal_rms > 0) & (magnitudes > 0)
    if zero_signal_mode == "exact":
        magnitudes = torch.where(active, magnitudes, 0.0)
    if zero_signal_mode == "threshold":
        maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(0, groups, r, reduce="amax")
        minima = r.new_full((group_count,), torch.inf).scatter_reduce_(0, groups, r, reduce="amin")
        active &= (maxima - minima > atol) & (rms > atol)
        magnitudes = torch.where(active, magnitudes, 0.0)
    if direction_mode == "historical":
        directions = signal / signal_rms.clamp_min(eps)[:, None]
    else:
        # Float64 norm avoids underflow for tiny but nonzero aggregates.
        norm = covariance.double().square().mean(1).sqrt()
        directions = (covariance.double() / torch.where(norm > 0, norm, 1.0)[:, None]).float()
    directions = torch.where(active[:, None], directions, 0.0)
    shape = (group_count, *endpoints.shape[1:])
    return {
        "group_covariance": covariance.reshape(shape),
        "group_directions": directions.reshape(shape),
        "group_guidance_magnitudes": magnitudes,
        "group_active": active,
        "group_counts": counts,
        "group_reward_means": means,
        "group_reward_stds": stds,
        "group_reward_abs_means": mad,
        "group_covariance_rms": rms,
    }
