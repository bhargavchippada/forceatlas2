"""NumPy-vectorized ForceAtlas2 force computation.

Provides a vectorized alternative to the loop-based fa2util.py for
5-20x speedup without Cython. Uses NumPy broadcasting for pairwise
force computation.

This module works with (N, dim) NumPy arrays directly, not Node objects.
The ForceAtlas2 class converts between representations at boundaries.
"""

import numpy as np


def apply_repulsion(positions, masses, coefficient):
    """Apply repulsion forces between all node pairs.

    Parameters
    ----------
    positions : ndarray of shape (N, dim)
    masses : ndarray of shape (N,)
    coefficient : float (scalingRatio)

    Returns
    -------
    forces : ndarray of shape (N, dim)
    """
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)

    # Pairwise differences: (N, N, dim)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # Pairwise squared distances: (N, N)
    dist2 = np.sum(diff * diff, axis=2)

    # Avoid division by zero on diagonal and coincident nodes
    mass_prod = masses[:, np.newaxis] * masses[np.newaxis, :]
    # Guard: treat zero-distance pairs (diagonal + coincident) as 1.0 for division,
    # then zero out their factors so they contribute no force
    dist2_safe = np.where(dist2 > 0, dist2, 1.0)
    factors = coefficient * mass_prod / dist2_safe
    factors[dist2 == 0] = 0.0

    # Force vectors: (N, dim)
    forces = np.sum(diff * factors[:, :, np.newaxis], axis=1)
    return forces


def apply_repulsion_adjustSizes(positions, masses, sizes, coefficient):
    """Apply anti-collision repulsion between all node pairs."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=2))
    distance = euclidean - sizes[:, np.newaxis] - sizes[np.newaxis, :]
    mass_prod = masses[:, np.newaxis] * masses[np.newaxis, :]

    factors = np.zeros((n, n))
    # Non-overlapping (distance > 0 strictly to avoid division by zero at touching)
    mask_pos = (distance > 0) & (euclidean > 0)
    factors[mask_pos] = coefficient * mass_prod[mask_pos] / (distance[mask_pos] ** 2)
    # Overlapping or touching (distance <= 0)
    mask_neg = (distance <= 0) & (euclidean > 0)
    factors[mask_neg] = 100.0 * coefficient * mass_prod[mask_neg]
    np.fill_diagonal(factors, 0.0)

    forces = np.sum(diff * factors[:, :, np.newaxis], axis=1)
    return forces


def apply_gravity(positions, masses, gravity, scalingRatio, useStrongGravity=False):
    """Apply gravitational forces toward origin.

    Returns
    -------
    forces : ndarray of shape (N, dim)
    """
    if useStrongGravity:
        # F = -coefficient * mass * gravity * pos (distance-independent)
        any_nonzero = np.any(positions != 0, axis=1)
        factors = np.where(any_nonzero, scalingRatio * masses * gravity, 0.0)
        return -positions * factors[:, np.newaxis]
    else:
        # F = -mass * gravity * pos / distance
        dist = np.sqrt(np.sum(positions * positions, axis=1))
        factors = np.zeros_like(dist)
        mask = dist > 0
        factors[mask] = masses[mask] * gravity / dist[mask]
        return -positions * factors[:, np.newaxis]


def apply_attraction(positions, edge_sources, edge_targets, edge_weights,
                     masses, distributedAttraction, coefficient,
                     edgeWeightInfluence, linLogMode=False, adjustSizes=False,
                     sizes=None):
    """Apply attraction forces along all edges.

    Parameters
    ----------
    positions : ndarray of shape (N, dim)
    edge_sources, edge_targets : ndarray of shape (M,), int
    edge_weights : ndarray of shape (M,)
    masses : ndarray of shape (N,)
    distributedAttraction : bool
    coefficient : float
    edgeWeightInfluence : float
    linLogMode : bool
    adjustSizes : bool
    sizes : ndarray of shape (N,) or None

    Returns
    -------
    forces : ndarray of shape (N, dim)
    """
    forces = np.zeros_like(positions)

    if len(edge_sources) == 0:
        return forces

    # Compute effective weights
    if edgeWeightInfluence == 0:
        w = np.ones(len(edge_weights))
    elif edgeWeightInfluence == 1:
        w = edge_weights
    else:
        w = np.power(edge_weights, edgeWeightInfluence)

    # Distance vectors along edges
    diff = positions[edge_sources] - positions[edge_targets]
    euclidean = np.sqrt(np.sum(diff * diff, axis=1))

    if adjustSizes and sizes is not None:
        distance = euclidean - sizes[edge_sources] - sizes[edge_targets]
        active = distance > 0
    else:
        active = np.ones(len(edge_sources), dtype=bool)
        distance = euclidean

    if linLogMode:
        # F = -coeff * w * log(1 + d) → factor = -coeff * w * log(1+d)/d
        safe_dist = np.where(active & (euclidean > 0), euclidean, 1.0)
        if adjustSizes and sizes is not None:
            safe_adj = np.where(active & (distance > 0), distance, 1.0)
            log_factor = np.log(1.0 + safe_adj) / safe_adj
        else:
            log_factor = np.log(1.0 + safe_dist) / safe_dist
        if distributedAttraction:
            factors = np.where(active & (euclidean > 0),
                               -coefficient * w * log_factor / masses[edge_sources], 0.0)
        else:
            factors = np.where(active & (euclidean > 0),
                               -coefficient * w * log_factor, 0.0)
    else:
        # F = -coeff * w * d → factor = -coeff * w
        if distributedAttraction:
            factors = np.where(active, -coefficient * w / masses[edge_sources], 0.0)
        else:
            factors = np.where(active, -coefficient * w, 0.0)

    # Apply forces to source and target nodes
    force_vecs = diff * factors[:, np.newaxis]
    np.add.at(forces, edge_sources, force_vecs)
    np.add.at(forces, edge_targets, -force_vecs)

    return forces


def adjustSpeedAndApplyForces(positions, forces, old_forces, masses, speed, speedEfficiency,
                              jitterTolerance, adjustSizes=False, sizes=None):
    """Adjust speed and apply forces to positions.

    Returns
    -------
    new_positions : ndarray of shape (N, dim)
    speed : float
    speedEfficiency : float
    """
    n = len(positions)

    # Swing and traction per node
    swing_vecs = old_forces - forces
    tract_vecs = old_forces + forces
    node_swinging = np.sqrt(np.sum(swing_vecs * swing_vecs, axis=1))
    node_traction = np.sqrt(np.sum(tract_vecs * tract_vecs, axis=1))

    totalSwinging = np.sum(masses * node_swinging)
    totalEffectiveTraction = 0.5 * np.sum(masses * node_traction)

    # Jitter tolerance
    estimatedOptimalJitterTolerance = 0.05 * np.sqrt(n)
    minJT = np.sqrt(estimatedOptimalJitterTolerance)
    maxJT = 10.0
    if n > 0 and totalEffectiveTraction > 0:
        jt = jitterTolerance * max(minJT,
                                   min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (n * n)))
    else:
        jt = jitterTolerance * minJT

    minSpeedEfficiency = 0.05

    if totalEffectiveTraction > 0 and totalSwinging / totalEffectiveTraction > 2.0:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= 0.5
        jt = max(jt, jitterTolerance)

    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= 0.7
    elif speed < 1000:
        speedEfficiency *= 1.3

    maxRise = 0.5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    # Per-node speed factors
    per_node_swing = masses * node_swinging
    per_node_factor = speed / (1.0 + np.sqrt(speed * per_node_swing))

    if adjustSizes and sizes is not None:
        max_factor = np.where(sizes > 0, 10.0 / sizes, per_node_factor)
        per_node_factor = np.minimum(per_node_factor, max_factor)

    new_positions = positions + forces * per_node_factor[:, np.newaxis]

    return new_positions, speed, speedEfficiency
