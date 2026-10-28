# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat
from gsplat.rendering import RendererConfig_ParallelBatch, rasterization


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device"),
    pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support is not built"),
]


def _scene(*, requires_grad: bool):
    device = torch.device("cuda")
    height = width = 16
    means = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        device=device,
        requires_grad=requires_grad,
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 2,
        device=device,
        requires_grad=requires_grad,
    )
    scales = torch.ones((2, 3), device=device, requires_grad=requires_grad)
    opacities = torch.full((2,), 0.5, device=device, requires_grad=requires_grad)
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device=device,
        requires_grad=requires_grad,
    )
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rays = torch.zeros((1, height, width, 6), device=device)
    rays[..., 5] = 1.0
    rays.requires_grad_(requires_grad)
    return means, quats, scales, opacities, colors, viewmats, Ks, rays


def _render(*, requires_grad: bool, with_eval3d: bool = True):
    scene = _scene(requires_grad=requires_grad)
    render, alpha, meta = rasterization(
        *scene[:7],
        16,
        16,
        packed=False,
        tile_size=16,
        with_ut=True,
        with_eval3d=with_eval3d,
        rays=scene[7],
        return_depth_dispersion=True,
        # Diagnostics transparently use the one-pass serial traversal.
        renderer_config=RendererConfig_ParallelBatch(),
    )
    return render, alpha, meta, scene[:5] + (scene[7],)


def test_squared_depth_dispersion_matches_pairwise_expectation_and_keeps_mass_separate():
    render, alpha, meta, _ = _render(requires_grad=False)

    raw = meta["depth_squared_dispersion"]
    log = meta["log_depth_squared_dispersion"]
    raw_unnormalized = meta["depth_squared_dispersion_unnormalized"]
    log_unnormalized = meta["log_depth_squared_dispersion_unnormalized"]
    moments = meta["depth_linear_moments"]
    assert raw.shape == (1, 16, 16)
    assert log.shape == raw.shape
    assert moments.shape == (1, 16, 16, 4)
    assert not raw.requires_grad
    assert not log.requires_grad

    # Alpha compositing gives weights (0.5, 0.25), total mass W=0.75.
    # Both outputs are E[(X-X')**2], with the compositing mass normalized by W**2.
    expected_raw = 2.0 * 0.5 * 0.25 * (3.0 - 2.0) ** 2 / 0.75**2
    expected_log = 2.0 * 0.5 * 0.25 * (math.log(3.0) - math.log(2.0)) ** 2 / 0.75**2
    expected_moments = torch.tensor(
        [
            0.5 * 2.0 + 0.25 * 3.0,
            0.5 * 2.0**2 + 0.25 * 3.0**2,
            0.5 * math.log(2.0) + 0.25 * math.log(3.0),
            0.5 * math.log(2.0) ** 2 + 0.25 * math.log(3.0) ** 2,
        ],
        device="cuda",
    )
    torch.testing.assert_close(alpha[0, 8, 8, 0], torch.tensor(0.75, device="cuda"))
    torch.testing.assert_close(raw[0, 8, 8], torch.tensor(expected_raw, device="cuda"))
    torch.testing.assert_close(log[0, 8, 8], torch.tensor(expected_log, device="cuda"))
    torch.testing.assert_close(
        raw_unnormalized[0, 8, 8], torch.tensor(expected_raw * 0.75**2, device="cuda")
    )
    torch.testing.assert_close(
        log_unnormalized[0, 8, 8], torch.tensor(expected_log * 0.75**2, device="cuda")
    )
    torch.testing.assert_close(moments[0, 8, 8], expected_moments)
    assert torch.isfinite(render).all()


def test_squared_depth_dispersion_accepts_descending_hit_distance_order():
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )

    scene = _scene(requires_grad=False)
    means, quats, scales, opacities, colors, viewmats, Ks, rays = scene
    opacities = torch.tensor([0.8, 0.2], device="cuda")
    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, 16, 16
    )
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, 16, 1, 1)
    isect_offsets = isect_offset_encode(isect_ids, 1, 1, 1)

    # The regular intersection order is near-to-far. Reverse the only tile's
    # list so accumulation sees hit distances 3 then 2.
    assert flatten_ids.numel() == 2
    outputs = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors.unsqueeze(0),
        opacities.unsqueeze(0),
        viewmats,
        Ks,
        16,
        16,
        16,
        isect_offsets,
        flatten_ids.flip(0),
        rays=rays.reshape(1, -1, 6),
        return_depth_dispersion=True,
        renderer_config=RendererConfig_ParallelBatch(),
    )
    moments = outputs[-1]

    # Reversed traversal gives compositing weights 0.2 at depth 3 and
    # (1-0.2)*0.8 = 0.64 at depth 2, with W=0.84.
    expected_moments = torch.tensor(
        [
            0.2 * 3.0 + 0.64 * 2.0,
            0.2 * 3.0**2 + 0.64 * 2.0**2,
            0.2 * math.log(3.0) + 0.64 * math.log(2.0),
            0.2 * math.log(3.0) ** 2 + 0.64 * math.log(2.0) ** 2,
        ],
        device="cuda",
    )
    torch.testing.assert_close(
        outputs[1][0, 8, 8, 0], torch.tensor(0.84, device="cuda")
    )
    torch.testing.assert_close(
        moments[0, 8, 8],
        expected_moments,
    )


def test_depth_dispersion_and_linear_moments_support_backward():
    render, alpha, meta, parameters = _render(requires_grad=True)
    differentiable_outputs = (
        meta["depth_linear_moments"],
        meta["depth_squared_dispersion_unnormalized"],
        meta["depth_squared_dispersion"],
        meta["log_depth_squared_dispersion_unnormalized"],
        meta["log_depth_squared_dispersion"],
    )
    assert all(output.requires_grad for output in differentiable_outputs)

    # At the center ray the Gaussian response is maximal, so compositing uses
    # w = [a0, (1-a0)*a1] and hit distances equal the two z coordinates. Build
    # the same closure in ordinary PyTorch as a gradient reference.
    reference_depths = torch.tensor([2.0, 3.0], device="cuda", requires_grad=True)
    reference_opacities = torch.full((2,), 0.5, device="cuda", requires_grad=True)
    reference_weights = torch.stack(
        [
            reference_opacities[0],
            (1.0 - reference_opacities[0]) * reference_opacities[1],
        ]
    )
    reference_weight_sum = reference_weights.sum()
    reference_log_depths = reference_depths.log()
    reference_moments = torch.stack(
        [
            (reference_weights * reference_depths).sum(),
            (reference_weights * reference_depths.square()).sum(),
            (reference_weights * reference_log_depths).sum(),
            (reference_weights * reference_log_depths.square()).sum(),
        ]
    )
    reference_depth_unnormalized = 2.0 * (
        reference_weight_sum * reference_moments[1] - reference_moments[0].square()
    )
    reference_log_unnormalized = 2.0 * (
        reference_weight_sum * reference_moments[3] - reference_moments[2].square()
    )
    reference_loss = (
        reference_moments.sum()
        + reference_depth_unnormalized
        + reference_depth_unnormalized / reference_weight_sum.square()
        + reference_log_unnormalized
        + reference_log_unnormalized / reference_weight_sum.square()
    )
    reference_loss.backward()

    loss = sum(output[0, 8, 8].sum() for output in differentiable_outputs)
    loss.backward()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert parameters[0].grad[:, 2].abs().sum() > 0
    assert parameters[3].grad.abs().sum() > 0
    assert parameters[-1].grad.abs().sum() > 0
    torch.testing.assert_close(
        parameters[0].grad[:, 2], reference_depths.grad, rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        parameters[3].grad, reference_opacities.grad, rtol=2e-4, atol=2e-4
    )


def test_linear_moment_backward_matches_reference_across_batch_boundary():
    # tile_size=16 makes one logical raster batch 256 Gaussians. Two extra
    # samples exercise the persisted moment prefix used by the shared
    # batch-parallel backward kernel.
    sample_count = 258
    device = torch.device("cuda")
    depths = torch.linspace(2.0, 4.0, sample_count, device=device)
    means = torch.zeros((sample_count, 3), device=device)
    means[:, 2] = depths
    means.requires_grad_()
    opacities = torch.full((sample_count,), 5.0e-3, device=device, requires_grad=True)
    quats = torch.zeros((sample_count, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.ones((sample_count, 3), device=device)
    colors = torch.ones((sample_count, 3), device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rays = torch.zeros((1, 16, 16, 6), device=device)
    rays[..., 5] = 1.0

    _, _, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        16,
        16,
        packed=False,
        tile_size=16,
        with_ut=True,
        with_eval3d=True,
        rays=rays,
        return_depth_dispersion=True,
        renderer_config=RendererConfig_ParallelBatch(),
    )
    meta["depth_linear_moments"][0, 8, 8].sum().backward()

    reference_depths = depths.detach().clone().requires_grad_()
    reference_opacities = opacities.detach().clone().requires_grad_()
    transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones((1,), device=device),
                1.0 - reference_opacities[:-1],
            ]
        ),
        dim=0,
    )
    weights = transmittance * reference_opacities
    log_depths = reference_depths.log()
    reference_moments = torch.stack(
        [
            (weights * reference_depths).sum(),
            (weights * reference_depths.square()).sum(),
            (weights * log_depths).sum(),
            (weights * log_depths.square()).sum(),
        ]
    )
    reference_moments.sum().backward()

    torch.testing.assert_close(
        means.grad[:, 2], reference_depths.grad, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        opacities.grad, reference_opacities.grad, rtol=1e-3, atol=1e-3
    )


def test_depth_dispersion_requires_eval3d():
    with pytest.raises(
        ValueError,
        match="return_depth_dispersion=True requires with_eval3d=True",
    ):
        _render(requires_grad=False, with_eval3d=False)
