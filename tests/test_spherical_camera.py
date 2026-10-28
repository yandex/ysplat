# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gsplat.cuda._wrapper import fully_fused_projection_with_ut
from gsplat.rendering import rasterization


def _directions(longitudes: torch.Tensor, latitudes: torch.Tensor) -> torch.Tensor:
    cos_lat = torch.cos(latitudes)
    return torch.stack(
        (
            torch.sin(longitudes) * cos_lat,
            torch.sin(latitudes),
            torch.cos(longitudes) * cos_lat,
        ),
        dim=-1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_spherical_ut_projection_matches_calibration():
    device = torch.device("cuda")
    width, height = 1024, 820
    fx, fy, cx, cy = 390.757, 391.044, 511.5, 409.5
    degrees = torch.tensor(
        [[0.0, 0.0], [75.0, 0.0], [-75.0, 0.0], [0.0, 60.0], [0.0, -60.0]],
        device=device,
    )
    angles = torch.deg2rad(degrees)
    means = 10.0 * _directions(angles[:, 0], angles[:, 1])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(means), device=device)
    scales = torch.full_like(means, 1e-2)
    opacities = torch.full((len(means),), 0.5, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], device=device)

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        camera_model="spherical",
        global_z_order=False,
    )

    expected = torch.stack((fx * angles[:, 0] + cx, fy * angles[:, 1] + cy), dim=-1)
    assert torch.all(radii > 0)
    # The UT result is the weighted mean of the projected sigma points rather
    # than the projection of the Gaussian center, so a small nonlinear shift is
    # expected away from the equator.
    torch.testing.assert_close(means2d[0], expected, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(
        depths[0], torch.full_like(depths[0], 10.0), atol=2e-5, rtol=0.0
    )


def _render_scene(*, explicit_rays: bool, camera_requires_grad: bool = False):
    device = torch.device("cuda")
    height, width = 16, 20
    fx, fy, cx, cy = 7.0, 7.5, 9.5, 7.5
    means = torch.tensor(
        [[0.0, 0.0, 3.0], [1.5, -0.7, 4.0]],
        device=device,
        requires_grad=True,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device, requires_grad=True)
    scales = torch.full((2, 3), 0.45, device=device, requires_grad=True)
    opacities = torch.full((2,), 0.7, device=device, requires_grad=True)
    colors = torch.tensor(
        [[1.0, 0.2, 0.1], [0.1, 0.3, 1.0]], device=device, requires_grad=True
    )
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], device=device)
    if camera_requires_grad:
        viewmats.requires_grad_()
        Ks.requires_grad_()

    rays = None
    if explicit_rays:
        rows, cols = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        longitude = (cols - cx) / fx
        latitude = (rows - cy) / fy
        directions = _directions(longitude, latitude)
        rays = torch.cat((torch.zeros_like(directions), directions), dim=-1).unsqueeze(
            0
        )

    render, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        packed=False,
        with_ut=True,
        with_eval3d=True,
        camera_model="spherical",
        global_z_order=False,
        rays=rays,
    )
    return render, alpha, (means, quats, scales, opacities, colors, viewmats, Ks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_spherical_native_rays_match_explicit_rays():
    native_render, native_alpha, _ = _render_scene(explicit_rays=False)
    explicit_render, explicit_alpha, _ = _render_scene(explicit_rays=True)
    torch.testing.assert_close(native_render, explicit_render, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(native_alpha, explicit_alpha, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_spherical_render_backprops_only_render_parameters():
    render, alpha, params = _render_scene(
        explicit_rays=False, camera_requires_grad=True
    )
    means, quats, scales, opacities, colors, viewmats, Ks = params
    (render.sum() + alpha.sum()).backward()

    for parameter in (means, quats, scales, opacities, colors):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert viewmats.grad is None
    assert Ks.grad is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"with_ut": False}, "with_ut=True and with_eval3d=True"),
        ({"with_eval3d": False}, "with_ut=True and with_eval3d=True"),
        ({"global_z_order": True}, "global_z_order=False"),
        (
            {"radial_coeffs": torch.zeros((1, 4))},
            "does not support camera distortion",
        ),
    ],
)
def test_spherical_public_api_restrictions(overrides, message):
    kwargs = {
        "with_ut": True,
        "with_eval3d": True,
        "global_z_order": False,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        rasterization(
            torch.zeros((1, 3)),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.ones((1, 3)),
            torch.ones((1,)),
            torch.ones((1, 3)),
            torch.eye(4).unsqueeze(0),
            torch.eye(3).unsqueeze(0),
            16,
            16,
            packed=False,
            camera_model="spherical",
            **kwargs,
        )
