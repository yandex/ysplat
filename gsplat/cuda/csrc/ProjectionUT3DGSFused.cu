/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGUT

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <ATen/cuda/Atomic.cuh>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>
#    include <cuda/std/optional>

#    include "Common.h"
#    include "ExternalDistortion.cuh"
#    include "Projection.h"
#    include "Utils.cuh"
#    include "Cameras.cuh"
#    include "Lidars.cuh"
#    include "Sensors.cuh"
#    include "Dispatch.h"

namespace gsplat
{
namespace cg = cooperative_groups;

struct LidarPhaseEntry
{
    bool valid            = false;
    glm::fvec2 mean       = glm::fvec2{0.f};
    glm::fmat2 covariance = glm::fmat2{0.f};
    float radius_x        = 0.f;
    float radius_y        = 0.f;
    float compensation    = 0.f;
    float center_time     = 0.5f;
    float azimuth_extent  = 0.f;
    bool unresolved       = false;
};

template<typename DepthFn>
__device__ void project_lidar_phases(
    const RowOffsetStructuredSpinningLidarModel &sensor,
    const RollingShutterParameters &rs_params,
    const UnscentedTransformParameters &ut_params,
    const glm::fvec3 &mean,
    const glm::fvec3 &scale,
    const glm::fquat &quat,
    const glm::fvec3 &velocity,
    const float *opacity,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const float row_offset_spread,
    const float seam_band,
    const DepthFn &observed_depth,
    const uint32_t primary_idx,
    const uint32_t secondary_idx,
    int32_t *__restrict__ radii,
    float *__restrict__ means2d,
    float *__restrict__ depths,
    float *__restrict__ conics,
    float *__restrict__ compensations
)
{
    constexpr float kSeamMarginPix         = 2.f;
    constexpr float kMaxRelativeSweepSlope = 0.5f;

    const auto &lidar           = sensor.parameters.lidar;
    const float angle_to_pixel  = lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
    const float full_circle_pix = 2.f * PI * angle_to_pixel;
    const float half_circle_pix = PI * angle_to_pixel;
    const float span_pix        = lidar.fov_horiz_rad.span * angle_to_pixel;
    const float tile_width_pix  = span_pix / lidar.n_bins_azimuth;
    const float max_time_residual
        = (row_offset_spread / (2.f * PI) + 2.f / (lidar.n_columns - 1)) / (1.f - kMaxRelativeSweepSlope);

    auto project_phase = [&](LidarPhase phase) -> LidarPhaseEntry
    {
        LidarPhaseEntry entry;
        const LidarPhaseGaussianReturn phase_gaussian = lidar_gaussian_to_phase_gaussian_unscented_transform(
            sensor, rs_params, ut_params, mean, scale, quat, velocity, phase, max_time_residual
        );
        if(!phase_gaussian.valid)
        {
            return entry;
        }

        glm::fmat2 covar2d = phase_gaussian.covariance;
        float compensation;
        const float det = add_blur(eps2d, covar2d, compensation);
        if(det <= 0.f || covar2d[0][0] < 0.f || covar2d[1][1] < 0.f)
        {
            return entry;
        }

        float extend = GAUSSIAN_EXTEND;
        if(opacity != nullptr)
        {
            const float compensated_opacity = *opacity * compensation;
            if(compensated_opacity < ALPHA_THRESHOLD)
            {
                return entry;
            }
            extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(compensated_opacity / ALPHA_THRESHOLD)));
        }

        const float b   = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        const float tmp = sqrtf(max(0.01f, b * b - det));
        const float r1  = extend * sqrtf(b + tmp);
        entry.radius_x  = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
        entry.radius_y  = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));
        if(entry.radius_x <= radius_clip && entry.radius_y <= radius_clip)
        {
            return entry;
        }

        entry.valid          = true;
        entry.mean           = phase_gaussian.mean;
        entry.covariance     = covar2d;
        entry.compensation   = compensation;
        entry.center_time    = phase_gaussian.center_relative_frame_time;
        entry.azimuth_extent = phase_gaussian.azimuth_extent;
        entry.unresolved     = !phase_gaussian.resolved;
        return entry;
    };

    auto set_azimuth_interval = [](LidarPhaseEntry &entry, const float lo, const float hi)
    {
        entry.radius_x = ceilf(0.5f * (hi - lo));
        entry.mean.x   = 0.5f * (lo + hi);
    };

    auto clip_to_scan = [&](LidarPhaseEntry &entry)
    {
        if(!entry.valid)
        {
            return;
        }
        const float lo = fmaxf(entry.mean.x - entry.radius_x, kSeamMarginPix);
        const float hi = fminf(entry.mean.x + entry.radius_x, span_pix - kSeamMarginPix);
        if(entry.azimuth_extent > half_circle_pix || lo >= hi)
        {
            entry.valid = false;
            return;
        }
        set_azimuth_interval(entry, lo, hi);
    };

    auto write_entry = [&](const uint32_t slot, const LidarPhaseEntry &entry, const float depth, const bool in_range)
    {
        if(!entry.valid || !in_range)
        {
            radii[slot * 2]     = 0;
            radii[slot * 2 + 1] = 0;
            return;
        }

        const bool clockwise  = lidar.spinning_direction == gsplat::SpinningDirection::CLOCKWISE;
        float azimuth_pix     = lidar.fov_horiz_rad.start * angle_to_pixel + (clockwise ? -entry.mean.x : entry.mean.x);
        azimuth_pix          -= full_circle_pix * rintf(azimuth_pix / full_circle_pix);

        glm::fmat2 covariance = entry.covariance;
        if(clockwise)
        {
            covariance[0][1] = -covariance[0][1];
            covariance[1][0] = -covariance[1][0];
        }
        const glm::fmat2 conic = glm::inverse(covariance);

        radii[slot * 2]       = (int32_t)entry.radius_x;
        radii[slot * 2 + 1]   = (int32_t)entry.radius_y;
        means2d[slot * 2]     = azimuth_pix;
        means2d[slot * 2 + 1] = entry.mean.y;
        depths[slot]          = depth;
        conics[slot * 3]      = conic[0][0];
        conics[slot * 3 + 1]  = conic[0][1];
        conics[slot * 3 + 2]  = conic[1][1];
        if(compensations != nullptr)
        {
            compensations[slot] = entry.compensation;
        }
    };

    LidarPhaseEntry primary = project_phase(LidarPhase::CENTER);
    LidarPhaseEntry secondary;
    const bool center_clean = primary.valid && !primary.unresolved && primary.azimuth_extent <= half_circle_pix;
    const bool moving       = velocity.x != 0.f || velocity.y != 0.f || velocity.z != 0.f;
    const float seam_band_pix      = seam_band * angle_to_pixel + tile_width_pix;
    const float min_separation_pix = tile_width_pix + kSeamMarginPix;
    bool split                     = primary.valid;
    if(center_clean && !moving)
    {
        const glm::fvec3 start_c = glm::rotate(rs_params.q_start, mean) + rs_params.t_start;
        const glm::fvec3 end_c   = glm::rotate(rs_params.q_end, mean) + rs_params.t_end;
        float drift              = atan2f(end_c.y, end_c.x) - atan2f(start_c.y, start_c.x);
        drift                    = fabsf(drift - 2.f * PI * rintf(drift / (2.f * PI))) * angle_to_pixel + 2.f;
        split                    = primary.mean.x - primary.radius_x - drift < seam_band_pix
                || primary.mean.x + primary.radius_x + drift > full_circle_pix;
    }
    bool merged       = false;
    float merged_time = 0.f;
    if(split)
    {
        LidarPhaseEntry early = project_phase(LidarPhase::EARLY);
        LidarPhaseEntry late  = project_phase(LidarPhase::LATE);
        clip_to_scan(early);
        if(late.valid)
        {
            const float lo = fmaxf(late.mean.x - late.radius_x, kSeamMarginPix);
            const float hi = fminf(late.mean.x + late.radius_x, span_pix + seam_band_pix);
            late.valid     = late.azimuth_extent <= half_circle_pix && lo < hi;
            set_azimuth_interval(late, lo, hi);
        }
        early.valid = early.valid && !early.unresolved && early.center_time >= -seam_band_pix / full_circle_pix;
        late.valid  = late.valid && !late.unresolved && late.center_time <= 1.f + seam_band_pix / full_circle_pix;
        const LidarPhaseEntry a = !early.valid && late.valid && center_clean ? primary : early;
        const LidarPhaseEntry b = early.valid && !late.valid && center_clean ? primary : late;
        if(a.valid && b.valid)
        {
            const float b_mean = b.mean.x - full_circle_pix * rintf((b.mean.x - a.mean.x) / full_circle_pix);
            const float a_lo   = a.mean.x - a.radius_x;
            const float a_hi   = a.mean.x + a.radius_x;
            const float b_lo   = b_mean - b.radius_x;
            const float b_hi   = b_mean + b.radius_x;
            const bool right   = b_mean >= a.mean.x;
            const float gap    = right ? b_lo - a_hi : a_lo - b_hi;
            const float wrap   = right ? a_lo + full_circle_pix - b_hi : b_lo + full_circle_pix - a_hi;
            primary            = a;
            if(gap >= min_separation_pix && wrap >= min_separation_pix)
            {
                secondary = b;
            }
            else
            {
                float lo = fminf(a_lo, b_lo);
                float hi = lo + full_circle_pix;
                if(wrap >= min_separation_pix)
                {
                    hi = fmaxf(a_hi, b_hi);
                }
                else if(gap >= min_separation_pix)
                {
                    lo = right ? b_lo : a_lo;
                    hi = right ? a_hi + full_circle_pix : b_hi + full_circle_pix;
                }
                const float y_lo = fminf(a.mean.y - a.radius_y, b.mean.y - b.radius_y);
                const float y_hi = fmaxf(a.mean.y + a.radius_y, b.mean.y + b.radius_y);
                set_azimuth_interval(primary, lo, hi);
                primary.radius_y = ceilf(0.5f * (y_hi - y_lo));
                primary.mean.y   = 0.5f * (y_lo + y_hi);
                merged           = true;
                merged_time      = b.center_time;
            }
        }
        else if(a.valid)
        {
            primary = a;
        }
        else if(b.valid)
        {
            primary = b;
        }
    }

    auto depth_in_range = [&](const float depth) { return depth >= near_plane && depth <= far_plane; };

    const float primary_depth = observed_depth(primary.center_time);
    if(merged)
    {
        const float merged_depth = observed_depth(merged_time);
        write_entry(
            primary_idx,
            primary,
            fmaxf(near_plane, fminf(primary_depth, merged_depth)),
            depth_in_range(primary_depth) || depth_in_range(merged_depth)
        );
    }
    else
    {
        write_entry(primary_idx, primary, primary_depth, depth_in_range(primary_depth));
    }
    if(secondary.valid)
    {
        const float secondary_depth = observed_depth(secondary.center_time);
        write_entry(secondary_idx, secondary, secondary_depth, depth_in_range(secondary_depth));
    }
}

// Cap registers to 64 (>= 4 blocks/SM) only for paths where measurement shows it
// wins. The default min-blocks policy is 1, which keeps camera codegen close to
// baseline without needing a separate kernel wrapper.
template<typename SensorModel, typename scalar_t, bool PhaseModeling>
__global__ void
    __launch_bounds__(256, ProjectionUTCodegenTraits<SensorModel>::kMinBlocks) projection_ut_3dgs_fused_kernel(
        const uint32_t B,
        const uint32_t C,
        const uint32_t N,
        const scalar_t *__restrict__ means,      // [B, N, 3]
        const scalar_t *__restrict__ quats,      // [B, N, 4]
        const scalar_t *__restrict__ scales,     // [B, N, 3]
        const scalar_t *__restrict__ opacities,  // [B, N] optional
        const scalar_t *__restrict__ velocities, // [B, N, 3] optional
        const scalar_t *__restrict__ viewmats0,  // [B, C, 4, 4]
        const scalar_t *__restrict__ viewmats1,  // [B, C, 4, 4] optional for rolling shutter
        // Image width and height are redundant, they are in the sensor model parameters
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const bool global_z_order,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        const float *__restrict__ lidar_seam_params,
        // sensor model parameters
        const __grid_constant__ typename SensorModel::KernelParameters sensor_model_params,
        // outputs
        int32_t *__restrict__ radii,         // [B, C, N, 2] or [B, C, 2N, 2] with phase modeling
        scalar_t *__restrict__ means2d,      // [B, C, N, 2] or [B, C, 2N, 2] with phase modeling
        scalar_t *__restrict__ depths,       // [B, C, N] or [B, C, 2N] with phase modeling
        scalar_t *__restrict__ conics,       // [B, C, N, 3] or [B, C, 2N, 3] with phase modeling
        scalar_t *__restrict__ compensations // [B, C, N] or [B, C, 2N] with phase modeling, optional
    )
{
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if(idx >= B * C * N)
    {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N;       // gaussian id

    if constexpr(PhaseModeling)
    {
        idx = (bid * C + cid) * 2 * N + gid;
        radii[(idx + N) * 2]     = 0;
        radii[(idx + N) * 2 + 1] = 0;
    }

    // shift pointers to the current gaussian
    const glm::fvec3 mean  = glm::make_vec3(means + bid * N * 3 + gid * 3);
    const glm::fvec3 scale = glm::make_vec3(scales + bid * N * 3 + gid * 3);
    const glm::fvec3 velocity
        = velocities == nullptr ? glm::fvec3{0.f} : glm::make_vec3(velocities + bid * N * 3 + gid * 3);
    const bool moving      = velocity.x != 0.f || velocity.y != 0.f || velocity.z != 0.f;
    glm::fquat quat        = glm::fquat{
        quats[bid * N * 4 + gid * 4 + 0],
        quats[bid * N * 4 + gid * 4 + 1],
        quats[bid * N * 4 + gid * 4 + 2],
        quats[bid * N * 4 + gid * 4 + 3]
    }; // w,x,y,z quaternion
    // A zero-length quaternion has no defined orientation — the Gaussian is
    // degenerate. Skip it to avoid NaN from glm::normalize dividing by zero.
    // Under --use_fast_math, 0/0 silently gives 0 instead of NaN, causing
    // glm::mat3_cast to produce an identity rotation (wrong, not NaN).
    float quat_norm2 = glm::dot(quat, quat);
    if(is_near_zero(quat_norm2))
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // A Gaussian with near-zero scale along any axis has a divergent precision
    // matrix (1/scale → inf). The 3DGUT rasterization kernel evaluates
    // Gaussians in 3D using the precision matrix, so such Gaussians must not
    // enter the intersection list.  Cull them here.
    if(is_near_zero(scale[0]) || is_near_zero(scale[1]) || is_near_zero(scale[2]))
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // Create rolling shutter parameter
    auto rs_params = RollingShutterParameters(
        viewmats0 + bid * C * 16 + cid * 16, viewmats1 == nullptr ? nullptr : viewmats1 + bid * C * 16 + cid * 16
    );

    // transform Gaussian center to camera space
    // Interpolate to *center* shutter pose as single per-Gaussian camera pose
    const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
    const vec3 mean_c       = glm::rotate(shutter_pose.q, mean) + shutter_pose.t;

    const float cull_depth = global_z_order ? mean_c.z : glm::length(mean_c);
    if(!moving && (cull_depth < near_plane || cull_depth > far_plane))
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // Make sure the rotation quaternion is normalized
    quat *= rsqrtf(quat_norm2);

    auto observed_depth = [&](float relative_frame_time) -> float
    {
        if(!moving)
        {
            return cull_depth;
        }
        const vec3 observed_c = glm::rotate(shutter_pose.q, mean + relative_frame_time * velocity) + shutter_pose.t;
        return global_z_order ? observed_c.z : glm::length(observed_c);
    };

    if constexpr(PhaseModeling)
    {
        project_lidar_phases(
            SensorModel(sensor_model_params, bid * C + cid),
            rs_params,
            ut_params,
            mean,
            scale,
            quat,
            velocity,
            opacities != nullptr ? opacities + bid * N + gid : nullptr,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            lidar_seam_params[0],
            lidar_seam_params[1],
            observed_depth,
            idx,
            idx + N,
            radii,
            means2d,
            depths,
            conics,
            compensations
        );
        return;
    }

    // projection using unscented transform
    ImageGaussianReturn image_gaussian_return = world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
        SensorModel(sensor_model_params, bid * C + cid), rs_params, ut_params, mean, scale, quat, velocity
    );

    auto [mean2d, covar2d, valid_ut, center_time] = image_gaussian_return;
    if(!valid_ut)
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if(det <= 0.f)
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // The UT center covariance weight can be very negative (e.g. ≈ -96 with
    // default alpha=0.1).  This means the UT covariance estimate is not
    // guaranteed positive-semidefinite — a diagonal entry can be negative even
    // when the determinant is positive.  Cull these: the UT has failed to
    // produce a valid 2D covariance for this Gaussian/camera combination, and
    // sqrtf(negative diagonal) below would produce NaN.
    if(covar2d[0][0] < 0.f || covar2d[1][1] < 0.f)
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = GAUSSIAN_EXTEND;
    // Note: the optimizations from StopThePop (https://arxiv.org/pdf/2402.00525) give identical
    // results when the Gaussian's contribution is evaluated in 2D from the projected 2D Gaussian.
    // However, with UT, the Gaussian's contribution is evaluated in 3D by intersecting the ray
    // with the 3D Gaussian.  As a result, the 2D culling criteria uses an approximation of what the
    // projected Gaussian would look like.  Therefore, this 2D-based culling can eliminate Gaussians
    // which would have a contribution in the 3D space and produce slightly different results,
    // especially with very non-linear projections / distortions.  Using these optimizations is
    // therefore a compromise between performance and quality, which is still reasonable for most cases.
    if(opacities != nullptr)
    {
        float opacity  = opacities[bid * N + gid];
        opacity       *= compensation;
        if(opacity < ALPHA_THRESHOLD)
        {
            radii[idx * 2]     = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float b        = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float tmp      = sqrtf(max(0.01f, b * b - det));
    float v1       = b + tmp; // larger eigenvalue
    float r1       = extend * sqrtf(v1);
    float radius_x = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
    float radius_y = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));

    if(radius_x <= radius_clip && radius_y <= radius_clip)
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    if constexpr(!ProjectionUTCodegenTraits<SensorModel>::kNeedsCulling)
    {
        // LIDAR culling is performed inside world_gaussian_to_image_gaussian_unscented_transform_shutter_pose,
        // so no additional 2D image-region masking is needed here.
    }
    else
    {
        // mask out gaussians outside the image region
        if(mean2d.x + radius_x <= 0
           || mean2d.x - radius_x >= image_width
           || mean2d.y + radius_y <= 0
           || mean2d.y - radius_y >= image_height)
        {
            radii[idx * 2]     = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
    }

    // Depth for sorting: z-depth (global_z_order=true) or Euclidean distance (global_z_order=false)
    float depth = observed_depth(center_time);
    if(moving && (depth < near_plane || depth > far_plane))
    {
        radii[idx * 2]     = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // write to outputs
    radii[idx * 2]       = (int32_t)radius_x;
    radii[idx * 2 + 1]   = (int32_t)radius_y;
    means2d[idx * 2]     = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx]          = depth;
    conics[idx * 3]      = covar2d_inv[0][0];
    conics[idx * 3 + 1]  = covar2d_inv[0][1];
    conics[idx * 3 + 2]  = covar2d_inv[1][1];
    if(compensations != nullptr)
    {
        compensations[idx] = compensation;
    }
}

void launch_projection_ut_3dgs_fused_kernel(
    // inputs
    const at::Tensor means,                   // [..., N, 3]
    const at::Tensor quats,                   // [..., N, 4]
    const at::Tensor scales,                   // [..., N, 3]
    const at::optional<at::Tensor> opacities,  // [..., N] optional
    const at::optional<at::Tensor> velocities, // [..., N, 3] optional
    const at::Tensor viewmats0,                // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1,  // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                       // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    const bool global_z_order,
    // uncented transform
    const at::optional<c10::intrusive_ptr<UnscentedTransformParameters>> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const at::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>>
        &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>
        &external_distortion_params, // external distortion parameters
    const bool lidar_phase_modeling,
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
)
{
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t B = means.numel() / (N * 3); // number of batches
    uint32_t C = Ks.size(-3);             // number of cameras

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    // Set-up the sensor model parameters.
    // We do the dispatch between the runtime configurations and the actual concrete sensor model
    // by storing them in a std::variant, then calling the specialized kernel on the concrete type.
    const auto external_distortion_kernel_params = [&]() -> extdist::ExternalDistortionModelKernelParamsVariant
    {
        if(external_distortion_params.has_value())
        {
            return extdist::BivariateWindshieldModel::KernelParameters{*external_distortion_params.value()};
        }
        else
        {
            return extdist::EmptyExternalDistortionModel::KernelParameters{};
        }
    }();

    const auto sensor_model_params = [&]() -> SensorModelKernelParamsVariant
    {
        if(camera_model == CameraModelType::PINHOLE)
        {
            if(!radial_coeffs.has_value() && !tangential_coeffs.has_value() && !thin_prism_coeffs.has_value())
            {
                return to_sensor_model_kernel_params(
                    get_camera_model_kernel_params<PerfectPinholeCameraModel>(
                        {image_width, image_height},
                        rs_type,
                        external_distortion_kernel_params,
                        Ks.const_data_ptr<float>()
                    )
                );
            }
            else
            {
                return to_sensor_model_kernel_params(
                    get_camera_model_kernel_params<OpenCVPinholeCameraModel>(
                        {image_width, image_height},
                        rs_type,
                        external_distortion_kernel_params,
                        Ks.const_data_ptr<float>(),
                        radial_coeffs.has_value() ? radial_coeffs.value().const_data_ptr<float>() : nullptr,
                        tangential_coeffs.has_value() ? tangential_coeffs.value().const_data_ptr<float>() : nullptr,
                        thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().const_data_ptr<float>() : nullptr
                    )
                );
            }
        }
        else if(camera_model == CameraModelType::ORTHO)
        {
            return to_sensor_model_kernel_params(
                get_camera_model_kernel_params<OrthographicCameraModel>(
                    {image_width, image_height}, rs_type, external_distortion_kernel_params, Ks.const_data_ptr<float>()
                )
            );
        }
        else if(camera_model == CameraModelType::FISHEYE)
        {
            return to_sensor_model_kernel_params(
                get_camera_model_kernel_params<OpenCVFisheyeCameraModel>(
                    {image_width, image_height},
                    rs_type,
                    external_distortion_kernel_params,
                    Ks.const_data_ptr<float>(),
                    radial_coeffs.has_value() ? radial_coeffs.value().const_data_ptr<float>() : nullptr
                )
            );
        }
        else if(camera_model == CameraModelType::SPHERICAL)
        {
            return to_sensor_model_kernel_params(
                get_camera_model_kernel_params<SphericalCameraModel>(
                    {image_width, image_height}, rs_type, external_distortion_kernel_params, Ks.const_data_ptr<float>()
                )
            );
        }
        else if(camera_model == CameraModelType::FTHETA)
        {
            TORCH_CHECK(ftheta_coeffs.has_value(), "ftheta coefficients must be given for ftheta camera model");
            return to_sensor_model_kernel_params(
                get_camera_model_kernel_params<FThetaCameraModel>(
                    {image_width, image_height},
                    rs_type,
                    external_distortion_kernel_params,
                    Ks.const_data_ptr<float>(),
                    *ftheta_coeffs.value()
                )
            );
        }
        else if(camera_model == CameraModelType::LIDAR)
        {
            TORCH_CHECK(lidar_coeffs.has_value(), "Lidar coefficients must be given for lidar camera model");
            return RowOffsetStructuredSpinningLidarModel::KernelParameters{*lidar_coeffs.value()};
        }
        else
        {
            TORCH_CHECK(
                false,
                "Invalid camera model: only pinhole, ortho, fisheye, ftheta, spherical, and lidar camera models are "
                "supported"
            );
        }
    }();

    // Default unscented-transform parameters when the caller leaves them unset.
    const auto ut = ut_params.has_value() ? ut_params.value() : c10::make_intrusive<UnscentedTransformParameters>();

    at::Tensor lidar_seam_params;
    if(lidar_phase_modeling)
    {
        const auto &lidar             = *lidar_coeffs.value();
        const at::Tensor &row_offsets = lidar.row_azimuth_offsets_rad.to(at::kFloat);
        const at::Tensor row_start    = row_offsets.add(lidar.column_azimuths_rad[0]);
        const float fov_start         = gsplat::checked_deref(lidar.fov_horiz_rad, "fov_horiz_rad").start;
        at::Tensor seam_lag           = lidar.spinning_direction == SpinningDirection::CLOCKWISE
                                          ? row_start.neg().add(fov_start)
                                          : row_start.sub(fov_start);
        seam_lag                      = seam_lag.sub(seam_lag.div(2.f * PI).round().mul(2.f * PI));
        lidar_seam_params             = row_offsets.new_empty({2});
        lidar_seam_params[0].copy_(row_offsets.max().sub(row_offsets.min()));
        lidar_seam_params[1].copy_(seam_lag.max().clamp_min(0.f));
    }

    auto launch_kernel = [&]<typename SensorModel>()
    {
        auto launch_phase_variant = [&]<bool PhaseModeling>()
        {
            projection_ut_3dgs_fused_kernel<SensorModel, float, PhaseModeling>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                    B,
                    C,
                    N,
                    means.const_data_ptr<float>(),
                    quats.const_data_ptr<float>(),
                    scales.const_data_ptr<float>(),
                    opacities.has_value() ? opacities.value().const_data_ptr<float>() : nullptr,
                    velocities.has_value() ? velocities.value().const_data_ptr<float>() : nullptr,
                    viewmats0.const_data_ptr<float>(),
                    viewmats1.has_value() ? viewmats1.value().const_data_ptr<float>() : nullptr,
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    global_z_order,
                    // uncented transform
                    *ut,
                    PhaseModeling ? lidar_seam_params.const_data_ptr<float>() : nullptr,
                    std::get<typename SensorModel::KernelParameters>(sensor_model_params),
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<float>(),
                    depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    compensations.has_value() ? compensations.value().data_ptr<float>() : nullptr
                );
        };
        if constexpr(std::is_same_v<SensorModel, RowOffsetStructuredSpinningLidarModel>)
        {
            if(lidar_phase_modeling)
            {
                launch_phase_variant.template operator()<true>();
                return;
            }
        }
        launch_phase_variant.template operator()<false>();
    };
    const bool dispatched = dispatch::dispatch(
        dispatch::make_mapped_type_param<SensorModelFromKernelParamsMap>(sensor_model_params), std::move(launch_kernel)
    );
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
