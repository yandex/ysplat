/*
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

#pragma once

#include "Cameras.cuh" // For BaseCameraModel
#include "Lidars.h"

// Lidar camera parameters struct (device-side)
// TODO: Create a poper LidarModelParamters hierarchy.
struct RowOffsetStructuredSpinningLidarModelParametersExtDevice
{
    RowOffsetStructuredSpinningLidarModelParametersExtDevice(
        const gsplat::RowOffsetStructuredSpinningLidarModelParametersExt &params
    );

    // Sensor dimensions
    int n_rows;
    int n_columns;

    gsplat::FOVDevice fov_vert_rad;
    gsplat::FOVDevice fov_horiz_rad;
    float fov_eps_rad;

    // Per-element sensor angles (device pointers)
    const float *row_elevations_rad;
    const float *column_azimuths_rad;
    const float *row_azimuth_offsets_rad;

    // Spinning direction and frequency of the lidar
    gsplat::SpinningDirection spinning_direction;
    float spinning_frequency_hz;

    // Precomputed acceleration structure for rolling shutter timing
    // Maps (.x=azimuth, .y=elevation) angles to actual column indices.
    const int32_t *angles_to_columns_map = nullptr;
    int2 map_dim;              // Dimensions of the map grid
    float2 map_resolution_rad; // Grid cell size in radians [.x=azimuth, .y=elevation]

    // Angle to pixel scaling factor
    static constexpr int ANGLE_TO_PIXEL_SCALING_FACTOR = 1024;

    // Tiling info
    int n_bins_azimuth;
    int n_bins_elevation;
    int cdf_resolution_elevation;
    int cdf_resolution_azimuth;
    const int32_t *cdf_elevation;
    const int32_t *cdf_dense_ray_mask;
    const glm::ivec2 *tiles_pack_info;
    const glm::ivec2 *tiles_to_elements_map;
};

enum class LidarPhase : int
{
    CENTER = 0,
    EARLY  = 1,
    LATE   = 2,
};

// Lidar camera model for spinning lidar sensors (e.g., Hesai Pandar128, AT128)
// TODO: Create a proper LidarModel hierarchy.
struct RowOffsetStructuredSpinningLidarModel
    : BaseCameraModel<RowOffsetStructuredSpinningLidarModel, gsplat::extdist::EmptyExternalDistortionModel>
{
    using Base = BaseCameraModel<RowOffsetStructuredSpinningLidarModel, gsplat::extdist::EmptyExternalDistortionModel>;

    using KernelParameters = RowOffsetStructuredSpinningLidarModelParametersExtDevice;

    struct Parameters : Base::Parameters
    {
        RowOffsetStructuredSpinningLidarModelParametersExtDevice lidar;

        inline __device__ explicit Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : Base::Parameters(
                  {
                      {static_cast<unsigned int>(kernel_parameters.n_columns),
                       static_cast<unsigned int>(kernel_parameters.n_rows)},
                      ShutterType::ROLLING_LEFT_TO_RIGHT,
                      {},
        },
                  camera_index
              ),
            lidar(kernel_parameters)
        {
        }
    };

    inline __device__ RowOffsetStructuredSpinningLidarModel(const KernelParameters &kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
    {
    }

    Parameters parameters;

public:
    // Override shutter_relative_frame_time for lidar-specific rolling shutter timing
    __device__ float shutter_relative_frame_time(const glm::fvec2 &image_point) const
    {
        // Unlike cameras with simple linear rolling shutter (top-to-bottom, left-to-right),
        // spinning lidars have complex timing patterns:
        // - The sensor spins (clockwise or counter-clockwise)
        // - Each column corresponds to a specific capture time
        // - The mapping from (elevation, azimuth) angles to column (time) is non-linear
        //   due to mechanical characteristics and row-specific azimuth offsets
        //
        // This function uses the angleToColumnMap lookup table
        // to determine the correct capture time for a given image point.

        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        assert(lidar.angles_to_columns_map != nullptr);

        // Convert image point (in scaled pixel space) back to angles
        // image_point = [x, y] = [azimuth, elevation]
        constexpr float kToAngle = 1.f / lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const float azimuth      = image_point.x * kToAngle;
        const float elevation    = image_point.y * kToAngle;

        // Compute relative angles (within sensor FOV)
        const auto [rel_elevation, rel_azimuth] = this->relative_sensor_angles(elevation, azimuth);

        return this->relative_angles_to_frame_time(rel_elevation, rel_azimuth);
    }

    __device__ float relative_angles_to_frame_time(float rel_elevation, float rel_azimuth) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // Compute the index into the map and clamp it to valid range
        const int ihoriz = static_cast<int>(
            std::clamp<float>(rel_azimuth / lidar.map_resolution_rad.x + 0.5f, 0, lidar.map_dim.x - 1)
        );

        const int ivert = static_cast<int>(
            std::clamp<float>(rel_elevation / lidar.map_resolution_rad.y + 0.5f, 0, lidar.map_dim.y - 1)
        );

        assert(0 <= ihoriz && ihoriz < lidar.map_dim.x);
        assert(0 <= ivert && ivert < lidar.map_dim.y);

        // Lookup column index from the angle-to-column map
        // Note: map is on row-major layout
        const int column_idx = lidar.angles_to_columns_map[ivert * lidar.map_dim.x + ihoriz];

        // Normalize to [0, 1] range (relative frame time)
        return static_cast<float>(column_idx) / (lidar.n_columns - 1);
    }

    struct PhasePointReturn
    {
        glm::fvec2 phase_point;
        bool valid_flag;
        float relative_frame_time;
        bool resolved = true;
    };

    __device__ auto camera_ray_to_phase_point(const glm::fvec3 &cam_ray, LidarPhase phase, float margin_factor) const
        -> PhasePointReturn
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        const float ray_length = glm::length(cam_ray);
        if(ray_length < 1e-6f)
        {
            return {
                {0.f, 0.f},
                false,
                0.f
            };
        }

        const glm::fvec3 ray_normalized = cam_ray / ray_length;
        const float elevation           = std::asin(std::clamp(ray_normalized.z, -1.f, 1.f));
        const float azimuth             = std::atan2(ray_normalized.y, ray_normalized.x);

        const auto [rel_elevation, rel_azimuth] = this->relative_sensor_angles(elevation, azimuth);

        const float tol_azimuth   = margin_factor * lidar.fov_horiz_rad.span;
        const float tol_elevation = margin_factor * lidar.fov_vert_rad.span;

        const bool valid = rel_elevation >= -tol_elevation
                        && rel_elevation <= lidar.fov_vert_rad.span + tol_elevation
                        && rel_azimuth <= lidar.fov_horiz_rad.span + tol_azimuth;

        float phase_azimuth = rel_azimuth;
        if(phase == LidarPhase::EARLY && rel_azimuth >= PI)
        {
            phase_azimuth -= 2.f * PI;
        }
        else if(phase == LidarPhase::LATE && rel_azimuth < PI)
        {
            phase_azimuth += 2.f * PI;
        }

        const float frame_time  = this->relative_angles_to_frame_time(rel_elevation, rel_azimuth);
        const float time_offset = phase == LidarPhase::CENTER ? 0.f : rintf(phase_azimuth / (2.f * PI) - frame_time);
        return {
            {phase_azimuth * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR, elevation * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR},
            valid,
            frame_time + time_offset
        };
    }

    template<size_t N_ROLLING_SHUTTER_ITERATIONS = 10>
    __device__ auto world_point_to_phase_point_shutter_pose(
        const glm::fvec3 &world_point,
        const glm::fvec3 &world_velocity,
        const RollingShutterParameters &rolling_shutter_parameters,
        LidarPhase phase,
        float margin_factor,
        float max_time_residual,
        const QuaternionSlerper<true> &slerper
    ) const -> PhasePointReturn
    {
        const auto &rs       = rolling_shutter_parameters;
        PhasePointReturn proj = {};
        float projected_time = 0.f;
        if(phase == LidarPhase::CENTER)
        {
            const glm::fvec3 t_mid = 0.5f * rs.t_start + 0.5f * rs.t_end;
            const glm::fquat q_mid = precise_normalize(slerper(0.5f));
            proj                   = this->camera_ray_to_phase_point(
                glm::rotate(q_mid, world_point + 0.5f * world_velocity) + t_mid, phase, margin_factor
            );
            projected_time = 0.5f;
        }
        if(phase == LidarPhase::EARLY || (phase == LidarPhase::CENTER && !proj.valid_flag))
        {
            proj = this->camera_ray_to_phase_point(
                glm::rotate(rs.q_start, world_point) + rs.t_start, phase, margin_factor
            );
            projected_time = 0.f;
        }
        if(phase == LidarPhase::LATE || (phase == LidarPhase::CENTER && !proj.valid_flag))
        {
            const PhasePointReturn proj_end = this->camera_ray_to_phase_point(
                glm::rotate(rs.q_end, world_point + world_velocity) + rs.t_end, phase, margin_factor
            );
            if(phase == LidarPhase::CENTER && !proj_end.valid_flag)
            {
                return {proj_end.phase_point, false, 1.f};
            }
            proj           = proj_end;
            projected_time = 1.f;
        }

        for(auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j)
        {
            const float relative_frame_time = proj.relative_frame_time;
            if(relative_frame_time == projected_time)
            {
                break;
            }
            projected_time        = relative_frame_time;
            const glm::fvec3 t_rs = (1.f - relative_frame_time) * rs.t_start + relative_frame_time * rs.t_end;
            const glm::fquat q_rs = precise_normalize(slerper(relative_frame_time));
            proj                  = this->camera_ray_to_phase_point(
                glm::rotate(q_rs, world_point + relative_frame_time * world_velocity) + t_rs, phase, margin_factor
            );
        }

        const bool resolved = std::fabs(projected_time - proj.relative_frame_time) <= max_time_residual;
        return {proj.phase_point, proj.valid_flag, projected_time, resolved};
    }

    __device__ auto camera_ray_to_image_point(const glm::fvec3 &cam_ray, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // Normalize the camera ray (lidar uses normalized rays)
        const float ray_length = glm::length(cam_ray);
        if(ray_length < 1e-6f)
        {
            return {
                {0.f, 0.f},
                false
            };
        }

        const glm::fvec3 ray_normalized = cam_ray / ray_length;

        // Convert 3D ray to spherical coordinates (elevation, azimuth)
        // Clamp to [-1, 1] before asin: ray_normalized is unit-length so
        // |z| <= 1 mathematically, but FP rounding (especially under fast math)
        // can push the value past 1.0 by an ULP, making asin return NaN.
        const float elevation = std::asin(std::clamp(ray_normalized.z, -1.f, 1.f));
        const float azimuth   = std::atan2(ray_normalized.y, ray_normalized.x);

        // Image point: [x, y] = [column, row] = [azimuth, elevation] (in scaled angle space)
        const float row              = elevation * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const float column           = azimuth * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const glm::fvec2 image_point = glm::fvec2{column, row};

        // For validation, compute relative angles
        auto [rel_elevation, rel_azimuth] = this->relative_sensor_angles(elevation, azimuth);

        // Check FOV bounds with margin
        // The margin allows rays slightly outside FOV that might round to valid pixels
        const float tol_azimuth   = margin_factor * lidar.fov_horiz_rad.span;
        const float tol_elevation = margin_factor * lidar.fov_vert_rad.span;

        bool valid = rel_elevation >= -tol_elevation
                  && rel_azimuth >= -tol_azimuth
                  && rel_elevation <= lidar.fov_vert_rad.span + tol_elevation
                  && rel_azimuth <= lidar.fov_horiz_rad.span + tol_azimuth;

        return {image_point, valid};
    }

    // Convert pixel indices (j=column, i=row) to image points in scaled angle space.
    // Matches base class convention: j is the width axis (azimuth), i is the height axis (elevation).
    __device__ glm::fvec2 element_to_image_point(int j, int i) const
    {
        const int row     = i; // height axis = elevation
        const int col     = j; // width axis = azimuth
        const auto &lidar = parameters.lidar;
        assert(row >= 0 && row < lidar.n_rows);
        assert(col >= 0 && col < lidar.n_columns);

        const float elevation = lidar.row_elevations_rad[row];
        float azimuth         = lidar.column_azimuths_rad[col] + lidar.row_azimuth_offsets_rad[row];
        // Normalize to (-pi, pi]
        if(azimuth > PI)
        {
            azimuth -= 2.f * PI;
        }
        if(azimuth <= -PI)
        {
            azimuth += 2.f * PI;
        }
        return {azimuth * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR, elevation * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR};
    }

    __device__ CameraRay image_point_to_camera_ray(glm::fvec2 image_point) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // image_point = [x, y] = [column, row] = [azimuth, elevation]
        const float col = image_point.x;
        const float row = image_point.y;

        // Inverse of forward projection: elevation = row / scale
        const float elevation = row / lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
        const float azimuth   = col / lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;

        // Convert spherical coordinates (elevation, azimuth) to 3D ray
        const float sin_elevation = std::sin(elevation);
        const float cos_elevation = std::cos(elevation);
        const float sin_azimuth   = std::sin(azimuth);
        const float cos_azimuth   = std::cos(azimuth);

        const glm::fvec3 camera_ray = glm::fvec3{
            cos_azimuth * cos_elevation, // x component
            sin_azimuth * cos_elevation, // y component
            sin_elevation                // z component
        };

        return {camera_ray / glm::length(camera_ray), this->valid_sensor_angles(elevation, azimuth)};
    }

    __device__ float relative_clock_rotation(float begin, float end, gsplat::SpinningDirection direction) const
    {
        return (direction == gsplat::SpinningDirection::CLOCKWISE) ? (begin - end) : (end - begin);
    }

    __device__ float relative_angle(
        float angle_start, float angle_end, gsplat::SpinningDirection direction, float scale = 1
    ) const
    {
        const float period = scale * 2 * PI;

        float rel_angle = this->relative_clock_rotation(angle_start, angle_end, direction);
        // Avoid expensive fmod if the angle is already in a reasonable range
        // Similar optimization as in normalize_angle

        if(-period < rel_angle && rel_angle < 2 * period)
        {
            // Fast path: handle common cases without fmod
            rel_angle = (rel_angle >= period) ? rel_angle - period : rel_angle;
            rel_angle = (rel_angle < 0.f) ? rel_angle + period : rel_angle;
            return rel_angle;
        }
        else
        {
            // Slow path: use fmod for values outside the fast range
            rel_angle = std::fmod(rel_angle, period);
            // output range [0, period)
            return (rel_angle < 0.f) ? (rel_angle + period) : rel_angle;
        }
    }

    __device__ std::tuple<float, float> relative_sensor_angles(float elevation, float azimuth, float scale = 1) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = this->parameters.lidar;

        const float rel_elevation = this->relative_clock_rotation(
            lidar.fov_vert_rad.start * scale, elevation, gsplat::SpinningDirection::CLOCKWISE
        );
        const float rel_azimuth
            = this->relative_angle(lidar.fov_horiz_rad.start * scale, azimuth, lidar.spinning_direction, scale);

        return {rel_elevation, rel_azimuth};
    }

    // Checks if sensor angles are within the FOV of the sensor.
    __device__ bool valid_sensor_angles(float elevation, float azimuth, float scale = 1) const
    {
        const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = parameters.lidar;

        // Account for accumulated numerical errors via some epsilon in FOV check (1x eps on the start of the FOV)
        const float fov_vert_start_adj  = lidar.fov_vert_rad.start + lidar.fov_eps_rad;
        const float fov_horiz_start_adj = lidar.spinning_direction == gsplat::SpinningDirection::CLOCKWISE
                                            ? lidar.fov_horiz_rad.start + lidar.fov_eps_rad
                                            : lidar.fov_horiz_rad.start - lidar.fov_eps_rad;

        const float rel_elevation = this->relative_clock_rotation(
            fov_vert_start_adj * scale, elevation, gsplat::SpinningDirection::CLOCKWISE
        );
        const float rel_azimuth
            = this->relative_angle(fov_horiz_start_adj * scale, azimuth, lidar.spinning_direction, scale);

        // Also account for accumulated numerical errors via some epsilon in FOV check
        // (using 2x eps as 1x eps is "inherited" from the start of the FOV in the relative angles,
        //  so effectively this checks 1x eps on the end of the FOV)
        return (rel_elevation <= scale * (lidar.fov_vert_rad.span + lidar.fov_eps_rad * 2))
            && (rel_azimuth <= scale * (lidar.fov_horiz_rad.span + lidar.fov_eps_rad * 2));
    }
};

template<>
struct ProjectionUTCodegenTraits<RowOffsetStructuredSpinningLidarModel>
{
    static constexpr int kMinBlocks                = 4;
    static constexpr unsigned kUTUnroll            = 1u;
    static constexpr bool kNeedsCulling            = false;
    static constexpr bool kUseGaussianScopeSlerper = true;

    template<size_t N_ROLLING_SHUTTER_ITERATIONS>
    static constexpr auto rolling_shutter_unroll() -> unsigned
    {
        return 1u;
    }
};

struct LidarPhaseGaussianReturn
{
    glm::fvec2 mean;
    glm::fmat2 covariance;
    bool valid;
    float center_relative_frame_time;
    float azimuth_extent;
    bool resolved = true;
};

inline __device__ auto lidar_gaussian_to_phase_gaussian_unscented_transform(
    const RowOffsetStructuredSpinningLidarModel &lidar_model,
    const RollingShutterParameters &rolling_shutter_parameters,
    const UnscentedTransformParameters &unscented_transform_parameters,
    const glm::fvec3 &gaussian_world_mean,
    const glm::fvec3 &gaussian_world_scale,
    const glm::fquat &gaussian_world_rot,
    const glm::fvec3 &gaussian_world_velocity,
    LidarPhase phase,
    float max_time_residual
) -> LidarPhaseGaussianReturn
{
    const auto [lambda, sigma_points] = world_gaussian_sigma_points(
        unscented_transform_parameters, gaussian_world_mean, gaussian_world_scale, gaussian_world_rot
    );

    bool valid            = unscented_transform_parameters.require_all_sigma_points_valid;
    bool resolved         = true;
    auto phase_points     = std::array<glm::fvec2, 2 * UnscentedTransformParameters::D + 1>{};
    auto phase_mean       = glm::fvec2{0};
    auto phase_covariance = glm::fmat2{0};
    auto center_time      = 0.5f;
    auto azimuth_min      = std::numeric_limits<float>::max();
    auto azimuth_max      = -std::numeric_limits<float>::max();

    auto [mean, covariance, rest] = ut_weights(unscented_transform_parameters, lambda);

    const QuaternionSlerper<true> slerper(rolling_shutter_parameters.q_start, rolling_shutter_parameters.q_end);
    for(auto i = 0u; i < std::size(phase_points); ++i)
    {
        const auto [phase_point, point_valid, point_frame_time, point_resolved]
            = lidar_model.world_point_to_phase_point_shutter_pose<>(
                sigma_points.points[i],
                gaussian_world_velocity,
                rolling_shutter_parameters,
                phase,
                unscented_transform_parameters.in_image_margin_factor,
                max_time_residual,
                slerper
            );
        if(i == 0)
        {
            center_time = point_frame_time;
        }

        if(!std::isfinite(phase_point.x) || !std::isfinite(phase_point.y))
        {
            return {phase_mean, phase_covariance, false, center_time, 0.f};
        }
        resolved &= point_resolved;

        if(unscented_transform_parameters.require_all_sigma_points_valid)
        {
            valid &= point_valid;
            if(!point_valid)
            {
                return {phase_mean, phase_covariance, false, center_time, 0.f};
            }
        }
        else
        {
            valid |= point_valid;
        }
        phase_points[i]  = phase_point;
        phase_mean      += mean * phase_point;
        mean             = rest;
        azimuth_min      = fminf(azimuth_min, phase_point.x);
        azimuth_max      = fmaxf(azimuth_max, phase_point.x);
    }

    if(!valid)
    {
        return {phase_mean, phase_covariance, false, center_time, 0.f};
    }

    for(auto i = 0u; i < std::size(phase_points); ++i)
    {
        const auto phase_mean_vec  = phase_points[i] - phase_mean;
        phase_covariance          += covariance * glm::outerProduct(phase_mean_vec, phase_mean_vec);
        covariance                 = rest;
    }

    return {phase_mean, phase_covariance, valid, center_time, azimuth_max - azimuth_min, resolved};
}

// Type list of all lidar model types
using LidarModelTypes               = TypeList<RowOffsetStructuredSpinningLidarModel>;
using LidarModelKernelParamsVariant = gsplat::TypeListToKernelParamsVariant<LidarModelTypes>;
