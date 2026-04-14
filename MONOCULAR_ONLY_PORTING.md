# Monocular-Only Porting Notes

This document describes the code changes made to support `monocular + no IMU` operation in this repo.

Scope:
- This covers the estimator and config changes needed for pure visual monocular startup and tracking.
- This does not cover later Docker / OpenCV / build-environment fixes.

The target mode is:
- `num_of_cam = 1`
- `imu = 0`

Important limitation:
- This mode is still arbitrary-scale monocular VO/VIO.
- The implementation removes the IMU dependency and keeps the optimization well-posed, but it does not make the result metric by itself.

## High-Level Summary

Before these changes, the codebase supported:
- monocular + IMU
- stereo + IMU
- stereo-only

What was missing was:
- monocular-only

The main gap was not feature tracking or ROS subscription logic. The real gap was estimator startup:
- the existing monocular initializer always ended in `visualInitialAlign()`
- `visualInitialAlign()` calls `VisualIMUAlignment(...)`
- so monocular startup was implicitly hard-coupled to IMU

The fix was:
1. make `use_imu = 0` a valid estimator mode
2. stop creating / storing IMU preintegration objects when IMU is disabled
3. add a dedicated `mono && !imu` initialization branch
4. reuse the existing global SFM + PnP structure recovery
5. skip the IMU alignment stage in pure monocular mode
6. add an explicit scale-gauge anchor so pure monocular optimization does not become singular

## Files Changed

Estimator/config changes for monocular-only:
- `include/vins_estimator/estimator/parameters.h`
- `src/vins_estimator/estimator/parameters.cpp`
- `src/vins_estimator/estimator/estimator.cpp`
- `include/loop_fusion/parameters.h`
- `src/loop_fusion/parameters.cpp`
- `config/euroc/euroc_mono_config.yaml`

## Detailed Changes

### 1. Parameter Defaults And Config Parsing

Files:
- `include/vins_estimator/estimator/parameters.h`
- `src/vins_estimator/estimator/parameters.cpp`
- `include/loop_fusion/parameters.h`
- `src/loop_fusion/parameters.cpp`

Changes:
- `stereo` now defaults to `0`
- `use_imu` now defaults to `0`
- the parser explicitly resets `stereo = 0` before checking `num_of_cam`

Reason:
- previously, mono configs could depend on whatever value happened to be left in `stereo`
- that is dangerous when `num_of_cam == 1`
- pure monocular mode needs deterministic `stereo = 0`

Porting guidance:
- if your other repo has a `Parameters` struct or equivalent, make sure `stereo` and `use_imu` are initialized
- when parsing config, explicitly set `stereo = 0` before the `if (num_of_cam == 2)` branch
- apply the same fix in any loop-closure / visualization / auxiliary modules that reuse the same sensor-mode fields

### 2. Accept Vision-Only Sensor Mode

File:
- `src/vins_estimator/estimator/estimator.cpp`

Function:
- `Estimator::changeSensorType(int use_imu, int use_stereo)`

Changes:
- removed the old rejection of `!use_imu && !use_stereo`
- when switching IMU off, explicitly delete:
  - `last_marginalization_info`
  - `tmp_pre_integration`
- clear:
  - `last_marginalization_parameter_blocks`

Reason:
- the old code assumed "at least two sensors" and treated `mono-only` as invalid
- once IMU is disabled, any residual state containing IMU-based priors or preintegrations must be dropped

Porting guidance:
- if your repo can switch sensor modes at runtime, make sure turning IMU off also clears all IMU-derived state
- otherwise, stale priors can survive the mode switch and poison the optimization

### 3. Stop Creating Preintegration Objects In Vision-Only Mode

File:
- `src/vins_estimator/estimator/estimator.cpp`

Function:
- `Estimator::processImage(...)`

Changes:
- when constructing `ImageFrame`, set:
  - `imageframe.pre_integration = params.use_imu ? tmp_pre_integration : nullptr`
- after inserting the frame, only allocate a new `IntegrationBase` if `params.use_imu` is true
- otherwise, keep `tmp_pre_integration = nullptr`

Reason:
- pure monocular mode has no IMU measurements
- storing or allocating preintegration objects in that mode is incorrect and can later lead to null misuse or stale-state assumptions

Porting guidance:
- anywhere your image-frame structure stores IMU preintegration, make that conditional on `use_imu`
- make sure any later code touching `frame.pre_integration` is also guarded

### 4. Add A Dedicated Monocular-Only Initialization Branch

File:
- `src/vins_estimator/estimator/estimator.cpp`

Function:
- `Estimator::processImage(...)`

New branch:
- `if (!params.stereo && !params.use_imu) { ... }`

Behavior:
- wait until `frame_count == WINDOW_SIZE`
- gate attempts with `(header - initial_timestamp) > 0.1`
- call `initialStructure()`
- if successful:
  - call `optimization()`
  - call `updateLatestStates()`
  - set `solver_flag = NON_LINEAR`
  - call `slideWindow()`
- otherwise:
  - just `slideWindow()`

Reason:
- the code already had startup branches for:
  - mono + IMU
  - stereo + IMU
  - stereo-only
- monocular-only needed its own explicit startup path

Porting guidance:
- if your estimator has separate initialization logic by sensor mode, add a fourth branch rather than trying to force monocular-only through an IMU path
- the branch should reuse the visual initializer, then enter the normal nonlinear backend

### 5. Reuse Existing Global SFM And PnP

File:
- `src/vins_estimator/estimator/estimator.cpp`

Function:
- `Estimator::initialStructure()`

Important point:
- `initial_sfm.cpp` was not rewritten
- the existing `GlobalSFM` and per-frame PnP logic were reused

What stayed the same:
- build `SFMFeature` list from tracked features
- compute relative pose with `relativePose(...)`
- run `GlobalSFM::construct(...)`
- solve PnP for non-keyframes to recover the whole sliding-window pose set

Reason:
- the visual front-end and structure bootstrap were already sufficient for monocular-only
- the real dependency problem was the final IMU alignment step, not SFM itself

Porting guidance:
- if your other repo already has a visual SFM initializer for mono+IMU, first check whether the SFM stage itself is usable without IMU
- often the needed change is not in SFM, but in what comes after SFM

### 6. Make `initialStructure()` Conditional On IMU

File:
- `src/vins_estimator/estimator/estimator.cpp`

Function:
- `Estimator::initialStructure()`

Changes:

#### 6.1 IMU observability check is now conditional

Old behavior:
- the function always entered the IMU excitation / observability block

New behavior:
- that block only runs if `params.use_imu`

Reason:
- in monocular-only mode there is no preintegration to inspect

#### 6.2 IMU alignment is now optional

Old behavior:
- after SFM + PnP, the function always ended with:
  - `visualInitialAlign()`
- and `visualInitialAlign()` eventually calls `VisualIMUAlignment(...)`

New behavior:
- if `params.use_imu`:
  - behavior is unchanged
  - `visualInitialAlign()` is still required
- if `!params.use_imu`:
  - skip `visualInitialAlign()`
  - copy recovered visual poses from `all_image_frame` into estimator arrays:
    - `Ps[i]`
    - `Rs[i]`
  - mark all window frames as keyframes
  - clear existing feature depths with `f_manager.clearDepth()`
  - retriangulate with `f_manager.triangulate(frame_count, Ps, Rs, tic, ric)`
  - return `true`

Reason:
- the old mono initializer was visually bootstrapped but IMU-closed
- for monocular-only, the visual SFM reconstruction is the final initializer

Porting guidance:
- split your initializer into:
  - visual reconstruction
  - optional IMU alignment
- do not call IMU alignment in pure monocular mode
- after the visual reconstruction, explicitly propagate the solved poses into the estimator state arrays used by the nonlinear backend
- retriangulate feature depths from those poses before entering the backend

### 7. Add A Scale-Gauge Anchor For Pure Monocular Optimization

File:
- `src/vins_estimator/estimator/estimator.cpp`

New factor:
- `TranslationNormFactor`

What it does:
- constrains the distance between pose 0 and pose 1 to remain equal to the current baseline

Where it is used:
- in the live Ceres problem during `optimization()`
- in old-frame marginalization during `MARGIN_OLD`

Why it is needed:
- in visual-inertial mode, scale is observed through IMU
- in stereo-only mode, scale is observed through stereo baseline
- in monocular-only mode, global scale is unobservable
- fixing pose 0 alone removes only a subset of gauge freedoms
- the backend still has a free scale mode unless an additional constraint is added

Implementation details:
- after `problem.SetParameterBlockConstant(para_Pose[0])`, if:
  - `!params.use_imu`
  - `!params.stereo`
  - `frame_count > 0`
- compute:
  - `current_baseline = max((Ps[1] - Ps[0]).norm(), 1e-3)`
- add:
  - `TranslationNormFactor::Create(current_baseline, 1.0)`

Same logic is repeated in marginalization:
- when marginalizing the oldest frame, add the same residual to `marginalization_info`
- drop set `{0}` is used so the anchor remains compatible with the marginalization layout

Reason:
- without this, the monocular-only optimization is ill-posed
- even if the live problem seems stable, scale ambiguity can re-enter through marginalization priors

Porting guidance:
- if your other repo has no explicit monocular scale handling, add one
- the simplest acceptable fix is a single baseline-norm anchor between two early poses
- add it both:
  - to the active optimization problem
  - to the marginalization/prior-building path

Important caveat:
- this anchor does not make the result metric
- it only removes the singular gauge direction by freezing the arbitrary startup scale

### 8. Example Config For Visual-Only Monocular

File:
- `config/euroc/euroc_mono_config.yaml`

Key settings:
- `imu: 0`
- `num_of_cam: 1`
- `estimate_extrinsic: 0`
- `estimate_td: 0`

Reason:
- with no IMU, there is no meaningful IMU-camera extrinsic calibration step
- there is no IMU/image time-offset estimation either

Porting guidance:
- create one clean config dedicated to monocular-only mode
- keep IMU noise parameters only if the parser requires them for compatibility
- document clearly that scale remains arbitrary unless additional metric priors are enabled

## What Was Not Changed

These parts were intentionally not rewritten:
- `initial_sfm.cpp`
- the actual `GlobalSFM` implementation
- the standard mono+IMU path
- the stereo+IMU path
- the stereo-only path

This is important for porting:
- do not over-edit the initializer if your existing SFM already works
- the real porting target is the estimator glue around SFM, not necessarily the SFM core

## Behavior Of The Final Monocular-Only Mode

After the change, the runtime behavior is:

1. images are accepted with no IMU subscription requirement
2. no IMU preintegration is created or attached to frames
3. once the window is full, the estimator runs visual global SFM + PnP
4. the result is copied into estimator pose arrays
5. feature depths are retriangulated from those visual poses
6. the solver enters nonlinear mode
7. optimization runs with:
   - pose 0 fixed
   - a monocular baseline norm anchor between pose 0 and pose 1
8. marginalization preserves that same scale anchor

## Porting Checklist

If you want to replicate this in another repo, the minimum checklist is:

1. Make `stereo` and `use_imu` explicitly initialized to zero.
2. Reset `stereo = 0` before parsing `num_of_cam`.
3. Allow `mono && !imu` as a valid sensor configuration.
4. Stop allocating or storing IMU preintegrations when `use_imu == 0`.
5. Add a dedicated monocular-only initialization branch in the main image-processing path.
6. Reuse visual SFM + PnP for startup.
7. Skip `visualInitialAlign()` / `VisualIMUAlignment(...)` when IMU is disabled.
8. Copy visual initialization poses into the estimator state arrays used by the nonlinear backend.
9. Retriangulate feature depths after those poses are copied into state.
10. Add a scale anchor to the active monocular optimization problem.
11. Add the same scale anchor to the marginalization/prior path.
12. Add a clean monocular-only config file.

## Recommended Validation After Porting

At minimum, validate:
- the estimator starts in `mono + no IMU` mode without null-pointer access
- initialization succeeds with enough parallax
- the solver transitions from `INITIAL` to `NON_LINEAR`
- marginalization remains numerically stable over time
- trajectories are consistent up to an arbitrary global scale

Useful failure signatures:
- crashes around `pre_integration`
- startup still calling `VisualIMUAlignment(...)`
- Ceres rank deficiency or unstable priors after several window slides
- initialization finishing, but depths never becoming consistent because state arrays were not copied back correctly

## Final Caveat

This implementation gives you a working visual-only monocular mode, not a metric monocular SLAM solution.

If the other repo requires metric scale, you still need one of:
- stereo baseline
- IMU
- depth priors
- another external metric cue

Without that, the best you can do is keep the arbitrary scale internally consistent, which is what the added translation-norm anchor does.
