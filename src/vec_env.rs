use crate::cartridge::Cartridge;
use crate::emulator::{
    MarioAction, NesEmulator, FRAME_PIXELS_RGB, NES_HEIGHT, NES_WIDTH, RGB_CHANNELS,
};
use rayon::prelude::*;

const PARALLEL_ENV_THRESHOLD: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct VecEnvConfig {
    pub num_envs: usize,
    pub frame_skip: usize,
    pub grayscale: bool,
    pub frame_stack: usize,
    pub terminate_on_flag: bool,
    pub crop_top: usize,
    pub crop_bottom: usize,
    pub resize_width: usize,
    pub resize_height: usize,
}

impl VecEnvConfig {
    pub fn source_height(&self) -> usize {
        NES_HEIGHT - self.crop_top - self.crop_bottom
    }

    pub fn obs_width(&self) -> usize {
        self.resize_width
    }

    pub fn obs_height(&self) -> usize {
        self.resize_height
    }

    pub fn channels(&self) -> usize {
        if self.grayscale {
            self.frame_stack
        } else {
            self.frame_stack * RGB_CHANNELS
        }
    }

    pub fn obs_len_per_env(&self) -> usize {
        self.channels() * self.obs_height() * self.obs_width()
    }

    fn needs_resize(&self) -> bool {
        self.resize_width != NES_WIDTH || self.resize_height != self.source_height()
    }
}

pub struct MarioVecEnv {
    config: VecEnvConfig,
    resize_plan: AreaResizePlan,
    envs: Vec<NesEmulator>,
    scratch: Vec<Vec<u8>>,
    synced_lanes: bool,
}

impl MarioVecEnv {
    pub fn new(cart: Cartridge, config: VecEnvConfig) -> Self {
        let resize_plan = AreaResizePlan::new(
            NES_WIDTH,
            config.source_height(),
            config.resize_width,
            config.resize_height,
        );
        let scratch_len = if config.needs_resize() {
            native_frame_len(config)
        } else {
            0
        };
        let envs = (0..config.num_envs)
            .map(|_| NesEmulator::new_with_options(cart.clone(), config.terminate_on_flag))
            .collect::<Vec<_>>();
        let scratch = (0..config.num_envs)
            .map(|_| vec![0; scratch_len])
            .collect::<Vec<_>>();
        Self {
            config,
            resize_plan,
            envs,
            scratch,
            synced_lanes: true,
        }
    }

    pub fn config(&self) -> VecEnvConfig {
        self.config
    }

    pub fn reset_into(&mut self, obs: &mut [u8]) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        self.synced_lanes = true;
        if config.num_envs > 1 {
            self.envs[0].reset();
            write_reset_stack(
                config,
                &self.resize_plan,
                &self.envs[0],
                &mut self.scratch[0],
                &mut obs[..obs_stride],
            );
            copy_first_obs_to_remaining(obs, obs_stride);
            return;
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .for_each(|((env, scratch), obs_chunk)| {
                    env.reset();
                    write_reset_stack(config, resize_plan, env, scratch, obs_chunk);
                });
        } else {
            for env_idx in 0..config.num_envs {
                self.envs[env_idx].reset();
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                write_reset_stack(
                    config,
                    &self.resize_plan,
                    &self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    &mut obs[start..end],
                );
            }
        }
    }

    pub fn step_into(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        lives: &mut [u8],
    ) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        if self.synced_lanes && config.num_envs > 1 {
            let first_action = actions[0];
            if actions.iter().all(|&action| action == first_action) {
                step_one(
                    config,
                    &self.resize_plan,
                    &mut self.envs[0],
                    &mut self.scratch[0],
                    first_action,
                    &mut obs[..obs_stride],
                    &mut rewards[0],
                    &mut terminated[0],
                    &mut truncated[0],
                    &mut x_pos[0],
                    &mut lives[0],
                );
                copy_first_obs_to_remaining(obs, obs_stride);
                rewards.fill(rewards[0]);
                terminated.fill(terminated[0]);
                truncated.fill(truncated[0]);
                x_pos.fill(x_pos[0]);
                lives.fill(lives[0]);
                return;
            }

            self.materialize_synced_lanes();
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(actions.par_iter())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(lives.par_iter_mut())
                .for_each(
                    |(
                        (
                            (
                                (
                                    ((((env, scratch), action), obs_chunk), reward_out),
                                    terminated_out,
                                ),
                                truncated_out,
                            ),
                            x_out,
                        ),
                        lives_out,
                    )| {
                        step_one(
                            config,
                            resize_plan,
                            env,
                            scratch,
                            *action,
                            obs_chunk,
                            reward_out,
                            terminated_out,
                            truncated_out,
                            x_out,
                            lives_out,
                        );
                    },
                );
        } else {
            for env_idx in 0..config.num_envs {
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                step_one(
                    config,
                    &self.resize_plan,
                    &mut self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    actions[env_idx],
                    &mut obs[start..end],
                    &mut rewards[env_idx],
                    &mut terminated[env_idx],
                    &mut truncated[env_idx],
                    &mut x_pos[env_idx],
                    &mut lives[env_idx],
                );
            }
        }
    }

    fn materialize_synced_lanes(&mut self) {
        let env = self.envs[0].clone();
        for lane in self.envs.iter_mut().skip(1) {
            *lane = env.clone();
        }
        self.synced_lanes = false;
    }
}

fn copy_first_obs_to_remaining(obs: &mut [u8], obs_stride: usize) {
    let (first, rest) = obs.split_at_mut(obs_stride);
    for chunk in rest.chunks_exact_mut(obs_stride) {
        chunk.copy_from_slice(first);
    }
}

fn step_one(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &mut NesEmulator,
    scratch: &mut [u8],
    action_id: u8,
    obs_chunk: &mut [u8],
    reward_out: &mut f32,
    terminated_out: &mut bool,
    truncated_out: &mut bool,
    x_out: &mut u16,
    lives_out: &mut u8,
) {
    let action = MarioAction::from_u8(action_id);
    let mut reward = 0.0;
    for _ in 0..config.frame_skip {
        reward += env.step_frame(action);
        if env.is_done() {
            break;
        }
    }
    shift_stack_left(config, obs_chunk);
    write_current_frame_to_last_stack_slot(config, resize_plan, env, scratch, obs_chunk);

    *reward_out = reward;
    *terminated_out = env.is_done();
    *truncated_out = false;
    *x_out = env.x_pos();
    *lives_out = env.lives();
}

fn write_reset_stack(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    obs_chunk: &mut [u8],
) {
    let frame_len = frame_len(config);
    for stack_i in 0..config.frame_stack {
        let dst_start = stack_i * frame_len;
        let dst_end = dst_start + frame_len;
        write_current_frame(
            config,
            resize_plan,
            env,
            scratch,
            &mut obs_chunk[dst_start..dst_end],
        );
    }
}

fn shift_stack_left(config: VecEnvConfig, obs_chunk: &mut [u8]) {
    if config.frame_stack <= 1 {
        return;
    }

    let frame_len = frame_len(config);
    let move_len = (config.frame_stack - 1) * frame_len;
    obs_chunk.copy_within(frame_len..frame_len + move_len, 0);
}

fn write_current_frame_to_last_stack_slot(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    obs_chunk: &mut [u8],
) {
    let frame_len = frame_len(config);
    let dst_start = (config.frame_stack - 1) * frame_len;
    let dst_end = dst_start + frame_len;
    write_current_frame(
        config,
        resize_plan,
        env,
        scratch,
        &mut obs_chunk[dst_start..dst_end],
    );
}

fn write_current_frame(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    dst: &mut [u8],
) {
    if config.needs_resize() {
        let native_len = native_frame_len(config);
        let native = &mut scratch[..native_len];
        write_native_frame(config, env, native);
        resize_frame_area(config, resize_plan, native, dst);
    } else {
        write_native_frame(config, env, dst);
    }
}

fn write_native_frame(config: VecEnvConfig, env: &NesEmulator, dst: &mut [u8]) {
    let height = config.source_height();
    if config.grayscale {
        if config.crop_top == 0 && config.crop_bottom == 0 {
            env.write_gray_frame(dst);
        } else {
            env.write_gray_frame_cropped(dst, config.crop_top, height);
        }
    } else if config.crop_top == 0 && config.crop_bottom == 0 {
        env.write_rgb_frame(dst);
    } else {
        env.write_rgb_frame_cropped(dst, config.crop_top, height);
    }
}

#[inline]
fn frame_len(config: VecEnvConfig) -> usize {
    if config.grayscale {
        config.obs_width() * config.obs_height()
    } else {
        config.obs_width() * config.obs_height() * RGB_CHANNELS
    }
}

#[inline]
fn native_frame_len(config: VecEnvConfig) -> usize {
    if config.grayscale {
        NES_WIDTH * config.source_height()
    } else if config.crop_top == 0 && config.crop_bottom == 0 {
        FRAME_PIXELS_RGB
    } else {
        NES_WIDTH * config.source_height() * RGB_CHANNELS
    }
}

fn resize_frame_area(config: VecEnvConfig, plan: &AreaResizePlan, src: &[u8], dst: &mut [u8]) {
    if config.grayscale {
        resize_plane_area(src, dst, plan, 0, 0);
    } else {
        let src_plane = plan.src_width * plan.src_height;
        let dst_plane = plan.dst_width * plan.dst_height;
        for channel in 0..RGB_CHANNELS {
            resize_plane_area(src, dst, plan, channel * src_plane, channel * dst_plane);
        }
    }
}

fn resize_plane_area(
    src: &[u8],
    dst: &mut [u8],
    plan: &AreaResizePlan,
    src_offset: usize,
    dst_offset: usize,
) {
    let rounding = plan.denom / 2;
    for (dst_y, y_bin) in plan.y_bins.iter().enumerate() {
        for (dst_x, x_bin) in plan.x_bins.iter().enumerate() {
            let mut sum = 0u64;
            for (dy, &wy) in y_bin.weights.iter().enumerate() {
                let src_row = src_offset + (y_bin.start + dy) * plan.src_width;
                let wy = wy as u64;
                for (dx, &wx) in x_bin.weights.iter().enumerate() {
                    let weight = wy * wx as u64;
                    sum += src[src_row + x_bin.start + dx] as u64 * weight;
                }
            }
            dst[dst_offset + dst_y * plan.dst_width + dst_x] =
                ((sum + rounding) / plan.denom) as u8;
        }
    }
}

struct AreaResizePlan {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    x_bins: Vec<AreaAxisBin>,
    y_bins: Vec<AreaAxisBin>,
    denom: u64,
}

impl AreaResizePlan {
    fn new(src_width: usize, src_height: usize, dst_width: usize, dst_height: usize) -> Self {
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            x_bins: build_area_axis(src_width, dst_width),
            y_bins: build_area_axis(src_height, dst_height),
            denom: (src_width as u64) * (src_height as u64),
        }
    }
}

struct AreaAxisBin {
    start: usize,
    weights: Vec<u32>,
}

fn build_area_axis(src_len: usize, dst_len: usize) -> Vec<AreaAxisBin> {
    (0..dst_len)
        .map(|dst_i| {
            let start_num = dst_i * src_len;
            let end_num = (dst_i + 1) * src_len;
            let start = start_num / dst_len;
            let end = (end_num + dst_len - 1) / dst_len;
            let weights = (start..end)
                .map(|src_i| {
                    let pixel_start = src_i * dst_len;
                    let pixel_end = (src_i + 1) * dst_len;
                    pixel_end
                        .min(end_num)
                        .saturating_sub(pixel_start.max(start_num)) as u32
                })
                .collect::<Vec<_>>();
            AreaAxisBin { start, weights }
        })
        .collect()
}
