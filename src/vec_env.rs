use crate::cartridge::Cartridge;
use crate::emulator::{
    MarioAction, NesEmulator, StateLoadError, FRAME_PIXELS_RGB, NES_HEIGHT, NES_WIDTH, RGB_CHANNELS,
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
    initial_state: Option<Vec<u8>>,
    scratch: Vec<Vec<u8>>,
    synced_lanes: bool,
}

impl MarioVecEnv {
    pub fn new(
        cart: Cartridge,
        config: VecEnvConfig,
        initial_state: Option<Vec<u8>>,
    ) -> Result<Self, StateLoadError> {
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
        let mut env = Self {
            config,
            resize_plan,
            envs,
            initial_state,
            scratch,
            synced_lanes: true,
        };
        env.reset_envs()?;
        Ok(env)
    }

    fn reset_envs(&mut self) -> Result<(), StateLoadError> {
        if let Some(initial_state) = self.initial_state.as_deref() {
            for env in &mut self.envs {
                env.load_fceu_state(initial_state)?;
            }
            return Ok(());
        }

        for env in &mut self.envs {
            env.reset();
        }
        Ok(())
    }

    fn reset_env(
        env: &mut NesEmulator,
        initial_state: Option<&[u8]>,
    ) -> Result<(), StateLoadError> {
        if let Some(initial_state) = initial_state {
            env.load_fceu_state(initial_state)
        } else {
            env.reset();
            Ok(())
        }
    }

    pub fn config(&self) -> VecEnvConfig {
        self.config
    }

    pub fn reset_into(&mut self, obs: &mut [u8]) -> Result<(), StateLoadError> {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        self.synced_lanes = true;
        if config.num_envs > 1 {
            Self::reset_env(&mut self.envs[0], self.initial_state.as_deref())?;
            write_reset_stack(
                config,
                &self.resize_plan,
                &self.envs[0],
                &mut self.scratch[0],
                &mut obs[..obs_stride],
            );
            copy_first_obs_to_remaining(obs, obs_stride);
            return Ok(());
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            let initial_state = self.initial_state.as_deref();
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .try_for_each(|((env, scratch), obs_chunk)| {
                    Self::reset_env(env, initial_state)?;
                    write_reset_stack(config, resize_plan, env, scratch, obs_chunk);
                    Ok::<(), StateLoadError>(())
                })?;
        } else {
            for env_idx in 0..config.num_envs {
                Self::reset_env(&mut self.envs[env_idx], self.initial_state.as_deref())?;
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
        Ok(())
    }

    pub fn info_into(&self, x_pos: &mut [u16], lives: &mut [u8]) {
        if self.synced_lanes && self.config.num_envs > 1 {
            x_pos.fill(self.envs[0].x_pos());
            lives.fill(self.envs[0].lives());
            return;
        }

        for ((env, x_out), lives_out) in
            self.envs.iter().zip(x_pos.iter_mut()).zip(lives.iter_mut())
        {
            *x_out = env.x_pos();
            *lives_out = env.lives();
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
    debug_assert!(src.len() >= src_offset + plan.src_width * plan.src_height);
    debug_assert!(dst.len() >= dst_offset + plan.dst_width * plan.dst_height);

    for dst_i in 0..plan.pixel_starts.len() - 1 {
        let mut sum = 0u64;
        let start = plan.pixel_starts[dst_i];
        let end = plan.pixel_starts[dst_i + 1];
        for contribution in &plan.contributions[start..end] {
            // SAFETY: AreaResizePlan contributions are built from dimensions validated above.
            let source = unsafe { *src.get_unchecked(src_offset + contribution.src_offset) };
            sum += source as u64 * contribution.weight as u64;
        }
        // SAFETY: dst_i iterates over exactly dst_width * dst_height planned pixels.
        unsafe {
            *dst.get_unchecked_mut(dst_offset + dst_i) = ((sum + rounding) / plan.denom) as u8;
        }
    }
}

struct AreaResizePlan {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    pixel_starts: Vec<usize>,
    contributions: Vec<AreaContribution>,
    denom: u64,
}

impl AreaResizePlan {
    fn new(src_width: usize, src_height: usize, dst_width: usize, dst_height: usize) -> Self {
        let x_bins = build_area_axis(src_width, dst_width);
        let y_bins = build_area_axis(src_height, dst_height);
        let (pixel_starts, contributions) = build_area_pixels(src_width, &x_bins, &y_bins);
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            pixel_starts,
            contributions,
            denom: (src_width as u64) * (src_height as u64),
        }
    }
}

struct AreaContribution {
    src_offset: usize,
    weight: u32,
}

struct AreaAxisBin {
    start: usize,
    weights: Vec<u32>,
}

fn build_area_pixels(
    src_width: usize,
    x_bins: &[AreaAxisBin],
    y_bins: &[AreaAxisBin],
) -> (Vec<usize>, Vec<AreaContribution>) {
    let mut pixel_starts = Vec::with_capacity(x_bins.len() * y_bins.len() + 1);
    let mut contributions = Vec::new();
    for y_bin in y_bins {
        for x_bin in x_bins {
            pixel_starts.push(contributions.len());
            for (dy, &wy) in y_bin.weights.iter().enumerate() {
                let src_row = (y_bin.start + dy) * src_width;
                for (dx, &wx) in x_bin.weights.iter().enumerate() {
                    let weight = wy * wx;
                    if weight != 0 {
                        contributions.push(AreaContribution {
                            src_offset: src_row + x_bin.start + dx,
                            weight,
                        });
                    }
                }
            }
        }
    }
    pixel_starts.push(contributions.len());
    (pixel_starts, contributions)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_resize_plane_area(
        src: &[u8],
        dst: &mut [u8],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        src_offset: usize,
        dst_offset: usize,
    ) {
        let x_bins = build_area_axis(src_width, dst_width);
        let y_bins = build_area_axis(src_height, dst_height);
        let denom = (src_width as u64) * (src_height as u64);
        let rounding = denom / 2;

        for (dst_y, y_bin) in y_bins.iter().enumerate() {
            for (dst_x, x_bin) in x_bins.iter().enumerate() {
                let mut sum = 0u64;
                for (dy, &wy) in y_bin.weights.iter().enumerate() {
                    let src_row = src_offset + (y_bin.start + dy) * src_width;
                    let wy = wy as u64;
                    for (dx, &wx) in x_bin.weights.iter().enumerate() {
                        let weight = wy * wx as u64;
                        sum += src[src_row + x_bin.start + dx] as u64 * weight;
                    }
                }
                dst[dst_offset + dst_y * dst_width + dst_x] = ((sum + rounding) / denom) as u8;
            }
        }
    }

    #[test]
    fn precomputed_area_resize_matches_reference_default_grayscale() {
        let config = VecEnvConfig {
            num_envs: 16,
            frame_skip: 4,
            grayscale: true,
            frame_stack: 4,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 0,
            resize_width: 84,
            resize_height: 84,
        };
        let plan = AreaResizePlan::new(NES_WIDTH, config.source_height(), 84, 84);
        let src_len = NES_WIDTH * config.source_height();
        let src = (0..src_len)
            .map(|idx| ((idx * 37 + idx / 251 + 19) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut optimized = vec![0; 84 * 84];
        let mut reference = vec![0; 84 * 84];

        resize_frame_area(config, &plan, &src, &mut optimized);
        reference_resize_plane_area(
            &src,
            &mut reference,
            NES_WIDTH,
            config.source_height(),
            84,
            84,
            0,
            0,
        );

        assert_eq!(optimized, reference);
    }

    #[test]
    fn precomputed_area_resize_matches_reference_rgb_planes() {
        let src_width = 256;
        let src_height = 208;
        let dst_width = 84;
        let dst_height = 84;
        let config = VecEnvConfig {
            num_envs: 1,
            frame_skip: 4,
            grayscale: false,
            frame_stack: 1,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 0,
            resize_width: dst_width,
            resize_height: dst_height,
        };
        let plan = AreaResizePlan::new(src_width, src_height, dst_width, dst_height);
        let src_plane = src_width * src_height;
        let dst_plane = dst_width * dst_height;
        let src = (0..src_plane * RGB_CHANNELS)
            .map(|idx| ((idx * 17 + idx / 97 + 31) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut optimized = vec![0; dst_plane * RGB_CHANNELS];
        let mut reference = vec![0; dst_plane * RGB_CHANNELS];

        resize_frame_area(config, &plan, &src, &mut optimized);
        for channel in 0..RGB_CHANNELS {
            reference_resize_plane_area(
                &src,
                &mut reference,
                src_width,
                src_height,
                dst_width,
                dst_height,
                channel * src_plane,
                channel * dst_plane,
            );
        }

        assert_eq!(optimized, reference);
    }
}
