use crate::cartridge::Cartridge;
use crate::emulator::{
    MarioAction, NesEmulator, StateLoadError, RGB_CHANNELS, VISIBLE_FRAME_HEIGHT,
    VISIBLE_FRAME_WIDTH,
};
use crate::profiler::Profiler;
use rayon::prelude::*;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfoKey {
    XPos,
    Coins,
    LevelHi,
    LevelLo,
    Lives,
    Score,
    Scrolling,
    Time,
    XScrollHi,
    XScrollLo,
}

impl InfoKey {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "x_pos" => Some(Self::XPos),
            "coins" => Some(Self::Coins),
            "levelHi" => Some(Self::LevelHi),
            "levelLo" => Some(Self::LevelLo),
            "lives" => Some(Self::Lives),
            "score" => Some(Self::Score),
            "scrolling" => Some(Self::Scrolling),
            "time" => Some(Self::Time),
            "xscrollHi" => Some(Self::XScrollHi),
            "xscrollLo" => Some(Self::XScrollLo),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::XPos => "x_pos",
            Self::Coins => "coins",
            Self::LevelHi => "levelHi",
            Self::LevelLo => "levelLo",
            Self::Lives => "lives",
            Self::Score => "score",
            Self::Scrolling => "scrolling",
            Self::Time => "time",
            Self::XScrollHi => "xscrollHi",
            Self::XScrollLo => "xscrollLo",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoneOnInfoOp {
    Change,
    Increase,
    Decrease,
}

impl DoneOnInfoOp {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "change" => Some(Self::Change),
            "increase" => Some(Self::Increase),
            "decrease" => Some(Self::Decrease),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Change => "change",
            Self::Increase => "increase",
            Self::Decrease => "decrease",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DoneOnInfoRule {
    pub name: String,
    pub keys: Vec<InfoKey>,
    pub op: DoneOnInfoOp,
}

#[derive(Clone, Debug)]
pub struct FiredDoneOnInfoRule {
    pub name: String,
    pub keys: Vec<InfoKey>,
    pub op: DoneOnInfoOp,
    pub previous_values: Vec<i64>,
    pub current_values: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct InfoSnapshot {
    x_pos: i64,
    coins: i64,
    level_hi: i64,
    level_lo: i64,
    lives: i64,
    score: i64,
    scrolling: i64,
    time: i64,
    xscroll_hi: i64,
    xscroll_lo: i64,
}

impl InfoSnapshot {
    fn from_env(env: &NesEmulator) -> Self {
        Self {
            x_pos: i64::from(env.x_pos()),
            coins: i64::from(env.coins()),
            level_hi: i64::from(env.level_hi()),
            level_lo: i64::from(env.level_lo()),
            lives: i64::from(env.lives()),
            score: i64::from(env.score()),
            scrolling: i64::from(env.scrolling()),
            time: i64::from(env.time()),
            xscroll_hi: i64::from(env.xscroll_hi()),
            xscroll_lo: i64::from(env.xscroll_lo()),
        }
    }

    fn value(self, key: InfoKey) -> i64 {
        match key {
            InfoKey::XPos => self.x_pos,
            InfoKey::Coins => self.coins,
            InfoKey::LevelHi => self.level_hi,
            InfoKey::LevelLo => self.level_lo,
            InfoKey::Lives => self.lives,
            InfoKey::Score => self.score,
            InfoKey::Scrolling => self.scrolling,
            InfoKey::Time => self.time,
            InfoKey::XScrollHi => self.xscroll_hi,
            InfoKey::XScrollLo => self.xscroll_lo,
        }
    }
}

impl VecEnvConfig {
    pub fn source_width(&self) -> usize {
        VISIBLE_FRAME_WIDTH
    }

    pub fn source_height(&self) -> usize {
        VISIBLE_FRAME_HEIGHT - self.crop_top - self.crop_bottom
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
        self.resize_width != self.source_width() || self.resize_height != self.source_height()
    }

    fn uses_default_gray_area_resize(&self) -> bool {
        false
    }
}

pub struct MarioVecEnv {
    config: VecEnvConfig,
    resize_plan: AreaResizePlan,
    envs: Vec<NesEmulator>,
    initial_states: Vec<InitialState>,
    weighted_initial_states: bool,
    active_state_indices: Vec<i32>,
    done_on_info_rules: Vec<DoneOnInfoRule>,
    done_on_info_baselines: Vec<InfoSnapshot>,
    last_done_on_info: Vec<Vec<FiredDoneOnInfoRule>>,
    rng: XorShift64,
    scratch: Vec<Vec<u8>>,
    synced_lanes: bool,
    synced_groups: Vec<Vec<usize>>,
    profiler: Option<Profiler>,
    profile_shards: Vec<Profiler>,
}

impl MarioVecEnv {
    pub fn new(
        cart: Cartridge,
        config: VecEnvConfig,
        initial_states: Vec<InitialState>,
        weighted_initial_states: bool,
        seed: u64,
        done_on_info_rules: Vec<DoneOnInfoRule>,
    ) -> Result<Self, StateLoadError> {
        let resize_plan = AreaResizePlan::new(
            config.source_width(),
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
            initial_states,
            weighted_initial_states,
            active_state_indices: vec![-1; config.num_envs],
            done_on_info_rules,
            done_on_info_baselines: vec![InfoSnapshot::default(); config.num_envs],
            last_done_on_info: vec![Vec::new(); config.num_envs],
            rng: XorShift64::new(seed),
            scratch,
            synced_lanes: true,
            synced_groups: Vec::new(),
            profiler: None,
            profile_shards: Vec::new(),
        };
        env.reset_envs()?;
        Ok(env)
    }

    fn reset_envs(&mut self) -> Result<(), StateLoadError> {
        if self.initial_states.is_empty() {
            for env_idx in 0..self.config.num_envs {
                self.envs[env_idx].reset();
                self.refresh_done_baseline(env_idx);
            }
            self.active_state_indices.fill(-1);
            self.synced_lanes = true;
            self.synced_groups.clear();
            return Ok(());
        }

        let mut first_state_index: Option<usize> = None;
        let mut all_same = true;
        for env_idx in 0..self.config.num_envs {
            let state_index = self.initial_state_index_for_env(env_idx);
            if let Some(first) = first_state_index {
                all_same &= first == state_index;
            } else {
                first_state_index = Some(state_index);
            }
            self.active_state_indices[env_idx] = state_index as i32;
            self.envs[env_idx].load_fceu_state(&self.initial_states[state_index].data)?;
            self.refresh_done_baseline(env_idx);
        }
        self.synced_lanes = all_same;
        if self.synced_lanes {
            self.synced_groups.clear();
        } else {
            self.refresh_synced_groups();
        }
        Ok(())
    }

    fn initial_state_index_for_env(&mut self, env_idx: usize) -> usize {
        if self.weighted_initial_states {
            let sample = self.rng.next_unit_f64();
            for (idx, state) in self.initial_states.iter().enumerate() {
                if sample < state.cumulative_weight {
                    return idx;
                }
            }
            self.initial_states.len() - 1
        } else if self.initial_states.len() == 1 {
            0
        } else {
            env_idx
        }
    }

    pub fn config(&self) -> VecEnvConfig {
        self.config
    }

    pub fn reset_into(&mut self, obs: &mut [u8]) -> Result<(), StateLoadError> {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        self.reset_envs()?;
        for lane in &mut self.last_done_on_info {
            lane.clear();
        }
        if self.synced_lanes && config.num_envs > 1 {
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
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .for_each(|((env, scratch), obs_chunk)| {
                    write_reset_stack(config, resize_plan, env, scratch, obs_chunk);
                });
        } else {
            for env_idx in 0..config.num_envs {
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

    pub fn initial_state_names(&self) -> Vec<String> {
        self.initial_states
            .iter()
            .map(|state| state.name.clone())
            .collect()
    }

    pub fn active_state_indices(&self) -> &[i32] {
        &self.active_state_indices
    }

    pub fn done_on_info(&self) -> &[Vec<FiredDoneOnInfoRule>] {
        &self.last_done_on_info
    }

    pub fn seed(&mut self, seed: u64) {
        self.rng = XorShift64::new(seed);
    }

    pub fn enable_profiler(&mut self) {
        self.profiler = Some(Profiler::new());
        self.profile_shards = (0..self.config.num_envs).map(|_| Profiler::new()).collect();
    }

    pub fn reset_profiler(&mut self) {
        if let Some(profiler) = &mut self.profiler {
            profiler.clear();
        }
        for shard in &mut self.profile_shards {
            shard.clear();
        }
    }

    pub fn disable_profiler(&mut self) {
        self.profiler = None;
        self.profile_shards.clear();
    }

    pub fn profiler_snapshot_json(&self, top_n: usize) -> Option<String> {
        let mut merged = self.profiler.clone()?;
        for shard in &self.profile_shards {
            merged.add(shard);
        }
        Some(merged.to_json(top_n))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn info_into(
        &self,
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        if self.synced_lanes && self.config.num_envs > 1 {
            fill_info_from_env(
                &self.envs[0],
                x_pos,
                coins,
                level_hi,
                level_lo,
                lives,
                score,
                scrolling,
                time,
                xscroll_hi,
                xscroll_lo,
            );
            return;
        }
        if !self.synced_groups.is_empty() {
            for group in &self.synced_groups {
                let first = group[0];
                write_info_from_env(
                    &self.envs[first],
                    &mut x_pos[first],
                    &mut coins[first],
                    &mut level_hi[first],
                    &mut level_lo[first],
                    &mut lives[first],
                    &mut score[first],
                    &mut scrolling[first],
                    &mut time[first],
                    &mut xscroll_hi[first],
                    &mut xscroll_lo[first],
                );
                for &lane in group.iter().skip(1) {
                    copy_info_lane(
                        first, lane, x_pos, coins, level_hi, level_lo, lives, score, scrolling,
                        time, xscroll_hi, xscroll_lo,
                    );
                }
            }
            return;
        }

        for env_idx in 0..self.config.num_envs {
            write_info_from_env(
                &self.envs[env_idx],
                &mut x_pos[env_idx],
                &mut coins[env_idx],
                &mut level_hi[env_idx],
                &mut level_lo[env_idx],
                &mut lives[env_idx],
                &mut score[env_idx],
                &mut scrolling[env_idx],
                &mut time[env_idx],
                &mut xscroll_hi[env_idx],
                &mut xscroll_lo[env_idx],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_into(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        if self.profiler.is_some() {
            self.step_into_profiled(
                actions, obs, rewards, terminated, truncated, x_pos, coins, level_hi, level_lo,
                lives, score, scrolling, time, xscroll_hi, xscroll_lo,
            );
            return;
        }

        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        for lane in &mut self.last_done_on_info {
            lane.clear();
        }
        if self.synced_lanes && config.num_envs > 1 && !self.done_on_info_rules.is_empty() {
            self.materialize_synced_lanes();
        }
        if !self.synced_groups.is_empty() && !self.done_on_info_rules.is_empty() {
            self.materialize_synced_groups();
        }
        if self.synced_lanes && config.num_envs > 1 {
            let first_action = actions[0];
            if actions.iter().all(|&action| action == first_action) {
                step_one(
                    config,
                    &self.resize_plan,
                    &mut self.envs[0],
                    &mut self.scratch[0],
                    self.done_on_info_baselines[0],
                    &self.done_on_info_rules,
                    &mut self.last_done_on_info[0],
                    first_action,
                    &mut obs[..obs_stride],
                    &mut rewards[0],
                    &mut terminated[0],
                    &mut truncated[0],
                    &mut x_pos[0],
                    &mut coins[0],
                    &mut level_hi[0],
                    &mut level_lo[0],
                    &mut lives[0],
                    &mut score[0],
                    &mut scrolling[0],
                    &mut time[0],
                    &mut xscroll_hi[0],
                    &mut xscroll_lo[0],
                );
                copy_first_obs_to_remaining(obs, obs_stride);
                rewards.fill(rewards[0]);
                terminated.fill(terminated[0]);
                truncated.fill(truncated[0]);
                fill_info_from_first(
                    x_pos, coins, level_hi, level_lo, lives, score, scrolling, time, xscroll_hi,
                    xscroll_lo,
                );
                if terminated[0] || truncated[0] {
                    self.autoreset_done_lanes(
                        obs, terminated, truncated, x_pos, coins, level_hi, level_lo, lives, score,
                        scrolling, time, xscroll_hi, xscroll_lo,
                    );
                }
                return;
            }

            self.materialize_synced_lanes();
        }
        if !self.synced_groups.is_empty() {
            if self.synced_group_actions_are_uniform(actions) {
                self.step_synced_groups(
                    actions, obs, rewards, terminated, truncated, x_pos, coins, level_hi, level_lo,
                    lives, score, scrolling, time, xscroll_hi, xscroll_lo,
                );
                return;
            }
            self.materialize_synced_groups();
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(actions.par_iter())
                .zip(self.done_on_info_baselines.par_iter())
                .zip(self.last_done_on_info.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(coins.par_iter_mut())
                .zip(level_hi.par_iter_mut())
                .zip(level_lo.par_iter_mut())
                .zip(lives.par_iter_mut())
                .zip(score.par_iter_mut())
                .zip(scrolling.par_iter_mut())
                .zip(time.par_iter_mut())
                .zip(xscroll_hi.par_iter_mut())
                .zip(xscroll_lo.par_iter_mut())
                .for_each(|zipped| {
                    let (zipped, xscroll_lo_out) = zipped;
                    let (zipped, xscroll_hi_out) = zipped;
                    let (zipped, time_out) = zipped;
                    let (zipped, scrolling_out) = zipped;
                    let (zipped, score_out) = zipped;
                    let (zipped, lives_out) = zipped;
                    let (zipped, level_lo_out) = zipped;
                    let (zipped, level_hi_out) = zipped;
                    let (zipped, coins_out) = zipped;
                    let (zipped, x_out) = zipped;
                    let (zipped, truncated_out) = zipped;
                    let (zipped, terminated_out) = zipped;
                    let (zipped, reward_out) = zipped;
                    let (zipped, obs_chunk) = zipped;
                    let (zipped, fired_done_on_info) = zipped;
                    let (zipped, done_on_info_baseline) = zipped;
                    let ((env, scratch), action) = zipped;
                    step_one(
                        config,
                        resize_plan,
                        env,
                        scratch,
                        *done_on_info_baseline,
                        &self.done_on_info_rules,
                        fired_done_on_info,
                        *action,
                        obs_chunk,
                        reward_out,
                        terminated_out,
                        truncated_out,
                        x_out,
                        coins_out,
                        level_hi_out,
                        level_lo_out,
                        lives_out,
                        score_out,
                        scrolling_out,
                        time_out,
                        xscroll_hi_out,
                        xscroll_lo_out,
                    );
                });
        } else {
            for env_idx in 0..config.num_envs {
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                step_one(
                    config,
                    &self.resize_plan,
                    &mut self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    self.done_on_info_baselines[env_idx],
                    &self.done_on_info_rules,
                    &mut self.last_done_on_info[env_idx],
                    actions[env_idx],
                    &mut obs[start..end],
                    &mut rewards[env_idx],
                    &mut terminated[env_idx],
                    &mut truncated[env_idx],
                    &mut x_pos[env_idx],
                    &mut coins[env_idx],
                    &mut level_hi[env_idx],
                    &mut level_lo[env_idx],
                    &mut lives[env_idx],
                    &mut score[env_idx],
                    &mut scrolling[env_idx],
                    &mut time[env_idx],
                    &mut xscroll_hi[env_idx],
                    &mut xscroll_lo[env_idx],
                );
            }
        }

        if terminated.iter().any(|done| *done) || truncated.iter().any(|done| *done) {
            self.autoreset_done_lanes(
                obs, terminated, truncated, x_pos, coins, level_hi, level_lo, lives, score,
                scrolling, time, xscroll_hi, xscroll_lo,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_into_profiled(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        if self.profile_shards.len() != self.config.num_envs {
            self.profile_shards = (0..self.config.num_envs).map(|_| Profiler::new()).collect();
        }
        if let Some(profiler) = &mut self.profiler {
            profiler.record_batch_step(self.config.num_envs);
        }

        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        for lane in &mut self.last_done_on_info {
            lane.clear();
        }
        if self.synced_lanes && config.num_envs > 1 && !self.done_on_info_rules.is_empty() {
            self.materialize_synced_lanes_profiled();
        }
        if !self.synced_groups.is_empty() && !self.done_on_info_rules.is_empty() {
            self.materialize_synced_groups_profiled();
        }
        if self.synced_lanes && config.num_envs > 1 {
            let first_action = actions[0];
            if actions_are_uniform(actions, first_action) {
                step_one_profiled(
                    config,
                    &self.resize_plan,
                    &mut self.envs[0],
                    &mut self.scratch[0],
                    self.done_on_info_baselines[0],
                    &self.done_on_info_rules,
                    &mut self.last_done_on_info[0],
                    first_action,
                    &mut obs[..obs_stride],
                    &mut rewards[0],
                    &mut terminated[0],
                    &mut truncated[0],
                    &mut x_pos[0],
                    &mut coins[0],
                    &mut level_hi[0],
                    &mut level_lo[0],
                    &mut lives[0],
                    &mut score[0],
                    &mut scrolling[0],
                    &mut time[0],
                    &mut xscroll_hi[0],
                    &mut xscroll_lo[0],
                    &mut self.profile_shards[0],
                );
                let copy_start = Instant::now();
                copy_first_obs_to_remaining(obs, obs_stride);
                rewards.fill(rewards[0]);
                terminated.fill(terminated[0]);
                truncated.fill(truncated[0]);
                fill_info_from_first(
                    x_pos, coins, level_hi, level_lo, lives, score, scrolling, time, xscroll_hi,
                    xscroll_lo,
                );
                if let Some(profiler) = &mut self.profiler {
                    profiler.record_first_lane_broadcast(
                        config.num_envs - 1,
                        obs_stride * (config.num_envs - 1),
                    );
                    profiler.record_grouped_copy(
                        config.num_envs - 1,
                        obs_stride * (config.num_envs - 1),
                        copy_start.elapsed(),
                    );
                }
                if terminated[0] || truncated[0] {
                    self.autoreset_done_lanes(
                        obs, terminated, truncated, x_pos, coins, level_hi, level_lo, lives, score,
                        scrolling, time, xscroll_hi, xscroll_lo,
                    );
                }
                return;
            }

            self.materialize_synced_lanes_profiled();
        }
        if !self.synced_groups.is_empty() {
            if self.synced_group_actions_are_uniform(actions) {
                self.step_synced_groups_profiled(
                    actions, obs, rewards, terminated, truncated, x_pos, coins, level_hi, level_lo,
                    lives, score, scrolling, time, xscroll_hi, xscroll_lo,
                );
                return;
            }
            if let Some(profiler) = &mut self.profiler {
                profiler.record_group_miss();
            }
            self.materialize_synced_groups_profiled();
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(actions.par_iter())
                .zip(self.done_on_info_baselines.par_iter())
                .zip(self.last_done_on_info.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(coins.par_iter_mut())
                .zip(level_hi.par_iter_mut())
                .zip(level_lo.par_iter_mut())
                .zip(lives.par_iter_mut())
                .zip(score.par_iter_mut())
                .zip(scrolling.par_iter_mut())
                .zip(time.par_iter_mut())
                .zip(xscroll_hi.par_iter_mut())
                .zip(xscroll_lo.par_iter_mut())
                .zip(self.profile_shards.par_iter_mut())
                .for_each(|zipped| {
                    let (zipped, profiler) = zipped;
                    let (zipped, xscroll_lo_out) = zipped;
                    let (zipped, xscroll_hi_out) = zipped;
                    let (zipped, time_out) = zipped;
                    let (zipped, scrolling_out) = zipped;
                    let (zipped, score_out) = zipped;
                    let (zipped, lives_out) = zipped;
                    let (zipped, level_lo_out) = zipped;
                    let (zipped, level_hi_out) = zipped;
                    let (zipped, coins_out) = zipped;
                    let (zipped, x_out) = zipped;
                    let (zipped, truncated_out) = zipped;
                    let (zipped, terminated_out) = zipped;
                    let (zipped, reward_out) = zipped;
                    let (zipped, obs_chunk) = zipped;
                    let (zipped, fired_done_on_info) = zipped;
                    let (zipped, done_on_info_baseline) = zipped;
                    let ((env, scratch), action) = zipped;
                    step_one_profiled(
                        config,
                        resize_plan,
                        env,
                        scratch,
                        *done_on_info_baseline,
                        &self.done_on_info_rules,
                        fired_done_on_info,
                        *action,
                        obs_chunk,
                        reward_out,
                        terminated_out,
                        truncated_out,
                        x_out,
                        coins_out,
                        level_hi_out,
                        level_lo_out,
                        lives_out,
                        score_out,
                        scrolling_out,
                        time_out,
                        xscroll_hi_out,
                        xscroll_lo_out,
                        profiler,
                    );
                });
        } else {
            for env_idx in 0..config.num_envs {
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                step_one_profiled(
                    config,
                    &self.resize_plan,
                    &mut self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    self.done_on_info_baselines[env_idx],
                    &self.done_on_info_rules,
                    &mut self.last_done_on_info[env_idx],
                    actions[env_idx],
                    &mut obs[start..end],
                    &mut rewards[env_idx],
                    &mut terminated[env_idx],
                    &mut truncated[env_idx],
                    &mut x_pos[env_idx],
                    &mut coins[env_idx],
                    &mut level_hi[env_idx],
                    &mut level_lo[env_idx],
                    &mut lives[env_idx],
                    &mut score[env_idx],
                    &mut scrolling[env_idx],
                    &mut time[env_idx],
                    &mut xscroll_hi[env_idx],
                    &mut xscroll_lo[env_idx],
                    &mut self.profile_shards[env_idx],
                );
            }
        }

        if terminated.iter().any(|done| *done) || truncated.iter().any(|done| *done) {
            self.autoreset_done_lanes(
                obs, terminated, truncated, x_pos, coins, level_hi, level_lo, lives, score,
                scrolling, time, xscroll_hi, xscroll_lo,
            );
        }
    }

    fn materialize_synced_lanes(&mut self) {
        let env = self.envs[0].clone();
        for lane in self.envs.iter_mut().skip(1) {
            *lane = env.clone();
        }
        let active_state_index = self.active_state_indices[0];
        self.active_state_indices.fill(active_state_index);
        let done_on_info_baseline = self.done_on_info_baselines[0];
        self.done_on_info_baselines.fill(done_on_info_baseline);
        self.synced_lanes = false;
        self.synced_groups.clear();
    }

    fn materialize_synced_lanes_profiled(&mut self) {
        let start = Instant::now();
        let lanes = self.envs.len().saturating_sub(1);
        self.materialize_synced_lanes();
        if let Some(profiler) = &mut self.profiler {
            profiler.record_materialization(lanes, start.elapsed());
        }
    }

    fn refresh_synced_groups(&mut self) {
        self.synced_groups.clear();
        if self.initial_states.is_empty() || self.weighted_initial_states {
            return;
        }

        let mut groups: Vec<Vec<usize>> = Vec::new();
        'lanes: for lane in 0..self.config.num_envs {
            let state_index = self.active_state_indices[lane];
            if state_index < 0 {
                continue;
            }
            let state_index = state_index as usize;
            for group in &mut groups {
                let group_state_index = self.active_state_indices[group[0]] as usize;
                if self.initial_states[state_index].data
                    == self.initial_states[group_state_index].data
                {
                    group.push(lane);
                    continue 'lanes;
                }
            }
            groups.push(vec![lane]);
        }

        if groups.iter().any(|group| group.len() > 1) {
            self.synced_groups = groups;
        }
    }

    fn materialize_synced_groups(&mut self) {
        self.materialize_synced_groups_inner();
    }

    fn materialize_synced_groups_profiled(&mut self) {
        let lanes = self
            .synced_groups
            .iter()
            .map(|group| group.len().saturating_sub(1))
            .sum::<usize>();
        let start = Instant::now();
        self.materialize_synced_groups_inner();
        if let Some(profiler) = &mut self.profiler {
            profiler.record_materialization(lanes, start.elapsed());
        }
    }

    fn materialize_synced_groups_inner(&mut self) {
        for group in &self.synced_groups {
            let first = group[0];
            let env = self.envs[first].clone();
            let done_on_info_baseline = self.done_on_info_baselines[first];
            for &lane in group.iter().skip(1) {
                self.envs[lane] = env.clone();
                self.done_on_info_baselines[lane] = done_on_info_baseline;
            }
        }
        self.synced_groups.clear();
    }

    fn synced_group_actions_are_uniform(&self, actions: &[u8]) -> bool {
        self.synced_groups.iter().all(|group| {
            let first_action = actions[group[0]];
            group
                .iter()
                .skip(1)
                .all(|&lane| actions[lane] == first_action)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn step_synced_groups(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        let mut group_actions = vec![None; config.num_envs];
        for group in &self.synced_groups {
            let first = group[0];
            group_actions[first] = Some(actions[first]);
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(group_actions.par_iter())
                .zip(self.done_on_info_baselines.par_iter())
                .zip(self.last_done_on_info.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(coins.par_iter_mut())
                .zip(level_hi.par_iter_mut())
                .zip(level_lo.par_iter_mut())
                .zip(lives.par_iter_mut())
                .zip(score.par_iter_mut())
                .zip(scrolling.par_iter_mut())
                .zip(time.par_iter_mut())
                .zip(xscroll_hi.par_iter_mut())
                .zip(xscroll_lo.par_iter_mut())
                .for_each(|zipped| {
                    let (zipped, xscroll_lo_out) = zipped;
                    let (zipped, xscroll_hi_out) = zipped;
                    let (zipped, time_out) = zipped;
                    let (zipped, scrolling_out) = zipped;
                    let (zipped, score_out) = zipped;
                    let (zipped, lives_out) = zipped;
                    let (zipped, level_lo_out) = zipped;
                    let (zipped, level_hi_out) = zipped;
                    let (zipped, coins_out) = zipped;
                    let (zipped, x_out) = zipped;
                    let (zipped, truncated_out) = zipped;
                    let (zipped, terminated_out) = zipped;
                    let (zipped, reward_out) = zipped;
                    let (zipped, obs_chunk) = zipped;
                    let (zipped, fired_done_on_info) = zipped;
                    let (zipped, done_on_info_baseline) = zipped;
                    let ((env, scratch), group_action) = zipped;
                    if let Some(action) = *group_action {
                        step_one(
                            config,
                            resize_plan,
                            env,
                            scratch,
                            *done_on_info_baseline,
                            &self.done_on_info_rules,
                            fired_done_on_info,
                            action,
                            obs_chunk,
                            reward_out,
                            terminated_out,
                            truncated_out,
                            x_out,
                            coins_out,
                            level_hi_out,
                            level_lo_out,
                            lives_out,
                            score_out,
                            scrolling_out,
                            time_out,
                            xscroll_hi_out,
                            xscroll_lo_out,
                        );
                    }
                });
        } else {
            for group_idx in 0..self.synced_groups.len() {
                let first = self.synced_groups[group_idx][0];
                let start = first * obs_stride;
                let end = start + obs_stride;
                step_one(
                    config,
                    &self.resize_plan,
                    &mut self.envs[first],
                    &mut self.scratch[first],
                    self.done_on_info_baselines[first],
                    &self.done_on_info_rules,
                    &mut self.last_done_on_info[first],
                    actions[first],
                    &mut obs[start..end],
                    &mut rewards[first],
                    &mut terminated[first],
                    &mut truncated[first],
                    &mut x_pos[first],
                    &mut coins[first],
                    &mut level_hi[first],
                    &mut level_lo[first],
                    &mut lives[first],
                    &mut score[first],
                    &mut scrolling[first],
                    &mut time[first],
                    &mut xscroll_hi[first],
                    &mut xscroll_lo[first],
                );
            }
        }

        for group_idx in 0..self.synced_groups.len() {
            let first = self.synced_groups[group_idx][0];
            for peer_idx in 1..self.synced_groups[group_idx].len() {
                let lane = self.synced_groups[group_idx][peer_idx];
                copy_obs_lane(obs, obs_stride, first, lane);
                rewards[lane] = rewards[first];
                terminated[lane] = terminated[first];
                truncated[lane] = truncated[first];
                copy_info_lane(
                    first, lane, x_pos, coins, level_hi, level_lo, lives, score, scrolling, time,
                    xscroll_hi, xscroll_lo,
                );
            }
        }

        if terminated.iter().any(|done| *done) || truncated.iter().any(|done| *done) {
            self.autoreset_done_lanes(
                obs, terminated, truncated, x_pos, coins, level_hi, level_lo, lives, score,
                scrolling, time, xscroll_hi, xscroll_lo,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_synced_groups_profiled(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        let mut group_actions = vec![None; config.num_envs];
        let group_lane_count = self.synced_groups.iter().map(Vec::len).sum::<usize>();
        let leader_count = self.synced_groups.len();
        for group in &self.synced_groups {
            let first = group[0];
            group_actions[first] = Some(actions[first]);
        }
        if let Some(profiler) = &mut self.profiler {
            profiler.record_group_hit(group_lane_count, leader_count);
        }

        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(group_actions.par_iter())
                .zip(self.done_on_info_baselines.par_iter())
                .zip(self.last_done_on_info.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(coins.par_iter_mut())
                .zip(level_hi.par_iter_mut())
                .zip(level_lo.par_iter_mut())
                .zip(lives.par_iter_mut())
                .zip(score.par_iter_mut())
                .zip(scrolling.par_iter_mut())
                .zip(time.par_iter_mut())
                .zip(xscroll_hi.par_iter_mut())
                .zip(xscroll_lo.par_iter_mut())
                .zip(self.profile_shards.par_iter_mut())
                .for_each(|zipped| {
                    let (zipped, profiler) = zipped;
                    let (zipped, xscroll_lo_out) = zipped;
                    let (zipped, xscroll_hi_out) = zipped;
                    let (zipped, time_out) = zipped;
                    let (zipped, scrolling_out) = zipped;
                    let (zipped, score_out) = zipped;
                    let (zipped, lives_out) = zipped;
                    let (zipped, level_lo_out) = zipped;
                    let (zipped, level_hi_out) = zipped;
                    let (zipped, coins_out) = zipped;
                    let (zipped, x_out) = zipped;
                    let (zipped, truncated_out) = zipped;
                    let (zipped, terminated_out) = zipped;
                    let (zipped, reward_out) = zipped;
                    let (zipped, obs_chunk) = zipped;
                    let (zipped, fired_done_on_info) = zipped;
                    let (zipped, done_on_info_baseline) = zipped;
                    let ((env, scratch), group_action) = zipped;
                    if let Some(action) = *group_action {
                        step_one_profiled(
                            config,
                            resize_plan,
                            env,
                            scratch,
                            *done_on_info_baseline,
                            &self.done_on_info_rules,
                            fired_done_on_info,
                            action,
                            obs_chunk,
                            reward_out,
                            terminated_out,
                            truncated_out,
                            x_out,
                            coins_out,
                            level_hi_out,
                            level_lo_out,
                            lives_out,
                            score_out,
                            scrolling_out,
                            time_out,
                            xscroll_hi_out,
                            xscroll_lo_out,
                            profiler,
                        );
                    }
                });
        } else {
            for group_idx in 0..self.synced_groups.len() {
                let first = self.synced_groups[group_idx][0];
                let start = first * obs_stride;
                let end = start + obs_stride;
                step_one_profiled(
                    config,
                    &self.resize_plan,
                    &mut self.envs[first],
                    &mut self.scratch[first],
                    self.done_on_info_baselines[first],
                    &self.done_on_info_rules,
                    &mut self.last_done_on_info[first],
                    actions[first],
                    &mut obs[start..end],
                    &mut rewards[first],
                    &mut terminated[first],
                    &mut truncated[first],
                    &mut x_pos[first],
                    &mut coins[first],
                    &mut level_hi[first],
                    &mut level_lo[first],
                    &mut lives[first],
                    &mut score[first],
                    &mut scrolling[first],
                    &mut time[first],
                    &mut xscroll_hi[first],
                    &mut xscroll_lo[first],
                    &mut self.profile_shards[first],
                );
            }
        }

        let copy_start = Instant::now();
        let mut peer_count = 0usize;
        for group_idx in 0..self.synced_groups.len() {
            let first = self.synced_groups[group_idx][0];
            for peer_idx in 1..self.synced_groups[group_idx].len() {
                let lane = self.synced_groups[group_idx][peer_idx];
                copy_obs_lane(obs, obs_stride, first, lane);
                rewards[lane] = rewards[first];
                terminated[lane] = terminated[first];
                truncated[lane] = truncated[first];
                copy_info_lane(
                    first, lane, x_pos, coins, level_hi, level_lo, lives, score, scrolling, time,
                    xscroll_hi, xscroll_lo,
                );
                peer_count += 1;
            }
        }
        let copy_elapsed = copy_start.elapsed();
        if let Some(profiler) = &mut self.profiler {
            profiler.record_grouped_copy(peer_count, peer_count * obs_stride, copy_elapsed);
            profiler.record_info_copy(peer_count, copy_elapsed);
        }

        if terminated.iter().any(|done| *done) || truncated.iter().any(|done| *done) {
            self.autoreset_done_lanes(
                obs, terminated, truncated, x_pos, coins, level_hi, level_lo, lives, score,
                scrolling, time, xscroll_hi, xscroll_lo,
            );
        }
    }

    fn refresh_done_baseline(&mut self, env_idx: usize) {
        self.done_on_info_baselines[env_idx] = InfoSnapshot::from_env(&self.envs[env_idx]);
    }

    fn reset_one_env(&mut self, env_idx: usize) {
        if self.initial_states.is_empty() {
            self.envs[env_idx].reset();
            self.active_state_indices[env_idx] = -1;
            self.refresh_done_baseline(env_idx);
            return;
        }

        let state_index = self.initial_state_index_for_env(env_idx);
        self.active_state_indices[env_idx] = state_index as i32;
        self.envs[env_idx]
            .load_fceu_state(&self.initial_states[state_index].data)
            .expect("previously validated initial state failed to reload");
        self.refresh_done_baseline(env_idx);
    }

    #[allow(clippy::too_many_arguments)]
    fn autoreset_done_lanes(
        &mut self,
        obs: &mut [u8],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        for env_idx in 0..config.num_envs {
            if !terminated[env_idx] && !truncated[env_idx] {
                continue;
            }

            self.reset_one_env(env_idx);
            let start = env_idx * obs_stride;
            let end = start + obs_stride;
            write_reset_stack(
                config,
                &self.resize_plan,
                &self.envs[env_idx],
                &mut self.scratch[env_idx],
                &mut obs[start..end],
            );
            write_info_from_env(
                &self.envs[env_idx],
                &mut x_pos[env_idx],
                &mut coins[env_idx],
                &mut level_hi[env_idx],
                &mut level_lo[env_idx],
                &mut lives[env_idx],
                &mut score[env_idx],
                &mut scrolling[env_idx],
                &mut time[env_idx],
                &mut xscroll_hi[env_idx],
                &mut xscroll_lo[env_idx],
            );
        }
        self.synced_lanes = false;
        self.synced_groups.clear();
    }

    pub fn env_ram(&self, env_idx: usize) -> Option<&[u8; 2048]> {
        self.envs.get(env_idx).map(NesEmulator::ram)
    }

    pub fn env_oam(&self, env_idx: usize) -> Option<&[u8; 256]> {
        self.envs.get(env_idx).map(NesEmulator::oam)
    }

    pub fn env_bg_pixel(&self, env_idx: usize, x: usize, y: usize) -> Option<(u8, bool)> {
        self.envs.get(env_idx).map(|env| env.debug_bg_pixel(x, y))
    }
}

#[derive(Clone)]
pub struct InitialState {
    name: String,
    data: Vec<u8>,
    cumulative_weight: f64,
}

impl InitialState {
    pub fn new(name: String, data: Vec<u8>, cumulative_weight: f64) -> Self {
        Self {
            name,
            data,
            cumulative_weight,
        }
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x9e37_79b9_7f4a_7c15)
                ^ 0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self {
            state: state.max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn next_unit_f64(&mut self) -> f64 {
        const DENOM: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DENOM
    }
}

fn copy_first_obs_to_remaining(obs: &mut [u8], obs_stride: usize) {
    let (first, rest) = obs.split_at_mut(obs_stride);
    for chunk in rest.chunks_exact_mut(obs_stride) {
        chunk.copy_from_slice(first);
    }
}

fn copy_obs_lane(obs: &mut [u8], obs_stride: usize, src_lane: usize, dst_lane: usize) {
    if src_lane == dst_lane {
        return;
    }

    let src_start = src_lane * obs_stride;
    let dst_start = dst_lane * obs_stride;
    if src_start < dst_start {
        let (left, right) = obs.split_at_mut(dst_start);
        let src = &left[src_start..src_start + obs_stride];
        right[..obs_stride].copy_from_slice(src);
    } else {
        let (left, right) = obs.split_at_mut(src_start);
        let src = &right[..obs_stride];
        left[dst_start..dst_start + obs_stride].copy_from_slice(src);
    }
}

fn actions_are_uniform(actions: &[u8], first_action: u8) -> bool {
    match actions.len() {
        16 => {
            actions[1] == first_action
                && actions[2] == first_action
                && actions[3] == first_action
                && actions[4] == first_action
                && actions[5] == first_action
                && actions[6] == first_action
                && actions[7] == first_action
                && actions[8] == first_action
                && actions[9] == first_action
                && actions[10] == first_action
                && actions[11] == first_action
                && actions[12] == first_action
                && actions[13] == first_action
                && actions[14] == first_action
                && actions[15] == first_action
        }
        _ => actions.iter().all(|&action| action == first_action),
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_info_from_env(
    env: &NesEmulator,
    x_pos: &mut [u16],
    coins: &mut [u8],
    level_hi: &mut [i16],
    level_lo: &mut [i16],
    lives: &mut [i16],
    score: &mut [u32],
    scrolling: &mut [i16],
    time: &mut [u16],
    xscroll_hi: &mut [u8],
    xscroll_lo: &mut [u8],
) {
    x_pos.fill(env.x_pos());
    coins.fill(env.coins());
    level_hi.fill(env.level_hi());
    level_lo.fill(env.level_lo());
    lives.fill(env.lives());
    score.fill(env.score());
    scrolling.fill(env.scrolling());
    time.fill(env.time());
    xscroll_hi.fill(env.xscroll_hi());
    xscroll_lo.fill(env.xscroll_lo());
}

#[allow(clippy::too_many_arguments)]
fn fill_info_from_first(
    x_pos: &mut [u16],
    coins: &mut [u8],
    level_hi: &mut [i16],
    level_lo: &mut [i16],
    lives: &mut [i16],
    score: &mut [u32],
    scrolling: &mut [i16],
    time: &mut [u16],
    xscroll_hi: &mut [u8],
    xscroll_lo: &mut [u8],
) {
    x_pos.fill(x_pos[0]);
    coins.fill(coins[0]);
    level_hi.fill(level_hi[0]);
    level_lo.fill(level_lo[0]);
    lives.fill(lives[0]);
    score.fill(score[0]);
    scrolling.fill(scrolling[0]);
    time.fill(time[0]);
    xscroll_hi.fill(xscroll_hi[0]);
    xscroll_lo.fill(xscroll_lo[0]);
}

#[allow(clippy::too_many_arguments)]
fn copy_info_lane(
    src_lane: usize,
    dst_lane: usize,
    x_pos: &mut [u16],
    coins: &mut [u8],
    level_hi: &mut [i16],
    level_lo: &mut [i16],
    lives: &mut [i16],
    score: &mut [u32],
    scrolling: &mut [i16],
    time: &mut [u16],
    xscroll_hi: &mut [u8],
    xscroll_lo: &mut [u8],
) {
    x_pos[dst_lane] = x_pos[src_lane];
    coins[dst_lane] = coins[src_lane];
    level_hi[dst_lane] = level_hi[src_lane];
    level_lo[dst_lane] = level_lo[src_lane];
    lives[dst_lane] = lives[src_lane];
    score[dst_lane] = score[src_lane];
    scrolling[dst_lane] = scrolling[src_lane];
    time[dst_lane] = time[src_lane];
    xscroll_hi[dst_lane] = xscroll_hi[src_lane];
    xscroll_lo[dst_lane] = xscroll_lo[src_lane];
}

#[allow(clippy::too_many_arguments)]
fn write_info_from_env(
    env: &NesEmulator,
    x_out: &mut u16,
    coins_out: &mut u8,
    level_hi_out: &mut i16,
    level_lo_out: &mut i16,
    lives_out: &mut i16,
    score_out: &mut u32,
    scrolling_out: &mut i16,
    time_out: &mut u16,
    xscroll_hi_out: &mut u8,
    xscroll_lo_out: &mut u8,
) {
    *x_out = env.x_pos();
    *coins_out = env.coins();
    *level_hi_out = env.level_hi();
    *level_lo_out = env.level_lo();
    *lives_out = env.lives();
    *score_out = env.score();
    *scrolling_out = env.scrolling();
    *time_out = env.time();
    *xscroll_hi_out = env.xscroll_hi();
    *xscroll_lo_out = env.xscroll_lo();
}

#[allow(clippy::too_many_arguments)]
fn step_one(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &mut NesEmulator,
    scratch: &mut [u8],
    done_on_info_baseline: InfoSnapshot,
    done_on_info_rules: &[DoneOnInfoRule],
    fired_done_on_info: &mut Vec<FiredDoneOnInfoRule>,
    action_id: u8,
    obs_chunk: &mut [u8],
    reward_out: &mut f32,
    terminated_out: &mut bool,
    truncated_out: &mut bool,
    x_out: &mut u16,
    coins_out: &mut u8,
    level_hi_out: &mut i16,
    level_lo_out: &mut i16,
    lives_out: &mut i16,
    score_out: &mut u32,
    scrolling_out: &mut i16,
    time_out: &mut u16,
    xscroll_hi_out: &mut u8,
    xscroll_lo_out: &mut u8,
) {
    let action = MarioAction::from_u8(action_id);
    let mut reward = 0.0;
    let mut done = false;
    for _ in 0..config.frame_skip {
        reward += env.step_frame(action);
        let done_on_info = check_done_on_info(
            env,
            done_on_info_baseline,
            done_on_info_rules,
            fired_done_on_info,
        );
        done = env.is_done() || done_on_info;
        if done {
            break;
        }
    }
    shift_stack_left(config, obs_chunk);
    write_current_frame_to_last_stack_slot(config, resize_plan, env, scratch, obs_chunk);

    *reward_out = reward;
    *terminated_out = done;
    *truncated_out = false;
    write_info_from_env(
        env,
        x_out,
        coins_out,
        level_hi_out,
        level_lo_out,
        lives_out,
        score_out,
        scrolling_out,
        time_out,
        xscroll_hi_out,
        xscroll_lo_out,
    );
}

#[allow(clippy::too_many_arguments)]
fn step_one_profiled(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &mut NesEmulator,
    scratch: &mut [u8],
    done_on_info_baseline: InfoSnapshot,
    done_on_info_rules: &[DoneOnInfoRule],
    fired_done_on_info: &mut Vec<FiredDoneOnInfoRule>,
    action_id: u8,
    obs_chunk: &mut [u8],
    reward_out: &mut f32,
    terminated_out: &mut bool,
    truncated_out: &mut bool,
    x_out: &mut u16,
    coins_out: &mut u8,
    level_hi_out: &mut i16,
    level_lo_out: &mut i16,
    lives_out: &mut i16,
    score_out: &mut u32,
    scrolling_out: &mut i16,
    time_out: &mut u16,
    xscroll_hi_out: &mut u8,
    xscroll_lo_out: &mut u8,
    profiler: &mut Profiler,
) {
    let action = MarioAction::from_u8(action_id);
    let mut reward = 0.0;
    let mut done = false;
    for _ in 0..config.frame_skip {
        reward += env.step_frame_profiled(action, profiler);
        let done_on_info = check_done_on_info(
            env,
            done_on_info_baseline,
            done_on_info_rules,
            fired_done_on_info,
        );
        done = env.is_done() || done_on_info;
        if done {
            break;
        }
    }
    let shift_start = Instant::now();
    shift_stack_left(config, obs_chunk);
    profiler.record_stack_shift(shift_start.elapsed());
    write_current_frame_to_last_stack_slot_profiled(
        config,
        resize_plan,
        env,
        scratch,
        obs_chunk,
        profiler,
    );

    *reward_out = reward;
    *terminated_out = done;
    *truncated_out = false;
    write_info_from_env(
        env,
        x_out,
        coins_out,
        level_hi_out,
        level_lo_out,
        lives_out,
        score_out,
        scrolling_out,
        time_out,
        xscroll_hi_out,
        xscroll_lo_out,
    );
}

fn check_done_on_info(
    env: &NesEmulator,
    baseline: InfoSnapshot,
    rules: &[DoneOnInfoRule],
    fired_rules: &mut Vec<FiredDoneOnInfoRule>,
) -> bool {
    if rules.is_empty() {
        return false;
    }
    let current = InfoSnapshot::from_env(env);
    let mut fired_any = false;
    for rule in rules {
        let mut fired = false;
        let mut previous_values = Vec::with_capacity(rule.keys.len());
        let mut current_values = Vec::with_capacity(rule.keys.len());
        for key in &rule.keys {
            let previous = baseline.value(*key);
            let next = current.value(*key);
            previous_values.push(previous);
            current_values.push(next);
            if done_on_info_value_fired(rule.op, previous, next) {
                fired = true;
            }
        }
        if !fired {
            continue;
        }
        fired_any = true;
        fired_rules.push(FiredDoneOnInfoRule {
            name: rule.name.clone(),
            keys: rule.keys.clone(),
            op: rule.op,
            previous_values,
            current_values,
        });
    }
    fired_any
}

fn done_on_info_value_fired(op: DoneOnInfoOp, baseline: i64, current: i64) -> bool {
    match op {
        DoneOnInfoOp::Change => current != baseline,
        DoneOnInfoOp::Increase => current > baseline,
        DoneOnInfoOp::Decrease => current < baseline,
    }
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

fn write_current_frame_to_last_stack_slot_profiled(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    obs_chunk: &mut [u8],
    profiler: &mut Profiler,
) {
    let frame_len = frame_len(config);
    let dst_start = (config.frame_stack - 1) * frame_len;
    let dst_end = dst_start + frame_len;
    write_current_frame_profiled(
        config,
        resize_plan,
        env,
        scratch,
        &mut obs_chunk[dst_start..dst_end],
        profiler,
    );
}

fn write_current_frame(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    dst: &mut [u8],
) {
    if config.uses_default_gray_area_resize() {
        env.write_gray_frame_cropped_area_84x84(dst, scratch);
        return;
    }

    if config.needs_resize() {
        let native_len = native_frame_len(config);
        let native = &mut scratch[..native_len];
        write_native_frame(config, env, native);
        resize_frame_area(config, resize_plan, native, dst);
    } else {
        write_native_frame(config, env, dst);
    }
}

fn write_current_frame_profiled(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    dst: &mut [u8],
    profiler: &mut Profiler,
) {
    if config.uses_default_gray_area_resize() {
        let start = Instant::now();
        env.write_gray_frame_cropped_area_84x84(dst, scratch);
        profiler.record_render(start.elapsed());
        return;
    }

    if config.needs_resize() {
        let native_len = native_frame_len(config);
        let native = &mut scratch[..native_len];
        let render_start = Instant::now();
        write_native_frame(config, env, native);
        profiler.record_render(render_start.elapsed());
        let resize_start = Instant::now();
        resize_frame_area(config, resize_plan, native, dst);
        profiler.record_resize(resize_start.elapsed());
    } else {
        let start = Instant::now();
        write_native_frame(config, env, dst);
        profiler.record_render(start.elapsed());
    }
}

fn write_native_frame(config: VecEnvConfig, env: &NesEmulator, dst: &mut [u8]) {
    let height = config.source_height();
    if config.grayscale {
        env.write_gray_visible_frame_cropped(dst, config.crop_top, height);
    } else {
        env.write_rgb_visible_frame_cropped(dst, config.crop_top, height);
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
        config.source_width() * config.source_height()
    } else {
        config.source_width() * config.source_height() * RGB_CHANNELS
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
    debug_assert!(src.len() >= src_offset + plan.src_width * plan.src_height);
    debug_assert!(dst.len() >= dst_offset + plan.dst_width * plan.dst_height);

    for (dst_i, bin) in plan.bins.iter().enumerate() {
        let mut sum = 0u32;
        for sy in bin.y0..bin.y1 {
            let src_row = src_offset + sy * plan.src_width;
            for sx in bin.x0..bin.x1 {
                // SAFETY: AreaResizePlan bins are built from dimensions validated above.
                sum += unsafe { *src.get_unchecked(src_row + sx) } as u32;
            }
        }
        // SAFETY: dst_i iterates over exactly dst_width * dst_height planned pixels.
        unsafe {
            *dst.get_unchecked_mut(dst_offset + dst_i) = (sum / bin.count) as u8;
        }
    }
}

struct AreaResizePlan {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    bins: Vec<AreaResizeBin>,
}

impl AreaResizePlan {
    fn new(src_width: usize, src_height: usize, dst_width: usize, dst_height: usize) -> Self {
        let mut bins = Vec::with_capacity(dst_width * dst_height);
        for dy in 0..dst_height {
            let y0 = (dy * src_height) / dst_height;
            let y1 = (((dy + 1) * src_height) / dst_height)
                .max(y0 + 1)
                .min(src_height);
            for dx in 0..dst_width {
                let x0 = (dx * src_width) / dst_width;
                let x1 = (((dx + 1) * src_width) / dst_width)
                    .max(x0 + 1)
                    .min(src_width);
                bins.push(AreaResizeBin {
                    x0,
                    x1,
                    y0,
                    y1,
                    count: ((x1 - x0) * (y1 - y0)) as u32,
                });
            }
        }
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            bins,
        }
    }
}

struct AreaResizeBin {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    count: u32,
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
        let plan = AreaResizePlan::new(src_width, src_height, dst_width, dst_height);
        for (dst_i, bin) in plan.bins.iter().enumerate() {
            let mut sum = 0u32;
            for sy in bin.y0..bin.y1 {
                let src_row = src_offset + sy * src_width;
                for sx in bin.x0..bin.x1 {
                    sum += src[src_row + sx] as u32;
                }
            }
            dst[dst_offset + dst_i] = (sum / bin.count) as u8;
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
        let plan = AreaResizePlan::new(config.source_width(), config.source_height(), 84, 84);
        let src_len = config.source_width() * config.source_height();
        let src = (0..src_len)
            .map(|idx| ((idx * 37 + idx / 251 + 19) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut optimized = vec![0; 84 * 84];
        let mut reference = vec![0; 84 * 84];

        resize_frame_area(config, &plan, &src, &mut optimized);
        reference_resize_plane_area(
            &src,
            &mut reference,
            config.source_width(),
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
        let src_width = VISIBLE_FRAME_WIDTH;
        let src_height = VISIBLE_FRAME_HEIGHT - 32;
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
