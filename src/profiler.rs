use std::fmt::Write;
use std::time::Duration;

const CPU_OPCODE_COUNT: usize = 256;
const CPU_PC_COUNT: usize = 65_536;
const PC_RANGE_SIZE: usize = 16;
const PC_PAGE_SIZE: usize = 256;

#[derive(Clone, Default)]
pub struct Profiler {
    totals: ProfileTotals,
    timings: ProfileTimings,
    cpu: CpuProfile,
    vec_env: VecEnvProfile,
}

#[derive(Clone, Default)]
struct ProfileTotals {
    env_steps: u64,
    batch_steps: u64,
    frame_steps: u64,
    cpu_steps: u64,
    ppu_tick_calls: u64,
    ppu_tick_cycles: u64,
    ppu_completed_frames: u64,
    render_calls: u64,
    resize_calls: u64,
    stack_shifts: u64,
    grouped_peer_copies: u64,
    grouped_materializations: u64,
}

#[derive(Clone, Default)]
struct ProfileTimings {
    frame_stepping_ns: u64,
    rendering_ns: u64,
    resize_ns: u64,
    stack_shift_ns: u64,
    grouped_lane_copy_ns: u64,
    scalar_info_copy_ns: u64,
    materialization_ns: u64,
}

#[derive(Clone)]
struct CpuProfile {
    opcode_counts: [u64; CPU_OPCODE_COUNT],
    pc_counts: Vec<u64>,
}

impl Default for CpuProfile {
    fn default() -> Self {
        Self {
            opcode_counts: [0; CPU_OPCODE_COUNT],
            pc_counts: vec![0; CPU_PC_COUNT],
        }
    }
}

#[derive(Clone, Default)]
struct VecEnvProfile {
    group_leaders: u64,
    peer_copies: u64,
    materialized_lanes: u64,
    materialization_events: u64,
    uniform_action_group_hits: u64,
    uniform_action_group_misses: u64,
    copied_observation_bytes: u64,
    copied_info_lanes: u64,
    first_lane_broadcasts: u64,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn add(&mut self, other: &Self) {
        self.totals.add(&other.totals);
        self.timings.add(&other.timings);
        self.vec_env.add(&other.vec_env);
        for (dst, src) in self
            .cpu
            .opcode_counts
            .iter_mut()
            .zip(other.cpu.opcode_counts.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .cpu
            .pc_counts
            .iter_mut()
            .zip(other.cpu.pc_counts.iter())
        {
            *dst += *src;
        }
    }

    pub fn record_batch_step(&mut self, num_envs: usize) {
        self.totals.batch_steps += 1;
        self.totals.env_steps += num_envs as u64;
    }

    pub fn record_cpu_step(&mut self, pc: u16, opcode: u8) {
        self.totals.cpu_steps += 1;
        self.cpu.opcode_counts[opcode as usize] += 1;
        self.cpu.pc_counts[pc as usize] += 1;
    }

    pub fn record_frame_step(&mut self, elapsed: Duration) {
        self.totals.frame_steps += 1;
        self.timings.frame_stepping_ns += duration_ns(elapsed);
    }

    pub fn record_ppu_tick(&mut self, cycles: usize, completed_frame: bool) {
        self.totals.ppu_tick_calls += 1;
        self.totals.ppu_tick_cycles += cycles as u64;
        if completed_frame {
            self.totals.ppu_completed_frames += 1;
        }
    }

    pub fn record_render(&mut self, elapsed: Duration) {
        self.totals.render_calls += 1;
        self.timings.rendering_ns += duration_ns(elapsed);
    }

    pub fn record_resize(&mut self, elapsed: Duration) {
        self.totals.resize_calls += 1;
        self.timings.resize_ns += duration_ns(elapsed);
    }

    pub fn record_stack_shift(&mut self, elapsed: Duration) {
        self.totals.stack_shifts += 1;
        self.timings.stack_shift_ns += duration_ns(elapsed);
    }

    pub fn record_group_hit(&mut self, group_count: usize, leader_count: usize) {
        self.vec_env.uniform_action_group_hits += 1;
        self.vec_env.group_leaders += leader_count as u64;
        if group_count > leader_count {
            self.vec_env.peer_copies += (group_count - leader_count) as u64;
        }
    }

    pub fn record_group_miss(&mut self) {
        self.vec_env.uniform_action_group_misses += 1;
    }

    pub fn record_first_lane_broadcast(&mut self, peer_count: usize, obs_bytes: usize) {
        self.vec_env.first_lane_broadcasts += 1;
        self.vec_env.peer_copies += peer_count as u64;
        self.vec_env.copied_observation_bytes += obs_bytes as u64;
    }

    pub fn record_grouped_copy(&mut self, peer_count: usize, obs_bytes: usize, elapsed: Duration) {
        self.totals.grouped_peer_copies += peer_count as u64;
        self.vec_env.copied_observation_bytes += obs_bytes as u64;
        self.timings.grouped_lane_copy_ns += duration_ns(elapsed);
    }

    pub fn record_info_copy(&mut self, lanes: usize, elapsed: Duration) {
        self.vec_env.copied_info_lanes += lanes as u64;
        self.timings.scalar_info_copy_ns += duration_ns(elapsed);
    }

    pub fn record_materialization(&mut self, lanes: usize, elapsed: Duration) {
        self.totals.grouped_materializations += 1;
        self.vec_env.materialization_events += 1;
        self.vec_env.materialized_lanes += lanes as u64;
        self.timings.materialization_ns += duration_ns(elapsed);
    }

    pub fn to_json(&self, top_n: usize) -> String {
        let mut out = String::new();
        let ppu_cycles_per_tick = div_u64(self.totals.ppu_tick_cycles, self.totals.ppu_tick_calls);
        let cpu_steps_per_frame = div_u64(self.totals.cpu_steps, self.totals.frame_steps);
        let cpu_steps_per_env_step = div_u64(self.totals.cpu_steps, self.totals.env_steps);
        let render_ns_per_env_step = div_u64(self.timings.rendering_ns, self.totals.env_steps);
        let resize_ns_per_env_step = div_u64(self.timings.resize_ns, self.totals.env_steps);
        let copy_ns_per_env_step =
            div_u64(self.timings.grouped_lane_copy_ns, self.totals.env_steps);
        let group_attempts =
            self.vec_env.uniform_action_group_hits + self.vec_env.uniform_action_group_misses;
        let group_hit_rate = div_u64(self.vec_env.uniform_action_group_hits, group_attempts);

        out.push_str("{\n");
        out.push_str("  \"enabled\": true,\n");
        out.push_str("  \"measurement_window\": \"post_warmup_repeats_only\",\n");
        out.push_str("  \"totals\": {");
        write!(
            out,
            "\"env_steps\":{},\"batch_steps\":{},\"frame_steps\":{},\"cpu_steps\":{},\"ppu_tick_calls\":{},\"ppu_tick_cycles\":{},\"ppu_completed_frames\":{},\"render_calls\":{},\"resize_calls\":{},\"stack_shifts\":{},\"grouped_peer_copies\":{},\"grouped_materializations\":{}",
            self.totals.env_steps,
            self.totals.batch_steps,
            self.totals.frame_steps,
            self.totals.cpu_steps,
            self.totals.ppu_tick_calls,
            self.totals.ppu_tick_cycles,
            self.totals.ppu_completed_frames,
            self.totals.render_calls,
            self.totals.resize_calls,
            self.totals.stack_shifts,
            self.totals.grouped_peer_copies,
            self.totals.grouped_materializations,
        )
        .unwrap();
        out.push_str("},\n");
        out.push_str("  \"derived\": {");
        write!(
            out,
            "\"cpu_steps_per_frame\":{},\"cpu_steps_per_env_step\":{},\"ppu_cycles_per_tick_call\":{},\"render_ns_per_env_step\":{},\"resize_ns_per_env_step\":{},\"grouped_copy_ns_per_env_step\":{},\"group_hit_rate\":{}",
            json_f64(cpu_steps_per_frame),
            json_f64(cpu_steps_per_env_step),
            json_f64(ppu_cycles_per_tick),
            json_f64(render_ns_per_env_step),
            json_f64(resize_ns_per_env_step),
            json_f64(copy_ns_per_env_step),
            json_f64(group_hit_rate),
        )
        .unwrap();
        out.push_str("},\n");
        out.push_str("  \"cpu\": {\n");
        out.push_str("    \"top_opcodes\": ");
        write_count_entries(
            &mut out,
            &top_opcode_counts(&self.cpu.opcode_counts, top_n),
            "opcode",
        );
        out.push_str(",\n    \"top_pcs\": ");
        write_count_entries(&mut out, &top_pc_counts(&self.cpu.pc_counts, top_n), "pc");
        out.push_str(",\n    \"top_pc_ranges_16\": ");
        write_count_entries(
            &mut out,
            &top_range_counts(&self.cpu.pc_counts, PC_RANGE_SIZE, top_n),
            "range",
        );
        out.push_str(",\n    \"top_pc_pages\": ");
        write_count_entries(
            &mut out,
            &top_range_counts(&self.cpu.pc_counts, PC_PAGE_SIZE, top_n),
            "range",
        );
        out.push_str("\n  },\n");
        out.push_str("  \"vec_env\": {");
        write!(
            out,
            "\"group_leaders\":{},\"peer_copies\":{},\"materialized_lanes\":{},\"materialization_events\":{},\"uniform_action_group_hits\":{},\"uniform_action_group_misses\":{},\"copied_observation_bytes\":{},\"copied_info_lanes\":{},\"first_lane_broadcasts\":{}",
            self.vec_env.group_leaders,
            self.vec_env.peer_copies,
            self.vec_env.materialized_lanes,
            self.vec_env.materialization_events,
            self.vec_env.uniform_action_group_hits,
            self.vec_env.uniform_action_group_misses,
            self.vec_env.copied_observation_bytes,
            self.vec_env.copied_info_lanes,
            self.vec_env.first_lane_broadcasts,
        )
        .unwrap();
        out.push_str("},\n");
        out.push_str("  \"timings_ns\": {");
        write!(
            out,
            "\"frame_stepping\":{},\"rendering\":{},\"resize\":{},\"stack_shift\":{},\"grouped_lane_copy\":{},\"scalar_info_copy\":{},\"materialization\":{}",
            self.timings.frame_stepping_ns,
            self.timings.rendering_ns,
            self.timings.resize_ns,
            self.timings.stack_shift_ns,
            self.timings.grouped_lane_copy_ns,
            self.timings.scalar_info_copy_ns,
            self.timings.materialization_ns,
        )
        .unwrap();
        out.push_str("}\n");
        out.push('}');
        out
    }
}

impl ProfileTotals {
    fn add(&mut self, other: &Self) {
        self.env_steps += other.env_steps;
        self.batch_steps += other.batch_steps;
        self.frame_steps += other.frame_steps;
        self.cpu_steps += other.cpu_steps;
        self.ppu_tick_calls += other.ppu_tick_calls;
        self.ppu_tick_cycles += other.ppu_tick_cycles;
        self.ppu_completed_frames += other.ppu_completed_frames;
        self.render_calls += other.render_calls;
        self.resize_calls += other.resize_calls;
        self.stack_shifts += other.stack_shifts;
        self.grouped_peer_copies += other.grouped_peer_copies;
        self.grouped_materializations += other.grouped_materializations;
    }
}

impl ProfileTimings {
    fn add(&mut self, other: &Self) {
        self.frame_stepping_ns += other.frame_stepping_ns;
        self.rendering_ns += other.rendering_ns;
        self.resize_ns += other.resize_ns;
        self.stack_shift_ns += other.stack_shift_ns;
        self.grouped_lane_copy_ns += other.grouped_lane_copy_ns;
        self.scalar_info_copy_ns += other.scalar_info_copy_ns;
        self.materialization_ns += other.materialization_ns;
    }
}

impl VecEnvProfile {
    fn add(&mut self, other: &Self) {
        self.group_leaders += other.group_leaders;
        self.peer_copies += other.peer_copies;
        self.materialized_lanes += other.materialized_lanes;
        self.materialization_events += other.materialization_events;
        self.uniform_action_group_hits += other.uniform_action_group_hits;
        self.uniform_action_group_misses += other.uniform_action_group_misses;
        self.copied_observation_bytes += other.copied_observation_bytes;
        self.copied_info_lanes += other.copied_info_lanes;
        self.first_lane_broadcasts += other.first_lane_broadcasts;
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn div_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "0.000000".to_string()
    }
}

fn top_opcode_counts(counts: &[u64; CPU_OPCODE_COUNT], top_n: usize) -> Vec<(String, u64)> {
    let mut entries = counts
        .iter()
        .enumerate()
        .filter_map(|(opcode, &count)| (count > 0).then(|| (format!("0x{opcode:02X}"), count)))
        .collect::<Vec<_>>();
    sort_and_truncate(&mut entries, top_n);
    entries
}

fn top_pc_counts(counts: &[u64], top_n: usize) -> Vec<(String, u64)> {
    let mut entries = counts
        .iter()
        .enumerate()
        .filter_map(|(pc, &count)| (count > 0).then(|| (format!("0x{pc:04X}"), count)))
        .collect::<Vec<_>>();
    sort_and_truncate(&mut entries, top_n);
    entries
}

fn top_range_counts(counts: &[u64], range_size: usize, top_n: usize) -> Vec<(String, u64)> {
    let mut entries = counts
        .chunks(range_size)
        .enumerate()
        .filter_map(|(range_idx, chunk)| {
            let count = chunk.iter().sum::<u64>();
            if count == 0 {
                return None;
            }
            let start = range_idx * range_size;
            let end = start + range_size - 1;
            Some((format!("0x{start:04X}-0x{end:04X}"), count))
        })
        .collect::<Vec<_>>();
    sort_and_truncate(&mut entries, top_n);
    entries
}

fn sort_and_truncate(entries: &mut Vec<(String, u64)>, top_n: usize) {
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    entries.truncate(top_n);
}

fn write_count_entries(out: &mut String, entries: &[(String, u64)], key: &str) {
    out.push('[');
    for (idx, (name, count)) in entries.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write!(out, "{{\"{key}\":\"{name}\",\"count\":{count}}}").unwrap();
    }
    out.push(']');
}
