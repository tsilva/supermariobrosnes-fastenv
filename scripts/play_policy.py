from __future__ import annotations

import argparse
import ctypes
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from play import (
    DEFAULT_ROM,
    NES_HEIGHT,
    NES_WIDTH,
    SDL_INIT_VIDEO,
    SDL_PIXELFORMAT_RGB24,
    SDL_QUIT,
    SDL_RENDERER_ACCELERATED,
    SDL_TEXTUREACCESS_STREAMING,
    SDL_WINDOWPOS_CENTERED,
    SDL_WINDOW_SHOWN,
    SdlUnavailableError,
    configure_sdl,
    display_frame_from_obs,
    load_sdl2,
)
from supermariobrosnes_turbo import ACTION_SETS, SuperMarioBrosVecEnv


DEFAULT_HF_FILENAME = "ppo_supermariobros-nes-v0_4500000_steps.zip"
DEFAULT_GAME = "SuperMarioBros-Nes-v0"
HF_URL_RE = re.compile(r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)(?:/(?P<rest>.*))?$")
ACTION_BUTTONS = {
    "noop": (),
    "right": ("RIGHT",),
    "right_b": ("RIGHT", "B"),
    "right_a": ("RIGHT", "A"),
    "right_a_b": ("RIGHT", "A", "B"),
    "a": ("A",),
    "left": ("LEFT",),
    "start": ("START",),
}


class ModelResolutionError(RuntimeError):
    pass


def parse_hf_source(source: str) -> tuple[str, str | None, str | None] | None:
    match = HF_URL_RE.match(source)
    if match:
        repo_id = match.group("repo")
        rest = match.group("rest") or ""
        parts = rest.split("/") if rest else []
        if len(parts) >= 3 and parts[0] in {"blob", "resolve"}:
            revision = parts[1]
            filename = "/".join(parts[2:])
            return repo_id, filename, revision
        return repo_id, None, None
    if "/" in source and not source.endswith(".zip") and not Path(source).expanduser().exists():
        return source, None, None
    return None


def resolve_model_path(source: str, filename: str | None, cache_dir: Path) -> Path:
    local_path = Path(source).expanduser()
    if local_path.exists():
        return local_path

    hf_source = parse_hf_source(source)
    if hf_source is None:
        raise ModelResolutionError(f"model source does not exist and is not a Hugging Face repo/url: {source}")

    repo_id, source_filename, revision = hf_source
    target_filename = filename or source_filename
    if target_filename is None:
        target_filename = find_hf_zip_filename(repo_id, revision=revision) or DEFAULT_HF_FILENAME

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return download_direct_hf_file(
            repo_id,
            filename=target_filename,
            revision=revision or "main",
            cache_dir=cache_dir,
        )

    path = hf_hub_download(
        repo_id=repo_id,
        filename=target_filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    return Path(path)


def find_hf_zip_filename(repo_id: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return None
    files = list_repo_files(repo_id, revision=revision)
    zip_files = sorted(path for path in files if path.endswith(".zip"))
    if len(zip_files) == 1:
        return zip_files[0]
    if DEFAULT_HF_FILENAME in zip_files:
        return DEFAULT_HF_FILENAME
    return None


def download_direct_hf_file(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path:
    safe_name = urllib.parse.quote(f"{repo_id}/{revision}/{filename}", safe="")
    target = cache_dir.expanduser() / "direct" / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    quoted_filename = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quoted_filename}"
    urllib.request.urlretrieve(url, target)
    return target


def stable_action_masks(action_names: tuple[str, ...], rom_path: Path) -> np.ndarray:
    import stable_retro

    system = stable_retro.get_romfile_system(str(rom_path))
    core = stable_retro.get_system_info(system)
    buttons = tuple(None if name is None else str(name).upper() for name in core["buttons"])
    button_to_index = {name: index for index, name in enumerate(buttons) if name is not None}
    masks = np.zeros((len(action_names), len(buttons)), dtype=np.uint8)
    for action_index, action_name in enumerate(action_names):
        for button in ACTION_BUTTONS[action_name]:
            try:
                masks[action_index, button_to_index[button]] = 1
            except KeyError as exc:
                raise ValueError(f"Retro core buttons {buttons!r} do not include {button!r}") from exc
    return masks


def json_default(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


class SdlPolicyPlayer:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise SystemExit(
                "stable_baselines3 is required to play SB3 policies. "
                "Install it in this environment, then rerun the same command.",
            ) from exc

        self.model_path = resolve_model_path(args.model, args.filename, args.cache_dir)
        self.model = PPO.load(self.model_path, device=args.device)
        if getattr(self.model.action_space, "n", None) != len(ACTION_SETS[args.action_set]):
            raise ValueError(
                f"model action space {self.model.action_space} does not match "
                f"action_set={args.action_set!r} with {len(ACTION_SETS[args.action_set])} actions",
            )

        self.args = args
        self.action_names = ACTION_SETS[args.action_set]
        self.rom_path = args.rom_path.expanduser()
        self.env = self.make_env()
        self.obs = self.env.reset()
        self.display_env = self.make_display_env() if args.view == "raw" else None
        self.display_obs = self.display_env.reset() if self.display_env is not None else self.obs
        self.display_info: dict[str, object] = {}
        initial_frame = self.current_display_frame()
        self.display_height, self.display_width = initial_frame.shape[:2]
        self.scale = args.scale
        self.frame_delay_s = 1.0 / max(1, args.fps)
        self.episode = 1
        self.step = 0
        self.reward = 0.0
        self.max_x = 0
        self.info: dict[str, object] = {}
        self.action = 0
        self.frames_rendered = 0
        self.running = True
        self.next_tick = time.perf_counter() + self.frame_delay_s
        self.fps_window_start = time.perf_counter()
        self.fps_window_frames = 0
        self.display_fps = 0.0
        self.next_status_update = 0.0

        self.sdl = load_sdl2()
        configure_sdl(self.sdl)
        if self.sdl.SDL_Init(SDL_INIT_VIDEO) != 0:
            raise SdlUnavailableError(self.sdl_error())
        self.sdl.SDL_SetHint(b"SDL_RENDER_SCALE_QUALITY", b"nearest")
        self.window = self.sdl.SDL_CreateWindow(
            b"SuperMarioBros-Nes-turbo policy player",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            self.display_width * self.scale,
            self.display_height * self.scale,
            SDL_WINDOW_SHOWN,
        )
        if not self.window:
            error = self.sdl_error()
            self.sdl.SDL_Quit()
            raise SdlUnavailableError(error)
        self.renderer = self.sdl.SDL_CreateRenderer(self.window, -1, SDL_RENDERER_ACCELERATED)
        if not self.renderer:
            error = self.sdl_error()
            self.sdl.SDL_DestroyWindow(self.window)
            self.sdl.SDL_Quit()
            raise SdlUnavailableError(error)
        self.texture = self.sdl.SDL_CreateTexture(
            self.renderer,
            SDL_PIXELFORMAT_RGB24,
            SDL_TEXTUREACCESS_STREAMING,
            self.display_width,
            self.display_height,
        )
        if not self.texture:
            error = self.sdl_error()
            self.sdl.SDL_DestroyRenderer(self.renderer)
            self.sdl.SDL_DestroyWindow(self.window)
            self.sdl.SDL_Quit()
            raise SdlUnavailableError(error)

    def make_env(self):
        if self.args.backend == "native":
            return SuperMarioBrosVecEnv(
                rom_path=self.rom_path,
                num_envs=1,
                frame_skip=self.args.frame_skip,
                grayscale=True,
                frame_stack=self.args.frame_stack,
                frame_maxpool=self.args.max_pool_frames,
                terminate_on_flag=self.args.terminate_on_flag,
                crop_top=self.args.crop_top,
                crop_bottom=self.args.crop_bottom,
                resize_width=self.args.resize_width,
                resize_height=self.args.resize_height,
                state=self.args.state,
                state_dir=self.args.state_dir,
                action_set=self.args.action_set,
                seed=self.args.seed,
                terminate_on_life_loss=self.args.terminate_on_life_loss,
                terminate_on_level_change=self.args.terminate_on_level_change,
            )

        import stable_retro

        source_height = 224 - self.args.crop_top - self.args.crop_bottom
        obs_crop = None
        if self.args.crop_top != 0 or self.args.crop_bottom != 0:
            obs_crop = (self.args.crop_top, self.args.crop_bottom, 0, 0)
        obs_resize = None
        if self.args.resize_width != 240 or self.args.resize_height != source_height:
            obs_resize = (self.args.resize_height, self.args.resize_width)
        done_on = {}
        if self.args.terminate_on_life_loss:
            done_on["life_loss"] = ("lives", "decrease")
        if self.args.terminate_on_level_change:
            done_on["level_change"] = (("levelHi", "levelLo"), "change")
        env = stable_retro.RetroVecEnv(
            self.args.game,
            state=self.args.state,
            num_envs=1,
            num_threads=1,
            rom_path=str(self.rom_path),
            render_mode="rgb_array",
            use_restricted_actions=stable_retro.Actions.ALL,
            obs_crop=obs_crop,
            obs_resize=obs_resize,
            obs_grayscale=True,
            obs_resize_algorithm="area",
            frame_skip=self.args.frame_skip,
            frame_stack=self.args.frame_stack,
            frame_maxpool=self.args.max_pool_frames,
            reset_noops=0,
            action_sticky_prob=0.0,
            reward_clip=False,
            info_filter="all",
            obs_layout="chw",
            obs_copy="safe_view",
            done_on=done_on or None,
        )
        if hasattr(env, "seed"):
            env.seed(self.args.seed)
        self.retro_action_masks = stable_action_masks(self.action_names, self.rom_path)
        return env

    def make_display_env(self):
        if self.args.backend == "native":
            return SuperMarioBrosVecEnv(
                rom_path=self.rom_path,
                num_envs=1,
                frame_skip=self.args.frame_skip,
                grayscale=False,
                frame_stack=1,
                frame_maxpool=False,
                terminate_on_flag=self.args.terminate_on_flag,
                crop_top=0,
                crop_bottom=0,
                resize_width=NES_WIDTH,
                resize_height=NES_HEIGHT,
                state=self.args.state,
                state_dir=self.args.state_dir,
                action_set=self.args.action_set,
                seed=self.args.seed,
                terminate_on_life_loss=self.args.terminate_on_life_loss,
                terminate_on_level_change=self.args.terminate_on_level_change,
            )

        import stable_retro

        done_on = {}
        if self.args.terminate_on_life_loss:
            done_on["life_loss"] = ("lives", "decrease")
        if self.args.terminate_on_level_change:
            done_on["level_change"] = (("levelHi", "levelLo"), "change")
        env = stable_retro.RetroVecEnv(
            self.args.game,
            state=self.args.state,
            num_envs=1,
            num_threads=1,
            rom_path=str(self.rom_path),
            render_mode="rgb_array",
            use_restricted_actions=stable_retro.Actions.ALL,
            obs_crop=None,
            obs_resize=None,
            obs_grayscale=False,
            obs_resize_algorithm="area",
            frame_skip=self.args.frame_skip,
            frame_stack=1,
            frame_maxpool=False,
            reset_noops=0,
            action_sticky_prob=0.0,
            reward_clip=False,
            info_filter="all",
            obs_layout="chw",
            obs_copy="safe_view",
            done_on=done_on or None,
        )
        if hasattr(env, "seed"):
            env.seed(self.args.seed)
        return env

    def run(self) -> None:
        try:
            self.render()
            while self.running:
                self.poll_events()
                self.policy_step()
                self.render()
                self.frames_rendered += 1
                self.fps_window_frames += 1
                now = time.perf_counter()
                elapsed = now - self.fps_window_start
                if elapsed >= 0.5:
                    self.display_fps = self.fps_window_frames / elapsed
                    self.fps_window_frames = 0
                    self.fps_window_start = now
                if self.args.auto_close_frames is not None and self.frames_rendered >= self.args.auto_close_frames:
                    break
                self.sleep_until_next_frame()
        finally:
            self.close()

    def policy_step(self) -> None:
        action, _ = self.model.predict(self.obs, deterministic=self.args.deterministic)
        self.action = int(np.asarray(action).reshape(-1)[0])
        if self.args.backend == "native":
            obs, rewards, terminated, truncated, infos = self.env.step(
                np.asarray([self.action], dtype=np.uint8),
            )
            terminated_value = bool(terminated[0])
            truncated_value = bool(truncated[0])
        else:
            obs, rewards, dones, infos = self.env.step(self.retro_action_masks[[self.action]])
            terminated_value = bool(dones[0])
            truncated_value = False
        self.obs = obs
        self.step_display_env()
        self.reward += float(rewards[0])
        self.info = dict(infos[0])
        self.step += 1
        self.max_x = max(self.max_x, int(self.info.get("x_pos", 0)))
        completed = self.is_completed()
        if terminated_value or truncated_value or completed:
            self.hold_terminal_frame(completed)
            self.print_episode_summary(completed, terminated_value, truncated_value)
            if self.args.episodes > 0 and self.episode >= self.args.episodes:
                self.running = False
                return
            self.episode += 1
            self.step = 0
            self.reward = 0.0
            self.max_x = 0
            self.info = {}
            self.obs = self.env.reset()
            self.display_obs = self.display_env.reset() if self.display_env is not None else self.obs
            self.display_info = {}

    def step_display_env(self) -> None:
        if self.display_env is None:
            self.display_obs = self.obs
            self.display_info = self.info
            return
        if self.args.backend == "native":
            display_obs, _rewards, _terminated, _truncated, display_infos = self.display_env.step(
                np.asarray([self.action], dtype=np.uint8),
            )
        else:
            display_obs, _rewards, _dones, display_infos = self.display_env.step(
                self.retro_action_masks[[self.action]],
            )
        self.display_obs = display_obs
        self.display_info = dict(display_infos[0])

    def is_completed(self) -> bool:
        if bool(self.info.get("level_complete")) or bool(self.info.get("completion_event")):
            return True
        done_on_info = self.info.get("done_on_info")
        if isinstance(done_on_info, dict) and "level_change" in done_on_info:
            return True
        info_events = self.info.get("info_events")
        if isinstance(info_events, dict) and "level_change" in info_events:
            return True
        if int(self.info.get("levelLo", 0)) > 0:
            return True
        return self.args.completion_x_threshold > 0 and self.max_x >= self.args.completion_x_threshold

    def hold_terminal_frame(self, completed: bool) -> None:
        hold_frames = self.args.hold_complete_frames if completed else self.args.hold_done_frames
        terminal_observation = (
            self.display_info.get("terminal_observation")
            if self.display_env is not None
            else self.info.get("terminal_observation")
        )
        if isinstance(terminal_observation, np.ndarray):
            terminal_batch = terminal_observation[None, ...] if terminal_observation.ndim == 3 else terminal_observation
            if self.display_env is not None:
                self.display_obs = terminal_batch
            else:
                self.obs = terminal_batch
        for _ in range(max(0, hold_frames)):
            self.poll_events()
            if not self.running:
                break
            self.render()
            self.sleep_until_next_frame()

    def print_episode_summary(self, completed: bool, terminated: bool, truncated: bool) -> None:
        summary = {
            "episode": self.episode,
            "steps": self.step,
            "reward": self.reward,
            "max_x": self.max_x,
            "completed": completed,
            "terminated": terminated,
            "truncated": truncated,
            "final_info": self.info,
        }
        print(json.dumps(summary, default=json_default, sort_keys=True), flush=True)

    def poll_events(self) -> None:
        event = ctypes.create_string_buffer(64)
        while self.sdl.SDL_PollEvent(ctypes.byref(event)):
            event_type = ctypes.c_uint32.from_buffer(event).value
            if event_type == SDL_QUIT:
                self.running = False

    def render(self) -> None:
        frame = self.current_display_frame()
        if frame.ndim == 2:
            height, width = frame.shape
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = frame
            rgb[:, :, 1] = frame
            rgb[:, :, 2] = frame
            frame = rgb
        else:
            frame = np.ascontiguousarray(frame)

        if self.sdl.SDL_UpdateTexture(
            self.texture,
            None,
            frame.ctypes.data_as(ctypes.c_void_p),
            frame.strides[0],
        ) != 0:
            raise RuntimeError(self.sdl_error())
        self.sdl.SDL_RenderClear(self.renderer)
        self.sdl.SDL_RenderCopy(self.renderer, self.texture, None, None)
        self.sdl.SDL_RenderPresent(self.renderer)

        now = time.perf_counter()
        if now >= self.next_status_update:
            self.next_status_update = now + 0.1
            title = (
                "SuperMarioBros-Nes-turbo policy player  "
                f"episode={self.episode} step={self.step} "
                f"action={self.action_names[self.action]} "
                f"x={self.info.get('x_pos', 0)} max_x={self.max_x} "
                f"lives={self.info.get('lives', 0)} reward={self.reward:.1f} "
                f"fps={self.display_fps:.0f}"
            )
            self.sdl.SDL_SetWindowTitle(self.window, title.encode("utf-8"))

    def current_display_frame(self) -> np.ndarray:
        obs = self.display_obs if self.args.view == "raw" else self.obs
        return display_frame_from_obs(obs[0], grayscale=self.args.view != "raw")

    def sleep_until_next_frame(self) -> None:
        self.next_tick += self.frame_delay_s
        delay_s = self.next_tick - time.perf_counter()
        if delay_s < -self.frame_delay_s:
            self.next_tick = time.perf_counter() + self.frame_delay_s
            delay_s = self.frame_delay_s
        if delay_s > 0:
            self.sdl.SDL_Delay(max(1, round(delay_s * 1000)))

    def close(self) -> None:
        self.env.close()
        if self.display_env is not None:
            self.display_env.close()
        if getattr(self, "texture", None):
            self.sdl.SDL_DestroyTexture(self.texture)
            self.texture = None
        if getattr(self, "renderer", None):
            self.sdl.SDL_DestroyRenderer(self.renderer)
            self.renderer = None
        if getattr(self, "window", None):
            self.sdl.SDL_DestroyWindow(self.window)
            self.window = None
        if getattr(self, "sdl", None):
            self.sdl.SDL_Quit()

    def sdl_error(self) -> str:
        raw = self.sdl.SDL_GetError()
        return raw.decode("utf-8", errors="replace") if raw else "unknown SDL error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a Stable Baselines3 Mario policy from a local .zip or Hugging Face URL.",
    )
    parser.add_argument("model", help="Local SB3 .zip, HF repo id, or https://huggingface.co/... URL")
    parser.add_argument("--filename", default=None, help="Checkpoint filename inside an HF repo")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument(
        "--backend",
        choices=("stable-retro", "native"),
        default="stable-retro",
        help="stable-retro matches most HF/SB3 checkpoints; native is useful for fast-env parity checks",
    )
    parser.add_argument("--game", default=DEFAULT_GAME)
    parser.add_argument("--rom-path", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--view", choices=("raw", "preprocessed"), default="raw")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=0, help="0 means play forever")
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--action-set", choices=tuple(ACTION_SETS), default="simple")
    parser.add_argument("--completion-x-threshold", type=int, default=3160)
    parser.add_argument("--terminate-on-flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate-on-life-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate-on-level-change", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold-complete-frames", type=int, default=30)
    parser.add_argument("--hold-done-frames", type=int, default=0)
    parser.add_argument("--auto-close-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        SdlPolicyPlayer(args).run()
    except SdlUnavailableError as exc:
        raise SystemExit(f"SDL backend unavailable: {exc}") from exc


if __name__ == "__main__":
    main()
