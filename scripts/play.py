from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from supermariobrosnes_fastenv import ACTION_MEANINGS, SuperMarioBrosVecEnv


DEFAULT_ROM = Path("~/Desktop/roms/NES/mapper-000-NROM/SuperMarioBros-Nes-v0.nes")
NES_WIDTH = 256
NES_HEIGHT = 240

SDL_INIT_VIDEO = 0x00000020
SDL_WINDOWPOS_CENTERED = 0x2FFF0000
SDL_WINDOW_SHOWN = 0x00000004
SDL_RENDERER_ACCELERATED = 0x00000002
SDL_TEXTUREACCESS_STREAMING = 1
SDL_PIXELFORMAT_RGB24 = 0x17101803
SDL_QUIT = 0x100
SDL_KEYDOWN = 0x300
SDL_KEYUP = 0x301
SDLK_RETURN = 13
SDLK_ESCAPE = 27
SDLK_SPACE = 32
SDLK_RIGHT = 1073741903
SDLK_LEFT = 1073741904
SDL_SCANCODE_LSHIFT = 225
SDL_SCANCODE_RSHIFT = 229


class SdlUnavailableError(RuntimeError):
    pass


class SdlExternalVecPlayer:
    """Keyboard player that feeds actions through a one-lane vector env."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.view = args.view
        if self.view == "preprocessed":
            if args.frame_skip <= 0:
                raise ValueError("--frame-skip must be positive")
            if args.frame_stack <= 0:
                raise ValueError("--frame-stack must be positive")
            if args.crop_top < 0 or args.crop_bottom < 0:
                raise ValueError("--crop-top and --crop-bottom must be non-negative")
            frame_skip = args.frame_skip
            grayscale = True
            frame_stack = args.frame_stack
            crop_top = args.crop_top
            crop_bottom = args.crop_bottom
            resize_width = args.resize_width
            resize_height = args.resize_height
        else:
            frame_skip = 1
            grayscale = False
            frame_stack = 1
            crop_top = 0
            crop_bottom = 0
            resize_width = NES_WIDTH
            resize_height = NES_HEIGHT

        self.env = SuperMarioBrosVecEnv(
            rom_path=args.rom_path.expanduser(),
            num_envs=1,
            frame_skip=frame_skip,
            grayscale=grayscale,
            frame_stack=frame_stack,
            terminate_on_flag=False,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            resize_width=resize_width,
            resize_height=resize_height,
            state=args.state,
            state_dir=args.state_dir,
        )
        self.display_grayscale = grayscale
        self.scale = args.scale
        self.frame_delay_s = 1.0 / max(1, args.fps)
        self.obs = self.env.reset()[0]
        initial_frame = display_frame_from_obs(self.obs, self.display_grayscale)
        self.display_height, self.display_width = initial_frame.shape[:2]
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.info: dict[str, object] = {}
        self.frames_rendered = 0
        self.auto_close_frames = args.auto_close_frames

        self.sdl = load_sdl2()
        configure_sdl(self.sdl)
        if self.sdl.SDL_Init(SDL_INIT_VIDEO) != 0:
            raise SdlUnavailableError(self.sdl_error())
        self.sdl.SDL_SetHint(b"SDL_RENDER_SCALE_QUALITY", b"nearest")
        self.window = self.sdl.SDL_CreateWindow(
            b"SuperMarioBros-Nes-turbo external vector player",
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

        self.pressed_keys: set[int] = set()
        self.pressed_scancodes: set[int] = set()
        self.running = True
        self.next_tick = time.perf_counter() + self.frame_delay_s
        self.fps_window_start = time.perf_counter()
        self.fps_window_frames = 0
        self.display_fps = 0.0
        self.next_status_update = 0.0

    def run(self) -> None:
        try:
            self.render()
            while self.running:
                self.poll_events()
                action = self.current_action()
                self.obs, reward, self.terminated, self.truncated, self.info = self.step_one(action)
                self.reward += reward
                if self.terminated or self.truncated:
                    self.obs = self.env.reset()[0]
                    self.reward = 0.0

                self.render()
                self.frames_rendered += 1
                self.fps_window_frames += 1
                now = time.perf_counter()
                elapsed = now - self.fps_window_start
                if elapsed >= 0.5:
                    self.display_fps = self.fps_window_frames / elapsed
                    self.fps_window_frames = 0
                    self.fps_window_start = now
                if self.auto_close_frames is not None and self.frames_rendered >= self.auto_close_frames:
                    break

                self.next_tick += self.frame_delay_s
                delay_s = self.next_tick - time.perf_counter()
                if delay_s < -self.frame_delay_s:
                    self.next_tick = time.perf_counter() + self.frame_delay_s
                    delay_s = self.frame_delay_s
                if delay_s > 0:
                    self.sdl.SDL_Delay(max(1, round(delay_s * 1000)))
        finally:
            self.close()

    def poll_events(self) -> None:
        event = ctypes.create_string_buffer(64)
        while self.sdl.SDL_PollEvent(ctypes.byref(event)):
            event_type = ctypes.c_uint32.from_buffer(event).value
            if event_type == SDL_QUIT:
                self.running = False
            elif event_type in (SDL_KEYDOWN, SDL_KEYUP):
                scancode = ctypes.c_int32.from_buffer(event, 16).value
                keycode = ctypes.c_int32.from_buffer(event, 20).value
                if event_type == SDL_KEYDOWN:
                    self.pressed_scancodes.add(scancode)
                    self.pressed_keys.add(keycode)
                    if keycode == SDLK_ESCAPE:
                        self.running = False
                else:
                    self.pressed_scancodes.discard(scancode)
                    self.pressed_keys.discard(keycode)

    def render(self) -> None:
        frame = display_frame_from_obs(self.obs, self.display_grayscale)
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
                "SuperMarioBros-Nes-turbo external vector player  "
                f"view={self.view} obs={tuple(self.obs.shape)} "
                f"action={ACTION_MEANINGS[self.current_action()]} "
                f"x={self.info.get('x_pos', 0)} lives={self.info.get('lives', 0)} "
                f"reward={self.reward:.1f} fps={self.display_fps:.0f}"
            )
            self.sdl.SDL_SetWindowTitle(self.window, title.encode("utf-8"))

    def step_one(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        obs, rewards, terminated, truncated, infos = self.env.step(
            np.asarray([action], dtype=np.uint8)
        )
        return (
            obs[0],
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            infos[0],
        )

    def current_action(self) -> int:
        if SDLK_RETURN in self.pressed_keys:
            return action_id("start")

        right = SDLK_RIGHT in self.pressed_keys or ord("d") in self.pressed_keys
        left = SDLK_LEFT in self.pressed_keys or ord("a") in self.pressed_keys
        jump = any(key in self.pressed_keys for key in (ord("x"), ord("j"), SDLK_SPACE))
        run = (
            ord("z") in self.pressed_keys
            or ord("k") in self.pressed_keys
            or SDL_SCANCODE_LSHIFT in self.pressed_scancodes
            or SDL_SCANCODE_RSHIFT in self.pressed_scancodes
        )

        if left and not right:
            return action_id("left")
        if right and jump and run:
            return action_id("right_a_b")
        if right and jump:
            return action_id("right_a")
        if right and run:
            return action_id("right_b")
        if right:
            return action_id("right")
        if jump:
            return action_id("a")
        return action_id("noop")

    def close(self) -> None:
        self.env.close()
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


def load_sdl2() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("SDL2"),
        "/opt/homebrew/lib/libSDL2-2.0.0.dylib",
        "/opt/homebrew/lib/libSDL2.dylib",
        "/usr/local/lib/libSDL2-2.0.0.dylib",
        "/usr/local/lib/libSDL2.dylib",
    ]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    details = "; ".join(errors) if errors else "no SDL2 library candidates found"
    raise SdlUnavailableError(details)


def configure_sdl(sdl: ctypes.CDLL) -> None:
    if hasattr(sdl, "SDL_SetMainReady"):
        sdl.SDL_SetMainReady.argtypes = []
        sdl.SDL_SetMainReady.restype = None
        sdl.SDL_SetMainReady()

    sdl.SDL_Init.argtypes = [ctypes.c_uint32]
    sdl.SDL_Init.restype = ctypes.c_int
    sdl.SDL_Quit.argtypes = []
    sdl.SDL_Quit.restype = None
    sdl.SDL_GetError.argtypes = []
    sdl.SDL_GetError.restype = ctypes.c_char_p
    sdl.SDL_SetHint.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    sdl.SDL_SetHint.restype = ctypes.c_int
    sdl.SDL_CreateWindow.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    sdl.SDL_CreateWindow.restype = ctypes.c_void_p
    sdl.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]
    sdl.SDL_DestroyWindow.restype = None
    sdl.SDL_SetWindowTitle.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    sdl.SDL_SetWindowTitle.restype = None
    sdl.SDL_CreateRenderer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    sdl.SDL_CreateRenderer.restype = ctypes.c_void_p
    sdl.SDL_DestroyRenderer.argtypes = [ctypes.c_void_p]
    sdl.SDL_DestroyRenderer.restype = None
    sdl.SDL_CreateTexture.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    sdl.SDL_CreateTexture.restype = ctypes.c_void_p
    sdl.SDL_DestroyTexture.argtypes = [ctypes.c_void_p]
    sdl.SDL_DestroyTexture.restype = None
    sdl.SDL_UpdateTexture.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    sdl.SDL_UpdateTexture.restype = ctypes.c_int
    sdl.SDL_RenderClear.argtypes = [ctypes.c_void_p]
    sdl.SDL_RenderClear.restype = ctypes.c_int
    sdl.SDL_RenderCopy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    sdl.SDL_RenderCopy.restype = ctypes.c_int
    sdl.SDL_RenderPresent.argtypes = [ctypes.c_void_p]
    sdl.SDL_RenderPresent.restype = None
    sdl.SDL_PollEvent.argtypes = [ctypes.c_void_p]
    sdl.SDL_PollEvent.restype = ctypes.c_int
    sdl.SDL_Delay.argtypes = [ctypes.c_uint32]
    sdl.SDL_Delay.restype = None


def action_id(name: str) -> int:
    return ACTION_MEANINGS.index(name)


def latest_frame(obs: np.ndarray) -> np.ndarray:
    if obs.ndim != 3:
        raise ValueError(f"expected CHW observation, got shape {obs.shape}")
    if obs.shape[0] == 1:
        return np.ascontiguousarray(obs[0])
    if obs.shape[0] == 3:
        return np.ascontiguousarray(np.moveaxis(obs, 0, -1))
    raise ValueError(f"play mode expects unstacked grayscale or RGB observation, got shape {obs.shape}")


def display_frame_from_obs(obs: np.ndarray, grayscale: bool) -> np.ndarray:
    if obs.ndim != 3:
        raise ValueError(f"expected CHW observation, got shape {obs.shape}")
    if grayscale:
        return tile_grayscale_channels(obs)
    if obs.shape[0] == 1:
        return np.ascontiguousarray(obs[0])
    if obs.shape[0] == 3:
        return np.ascontiguousarray(np.moveaxis(obs, 0, -1))
    if obs.shape[0] % 3 == 0:
        return tile_rgb_frames(obs)
    return tile_grayscale_channels(obs)


def grid_size(n: int) -> tuple[int, int]:
    cols = 1
    while cols * cols < n:
        cols += 1
    rows = (n + cols - 1) // cols
    return rows, cols


def tile_grayscale_channels(obs: np.ndarray) -> np.ndarray:
    channels, height, width = obs.shape
    rows, cols = grid_size(channels)
    grid = np.zeros((rows * height, cols * width), dtype=np.uint8)
    for channel in range(channels):
        row = channel // cols
        col = channel % cols
        y0 = row * height
        x0 = col * width
        grid[y0 : y0 + height, x0 : x0 + width] = obs[channel]
    return np.ascontiguousarray(grid)


def tile_rgb_frames(obs: np.ndarray) -> np.ndarray:
    frame_count = obs.shape[0] // 3
    height = obs.shape[1]
    width = obs.shape[2]
    rows, cols = grid_size(frame_count)
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for frame_idx in range(frame_count):
        row = frame_idx // cols
        col = frame_idx % cols
        y0 = row * height
        x0 = col * width
        frame = np.moveaxis(obs[frame_idx * 3 : (frame_idx + 1) * 3], 0, -1)
        grid[y0 : y0 + height, x0 : x0 + width] = frame
    return np.ascontiguousarray(grid)


def png_from_frame(frame: np.ndarray) -> bytes:
    if frame.ndim == 2:
        height, width = frame.shape
        color_type = 0
        row_iter = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        height, width, _ = frame.shape
        color_type = 2
        row_iter = frame
    else:
        raise ValueError(f"expected HxW grayscale or HxWx3 RGB frame, got shape {frame.shape}")

    rows = bytearray()
    for row in row_iter:
        rows.append(0)
        rows.extend(row.tobytes())

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(data, checksum)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum & 0xFFFFFFFF)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=1)))
    png.extend(chunk(b"IEND", b""))
    return bytes(png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("external",), default="external")
    parser.add_argument("--view", choices=("raw", "preprocessed"), default="raw")
    parser.add_argument("--rom-path", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=84)
    parser.add_argument("--resize-height", type=int, default=84)
    parser.add_argument("--state", default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--auto-close-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "external":
        raise ValueError(f"unsupported play mode: {args.mode}")
    try:
        SdlExternalVecPlayer(args).run()
    except SdlUnavailableError as exc:
        raise SystemExit(f"SDL backend unavailable: {exc}") from exc


if __name__ == "__main__":
    main()
