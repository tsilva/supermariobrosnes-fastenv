from __future__ import annotations

import importlib.metadata
import os

import pytest

from scripts import compare_retro_vec_env as compare


REPRESENTATIVE_STATES = (
    "Level1-1",
    "Level1-2",
    "Level1-3",
    "Level1-4",
    "Level2-2",
    "Level2-4",
    "Level8-1",
)
ALL_STABLE_RETRO_STATES = (
    "Level1-1",
    "Level1-1-99lives",
    "Level1-2",
    "Level1-3",
    "Level1-4",
    "Level2-1",
    "Level2-1-clouds",
    "Level2-1-clouds-easy",
    "Level2-2",
    "Level2-3",
    "Level2-4",
    "Level3-1",
    "Level4-1",
    "Level5-1",
    "Level6-1",
    "Level7-1",
    "Level8-1",
)


def require_stable_retro_oracle() -> None:
    rom_path = compare.DEFAULT_ROM.expanduser()
    if not rom_path.exists():
        pytest.skip(f"local SuperMarioBros-Nes ROM is missing: {rom_path}")
    try:
        version = importlib.metadata.version("stable-retro-turbo")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "stable-retro-turbo oracle is not installed; run `uv sync --extra dev` "
            "under Python 3.14",
        )
    assert version == compare.EXPECTED_STABLE_RETRO_VERSION


def sandbox_level1_1_config(
    *,
    state: str = "Level1-1",
    steps: int,
    seed: int = 0,
    num_envs: int = 16,
) -> compare.ComparisonConfig:
    return compare.ComparisonConfig(
        rom_path=compare.DEFAULT_ROM.expanduser(),
        stable_retro_path=None,
        game=compare.DEFAULT_STABLE_RETRO_GAME,
        state=state,
        num_envs=num_envs,
        env_threads=4,
        steps=steps,
        seed=seed,
        frame_skip=4,
        frame_stack=4,
        grayscale=True,
        crop_top=32,
        crop_bottom=0,
        resize_width=84,
        resize_height=84,
        action_set="simple",
        frame_maxpool=False,
        obs_copy="safe_view",
        terminate_on_flag=False,
        terminate_on_life_loss=True,
        terminate_on_level_change=True,
        include_obs=True,
        include_rewards=True,
        include_dones=True,
        include_infos=True,
        stop_on_done=False,
        output_json=None,
        allow_version_mismatch=False,
        preprocessing_matrix=False,
    )


def run_oracle(config: compare.ComparisonConfig) -> dict:
    require_stable_retro_oracle()
    result = compare.run_comparison(config)
    assert result["status"] == "ok"
    assert result["compared_steps"] == config.steps
    return result


@pytest.mark.retro_oracle
def test_stable_retro_vec_env_constructs_with_new_keyword_surface() -> None:
    require_stable_retro_oracle()
    import stable_retro

    rom_path = compare.DEFAULT_ROM.expanduser()
    env = stable_retro.RetroVecEnv(
        compare.DEFAULT_STABLE_RETRO_GAME,
        state="Level1-1",
        num_envs=1,
        num_threads=1,
        rom_path=str(rom_path),
        render_mode="rgb_array",
        use_restricted_actions=stable_retro.Actions.ALL,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        obs_copy="safe_view",
        frame_skip=4,
        frame_stack=4,
        frame_maxpool=False,
        reset_noops=0,
        action_sticky_prob=0.0,
        reward_clip=False,
        info_filter={
            "mode": "terminal",
            "keys": ("lives", "levelHi", "levelLo"),
        },
        done_on={
            "life_loss": ("lives", "decrease"),
            "level_change": (("levelHi", "levelLo"), "change"),
        },
    )
    try:
        assert env.num_envs == 1
        assert getattr(env, "obs_copy", None) == "safe_view"
    finally:
        env.close()


@pytest.mark.retro_oracle
def test_sandbox_profile_matches_stable_retro_past_life_loss_regression_window() -> None:
    run_oracle(sandbox_level1_1_config(steps=2_600))


@pytest.mark.retro_oracle
def test_preprocessing_matrix_matches_raw_and_training_observations() -> None:
    require_stable_retro_oracle()
    result = compare.run_preprocessing_matrix(sandbox_level1_1_config(steps=1_100))
    assert result["status"] == "ok", result
    assert {item["name"] for item in result["results"]} == {
        "rgb_visible_no_crop_no_resize",
        "gray_visible_no_crop_no_resize",
        "gray_crop_no_resize",
        "gray_crop_resize",
    }
    assert all(item["compared_steps"] == 1_100 for item in result["results"])


@pytest.mark.retro_oracle
@pytest.mark.parametrize("state", REPRESENTATIVE_STATES)
def test_representative_saved_states_match_stable_retro_short_trace(state: str) -> None:
    run_oracle(sandbox_level1_1_config(state=state, steps=256, seed=7, num_envs=8))


@pytest.mark.retro_oracle
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SUPERMARIOBROSNES_RETRO_STRESS") != "1",
    reason="set SUPERMARIOBROSNES_RETRO_STRESS=1 to run long stable-retro parity stress tests",
)
def test_level1_1_long_training_profile_stress() -> None:
    run_oracle(sandbox_level1_1_config(steps=20_000))


@pytest.mark.retro_oracle
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SUPERMARIOBROSNES_RETRO_STRESS") != "1",
    reason="set SUPERMARIOBROSNES_RETRO_STRESS=1 to run long stable-retro parity stress tests",
)
@pytest.mark.parametrize("state", ALL_STABLE_RETRO_STATES)
def test_all_stable_retro_states_stress(state: str) -> None:
    run_oracle(sandbox_level1_1_config(state=state, steps=5_000))
