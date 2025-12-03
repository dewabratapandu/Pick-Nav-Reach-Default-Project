import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from callbacks import EpisodeOutcomeCallback

from pick_rl_barret_env import PickRLBarretEnv


def make_env():
    # no GUI for training, keeps things fast
    return Monitor(PickRLBarretEnv(gui=False))


if __name__ == "__main__":
    # --- 1) Vectorized training env: use 4 envs to speed up ---
    n_envs = 8
    exp = "reward_1"
    env = make_vec_env(make_env, n_envs=n_envs)

    # --- 2) Load the best model from phase 1 ---
    # path where EvalCallback saved it earlier
    best_model_path = f"./barret_models/{exp}/best_models_cont/best_model_02.zip"

    model = SAC.load(
        best_model_path,
        env=env,             # VERY important: pass new vec env here
    )

    # --- 3) Separate eval env (single instance, no vectorization) ---
    eval_env = make_vec_env(make_env, n_envs=1)

    # --- 4) New callbacks for phase 2 (save to different dirs) ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,                      # every 50k env steps
        save_path=f"./barret_models/{exp}/models",
        name_prefix="sac_grasp_phase2",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./barret_models/{exp}/best_models_cont",
        log_path=f"./barret/logs/{exp}/logs_sac_grasp_eval_phase2/",
        eval_freq=25_000//n_envs,                      # evaluate every 25k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    outcome_callback = EpisodeOutcomeCallback(
        log_every_episodes=1000,
        csv_path=f"./debug/{exp}/episode_outcomes_800k.csv",
        verbose=1,
    )

    # --- 5) Continue training ---
    # IMPORTANT: reset_num_timesteps=False so TB x-axis continues
    model.learn(
        total_timesteps=200_000,               # extra 100k steps in phase 2
        callback=[checkpoint_callback, eval_callback, outcome_callback],
        reset_num_timesteps=False,
    )

    # --- 6) Save final phase 2 model ---
    model.save(f"./barret_models/{exp}/final_models/sac_grasp_phase2_final_800K.zip")

    env.close()
    eval_env.close()
