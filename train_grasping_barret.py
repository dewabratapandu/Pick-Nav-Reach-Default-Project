import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from callbacks import EpisodeOutcomeCallback

from pick_rl_barret_env import PickRLBarretEnv

def make_env():
    return Monitor(PickRLBarretEnv(gui=False))

if __name__ == "__main__":
    exp = "reward_1"
    n_envs = 8
    env = make_vec_env(make_env, n_envs=n_envs)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=500_000,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",
        tensorboard_log=f"./barret/logs/{exp}/logs_sac_grasp",
        device="auto",
    )

    eval_env = make_vec_env(make_env, n_envs=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"./barret_models/{exp}/models",
        name_prefix="sac_grasp",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./barret_models/{exp}/best_models",
        log_path=f"./barret/logs/{exp}/logs_sac_grasp_eval",
        eval_freq=25_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    outcome_callback = EpisodeOutcomeCallback(
        log_every_episodes=1000,
        csv_path=f"./barret/debug/{exp}/episode_outcomes.csv",
        verbose=1,
    )

    model.learn(
        total_timesteps=200_000,
        callback=[checkpoint_callback, eval_callback, outcome_callback],
    )

    model.save(f"./barret_models/{exp}/final_models/sac_grasp_final_200k")
    env.close()
    eval_env.close()
