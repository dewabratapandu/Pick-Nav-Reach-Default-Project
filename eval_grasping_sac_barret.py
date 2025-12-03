from stable_baselines3 import SAC
from pick_rl_barret_env import PickRLBarretEnv

env = PickRLBarretEnv(gui=True, object_idx = 5)
model = SAC.load("./barret_models/reward_1/best_models_cont/best_model.zip", env=env)  # or pass env later

for ep in range(5):
    obs, _ = env.reset()
    print(f"Observation Initial: {obs}")
    done = False
    ep_reward = 0.0
    steps = 0

    while not done and steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

    print(f"[Eval] Episode {ep}: reward={ep_reward:.3f}, info={info}")

env.close()

