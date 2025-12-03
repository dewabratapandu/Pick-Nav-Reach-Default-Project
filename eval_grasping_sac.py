from stable_baselines3 import SAC
from pick_gym_env import PickEnv

env = PickEnv(gui=True)
model = SAC.load("./models_sac_grasp_best_phase2/reward_30/best_model.zip", env=env)  # or pass env later

for ep in range(10):
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

