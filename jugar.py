import gymnasium as gym
import ale_py
import time

gym.register_envs(ale_py)

env = gym.make(
    "ALE/Asteroids-v5",
    render_mode="human"
)

obs, info = env.reset()

try:
    for i in range(1000):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(0.02)

finally:
    env.close()