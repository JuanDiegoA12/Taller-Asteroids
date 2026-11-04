import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

MODEL_FILE = "modelo_asteroids.keras"


def create_model(n_inputs, n_actions):
    inputs = layers.Input(shape=(n_inputs,))

    # ReLU activation function
    x = layers.Dense(128, activation="relu")(inputs)
    # ReLU activation function
    x = layers.Dense(128, activation="relu")(x)

    # Softmax activation function / Action selection policy (Actor)
    actor = layers.Dense(n_actions, activation="softmax")(x)
    # State value function (Critic)
    critic = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])
    return model


def preprocess(obs):
    return np.array(obs, dtype=np.float32) / 255.0


class Curiosity:
    def __init__(self, max_size=2000, scale=0.5):
        self.memory = []
        self.max_size = max_size
        self.scale = scale

    def bonus(self, s):
        if len(self.memory) == 0:
            self.memory.append(s.copy())
            # Intrinsic reward based on novelty (curiosity)
            return self.scale
        # Euclidean distance
        distances = np.linalg.norm(np.array(self.memory) - s, axis=1)
        min_dist = float(np.min(distances))
        self.memory.append(s.copy())
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
        # Intrinsic reward based on novelty (curiosity)
        return self.scale * (min_dist / (min_dist + 1.0))


def train(model, episodes=1000, gamma=0.99):
    env = gym.make("ALE/Asteroids-v5", obs_type="ram")

    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    # Critic Huber loss function
    huber_loss = tf.keras.losses.Huber()
    curiosity = Curiosity()

    avg_history = []

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            state = preprocess(obs)

            total_reward = 0.0
            states = []
            actions = []
            rewards = []

            with tf.GradientTape() as tape:
                while True:
                    state_tf = tf.expand_dims(tf.convert_to_tensor(state), 0)

                    # Action selection policy (Actor)
                    action_probs, _ = model(state_tf)
                    action = np.random.choice(
                        env.action_space.n, p=np.squeeze(action_probs)
                    )

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = preprocess(next_obs)

                    # Intrinsic reward based on novelty (curiosity)
                    bonus = curiosity.bonus(next_state)

                    states.append(state_tf)
                    actions.append(action)
                    # Total combined reward
                    rewards.append(reward + bonus)

                    state = next_state
                    total_reward += reward  # solo la recompensa real del juego

                    if terminated or truncated:
                        break

                # Discounted cumulative return
                returns = []
                reward_sum = 0.0
                for r in rewards[::-1]:
                    reward_sum = r + gamma * reward_sum
                    returns.insert(0, reward_sum)
                returns = tf.convert_to_tensor(returns, dtype=tf.float32)

                states_tensor = tf.concat(states, axis=0)
                probs, values = model(states_tensor)
                values = tf.squeeze(values)

                # Advantage function
                advantage = returns - values
                # Advantage normalization
                advantage_norm = (advantage - tf.math.reduce_mean(advantage)) / (
                    tf.math.reduce_std(advantage) + 1e-8
                )

                actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
                indices = tf.range(len(actions))
                action_indices = tf.stack([indices, actions_tensor], axis=1)
                chosen_probs = tf.gather_nd(probs, action_indices)

                # Actor loss function
                actor_loss = tf.math.reduce_mean(
                    -tf.math.log(chosen_probs + 1e-8) * tf.stop_gradient(advantage_norm)
                )

                # Policy entropy
                entropy = tf.math.reduce_mean(
                    -tf.math.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
                )
                # Actor loss function with entropy regularization
                actor_loss = actor_loss - 0.01 * entropy

                # Critic Huber loss function
                critic_loss = huber_loss(returns, values)

                # Total Actor-Critic loss function
                total_loss = actor_loss + critic_loss

            # Adam optimizer
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Métricas de evaluación: recompensa y promedio móvil
            avg_history.append(total_reward)
            average = np.mean(avg_history[-20:])
            print(
                f"Episodio {ep}/{episodes} - Recompensa: {total_reward:.0f} "
                f"- Promedio (ult. 20): {average:.1f}"
            )

        model.save(MODEL_FILE)
        print(f"Modelo guardado en {MODEL_FILE}")

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente.")
        model.save(MODEL_FILE)
        print(f"Modelo guardado en {MODEL_FILE}")
    finally:
        env.close()


def play_demo(model):
    print("\nMostrando al agente jugar (cierra la ventana para terminar)...")
    env = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human")
    try:
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0.0

        while True:
            state_tf = tf.expand_dims(tf.convert_to_tensor(state), 0)
            # Action selection policy (Actor)
            probs, _ = model(state_tf)
            # Muestreamos segun la politica (no solo argmax) para que no se
            # quede clavado si la politica aun esta poco definida.
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs))

            obs, reward, terminated, truncated, _ = env.step(action)
            state = preprocess(obs)
            total_reward += reward

            time.sleep(0.02)
            if terminated or truncated:
                break

        print(f"Demo terminada - Recompensa del agente: {total_reward:.0f}")
    finally:
        env.close()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "todo"

    # Modo "jugar": carga el modelo ya entrenado y solo muestra la demo
    if mode == "jugar":
        print(f"Cargando modelo guardado desde {MODEL_FILE}...")
        model = tf.keras.models.load_model(MODEL_FILE)
        play_demo(model)
        return

    env = gym.make("ALE/Asteroids-v5", obs_type="ram")
    n_inputs = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    env.close()

    print(f"Iniciando TensorFlow - Estado: {n_inputs}, Acciones: {n_actions}")

    model = create_model(n_inputs, n_actions)

    print("\n--- Entrenamiento (TensorFlow Actor-Critic + descubrimiento) ---")
    print("(Puedes cortar con Ctrl+C cuando quieras y pasara a la demo)\n")
    train(model, episodes=1000)

    # Modo "entrenar": solo entrena y guarda; "todo" (por defecto) tambien juega
    if mode != "entrenar":
        play_demo(model)


if __name__ == "__main__":
    main()
