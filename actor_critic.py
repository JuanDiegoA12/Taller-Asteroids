import time
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def relu(x):
    return np.maximum(0, x)


def relu_derivada(x):
    return (x > 0).astype(np.float64)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def preprocesar(obs):
    return np.array(obs, dtype=np.float64) / 255.0


class Critic:
    def __init__(self, n_entradas, n_ocultas=32, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(n_entradas, n_ocultas) * 0.1
        self.b1 = np.zeros(n_ocultas)
        self.W2 = np.random.randn(n_ocultas, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, s):
        self.s = s
        self.z1 = s @ self.W1 + self.b1
        self.h = relu(self.z1)
        self.v = self.h @ self.W2 + self.b2
        return self.v[0]

    def actualizar(self, delta):
        dV = -delta
        dW2 = np.outer(self.h, dV)
        db2 = dV
        dh = (self.W2 * dV).flatten()
        dz1 = dh * relu_derivada(self.z1)
        dW1 = np.outer(self.s, dz1)
        db1 = dz1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


class Actor:
    def __init__(self, n_entradas, n_acciones, n_ocultas=32, lr=0.001):
        self.lr = lr
        self.n_acciones = n_acciones
        self.W1 = np.random.randn(n_entradas, n_ocultas) * 0.1
        self.b1 = np.zeros(n_ocultas)
        self.W2 = np.random.randn(n_ocultas, n_acciones) * 0.1
        self.b2 = np.zeros(n_acciones)

    def forward(self, s):
        self.s = s
        self.z1 = s @ self.W1 + self.b1
        self.h = relu(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def elegir_accion(self, s):
        probs = self.forward(s)
        accion = np.random.choice(self.n_acciones, p=probs)
        return accion

    def actualizar(self, accion, delta):
        dlog = -self.probs.copy()
        dlog[accion] += 1.0
        dz2 = -delta * dlog
        dW2 = np.outer(self.h, dz2)
        db2 = dz2
        dh = self.W2 @ dz2
        dz1 = dh * relu_derivada(self.z1)
        dW1 = np.outer(self.s, dz1)
        db1 = dz1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


class Curiosidad:
    def __init__(self, tam_max=500, escala=0.5):
        self.memoria = []
        self.tam_max = tam_max
        self.escala = escala

    def bonus(self, s):
        if len(self.memoria) == 0:
            self.memoria.append(s.copy())
            return self.escala
        distancias = [np.linalg.norm(s - m) for m in self.memoria]
        dist_min = min(distancias)
        self.memoria.append(s.copy())
        if len(self.memoria) > self.tam_max:
            self.memoria.pop(0)
        return self.escala * (dist_min / (dist_min + 1.0))


def entrenar(actor, critic, curiosidad, episodios=5, max_pasos=2000, gamma=0.99):
    env = gym.make("ALE/Asteroids-v5", obs_type="ram")
    try:
        for ep in range(1, episodios + 1):
            obs, info = env.reset()
            estado = preprocesar(obs)
            recompensa_total = 0.0

            for paso in range(max_pasos):
                accion = actor.elegir_accion(estado)

                obs2, recompensa, terminado, truncado, info = env.step(accion)
                estado2 = preprocesar(obs2)

                bonus = curiosidad.bonus(estado2)
                r_total = recompensa + bonus

                v_actual = critic.forward(estado)
                v_siguiente = 0.0 if (terminado or truncado) else critic.forward(estado2)
                delta = r_total + gamma * v_siguiente - v_actual

                critic.forward(estado)
                critic.actualizar(delta)
                actor.forward(estado)
                actor.actualizar(accion, delta)

                recompensa_total += recompensa
                estado = estado2

                if terminado or truncado:
                    break

            print(f"Episodio {ep}/{episodios} - Recompensa: {recompensa_total:.0f}")
    finally:
        env.close()


def jugar_demo(actor, max_pasos=2000):
    print("\nMostrando al agente jugar (cierra la ventana para terminar)...")
    env = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human")
    try:
        obs, info = env.reset()
        estado = preprocesar(obs)
        recompensa_total = 0.0

        for paso in range(max_pasos):
            probs = actor.forward(estado)
            accion = int(np.argmax(probs))

            obs, recompensa, terminado, truncado, info = env.step(accion)
            estado = preprocesar(obs)
            recompensa_total += recompensa

            time.sleep(0.02)
            if terminado or truncado:
                break

        print(f"Demo terminada - Recompensa del agente: {recompensa_total:.0f}")
    finally:
        env.close()


def main():
    env = gym.make("ALE/Asteroids-v5", obs_type="ram")
    n_entradas = env.observation_space.shape[0]
    n_acciones = env.action_space.n
    env.close()

    print(f"Estado: {n_entradas} valores, Acciones: {n_acciones}")

    actor = Actor(n_entradas, n_acciones)
    critic = Critic(n_entradas)
    curiosidad = Curiosidad()

    print("\n--- Entrenamiento (Actor-Critic + descubrimiento) ---")
    entrenar(actor, critic, curiosidad, episodios=5)

    jugar_demo(actor)


if __name__ == "__main__":
    main()
