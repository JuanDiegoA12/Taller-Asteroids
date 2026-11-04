import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ARCHIVO_MODELO = "modelo_asteroids.keras"


def crear_modelo(n_entradas, n_acciones):
    entradas = layers.Input(shape=(n_entradas,))

    x = layers.Dense(128, activation="relu")(entradas)
    x = layers.Dense(128, activation="relu")(x)

    actor = layers.Dense(n_acciones, activation="softmax")(x)
    critico = layers.Dense(1)(x)

    modelo = tf.keras.Model(inputs=entradas, outputs=[actor, critico])
    return modelo


def preprocesar(obs):
    return np.array(obs, dtype=np.float32) / 255.0


class Curiosidad:
    def __init__(self, tam_max=2000, escala=0.5):
        self.memoria = []
        self.tam_max = tam_max
        self.escala = escala

    def bonus(self, s):
        if len(self.memoria) == 0:
            self.memoria.append(s.copy())
            return self.escala
        distancias = np.linalg.norm(np.array(self.memoria) - s, axis=1)
        dist_min = float(np.min(distancias))
        self.memoria.append(s.copy())
        if len(self.memoria) > self.tam_max:
            self.memoria.pop(0)
        return self.escala * (dist_min / (dist_min + 1.0))


def entrenar_tensorflow(modelo, episodios=1000, gamma=0.99):
    env = gym.make("ALE/Asteroids-v5", obs_type="ram")

    optimizador = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    huber_loss = tf.keras.losses.Huber()
    curiosidad = Curiosidad()

    historial_promedio = []

    try:
        for ep in range(1, episodios + 1):
            obs, _ = env.reset()
            estado = preprocesar(obs)

            recompensa_total = 0.0
            historial_estados = []
            historial_acciones = []
            historial_recompensas = []

            with tf.GradientTape() as tape:
                while True:
                    estado_tf = tf.expand_dims(tf.convert_to_tensor(estado), 0)

                    probabilidades_accion, _ = modelo(estado_tf)
                    accion = np.random.choice(
                        env.action_space.n, p=np.squeeze(probabilidades_accion)
                    )

                    obs_siguiente, recompensa, terminado, truncado, _ = env.step(accion)
                    estado_siguiente = preprocesar(obs_siguiente)

                    # Componente de descubrimiento: recompensa del juego + curiosidad
                    bonus = curiosidad.bonus(estado_siguiente)

                    historial_estados.append(estado_tf)
                    historial_acciones.append(accion)
                    historial_recompensas.append(recompensa + bonus)

                    estado = estado_siguiente
                    recompensa_total += recompensa  # solo la recompensa real del juego

                    if terminado or truncado:
                        break

                # Retorno descontado (sin normalizar: conserva la escala real)
                recompensas_esperadas = []
                suma_recompensas = 0.0
                for r in historial_recompensas[::-1]:
                    suma_recompensas = r + gamma * suma_recompensas
                    recompensas_esperadas.insert(0, suma_recompensas)
                recompensas_esperadas = tf.convert_to_tensor(
                    recompensas_esperadas, dtype=tf.float32
                )

                estados_tensor = tf.concat(historial_estados, axis=0)
                probs, valores = modelo(estados_tensor)
                valores = tf.squeeze(valores)

                # Ventaja = retorno - V(s). El critic es el baseline.
                ventaja = recompensas_esperadas - valores
                # Normalizamos SOLO la ventaja (estabiliza, no borra la escala del retorno)
                ventaja_norm = (ventaja - tf.math.reduce_mean(ventaja)) / (
                    tf.math.reduce_std(ventaja) + 1e-8
                )

                acciones_tensor = tf.convert_to_tensor(historial_acciones, dtype=tf.int32)
                indices = tf.range(len(historial_acciones))
                indices_acciones = tf.stack([indices, acciones_tensor], axis=1)
                probs_elegidas = tf.gather_nd(probs, indices_acciones)

                # reduce_mean: la escala no depende de la duracion del episodio
                actor_loss = tf.math.reduce_mean(
                    -tf.math.log(probs_elegidas + 1e-8) * tf.stop_gradient(ventaja_norm)
                )

                entropia = tf.math.reduce_mean(
                    -tf.math.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
                )
                actor_loss = actor_loss - 0.01 * entropia

                critic_loss = huber_loss(recompensas_esperadas, valores)

                loss_total = actor_loss + critic_loss

            gradientes = tape.gradient(loss_total, modelo.trainable_variables)
            optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))

            # Métricas de evaluación: recompensa y promedio móvil
            historial_promedio.append(recompensa_total)
            promedio = np.mean(historial_promedio[-20:])
            print(
                f"Episodio {ep}/{episodios} - Recompensa: {recompensa_total:.0f} "
                f"- Promedio (ult. 20): {promedio:.1f}"
            )

        modelo.save(ARCHIVO_MODELO)
        print(f"Modelo guardado en {ARCHIVO_MODELO}")

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente.")
        modelo.save(ARCHIVO_MODELO)
        print(f"Modelo guardado en {ARCHIVO_MODELO}")
    finally:
        env.close()


def jugar_demo_tensorflow(modelo):
    print("\nMostrando al agente jugar (cierra la ventana para terminar)...")
    env = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human")
    try:
        obs, _ = env.reset()
        estado = preprocesar(obs)
        recompensa_total = 0.0

        while True:
            estado_tf = tf.expand_dims(tf.convert_to_tensor(estado), 0)
            probs, _ = modelo(estado_tf)
            # Muestreamos segun la politica (no solo argmax) para que no se
            # quede clavado si la politica aun esta poco definida.
            accion = np.random.choice(env.action_space.n, p=np.squeeze(probs))

            obs, recompensa, terminado, truncado, _ = env.step(accion)
            estado = preprocesar(obs)
            recompensa_total += recompensa

            time.sleep(0.02)
            if terminado or truncado:
                break

        print(f"Demo terminada - Recompensa del agente: {recompensa_total:.0f}")
    finally:
        env.close()


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"

    # Modo "jugar": carga el modelo ya entrenado y solo muestra la demo
    if modo == "jugar":
        print(f"Cargando modelo guardado desde {ARCHIVO_MODELO}...")
        modelo = tf.keras.models.load_model(ARCHIVO_MODELO)
        jugar_demo_tensorflow(modelo)
        return

    env = gym.make("ALE/Asteroids-v5", obs_type="ram")
    n_entradas = int(env.observation_space.shape[0])
    n_acciones = int(env.action_space.n)
    env.close()

    print(f"Iniciando TensorFlow - Estado: {n_entradas}, Acciones: {n_acciones}")

    modelo = crear_modelo(n_entradas, n_acciones)

    print("\n--- Entrenamiento (TensorFlow Actor-Critic + descubrimiento) ---")
    print("(Puedes cortar con Ctrl+C cuando quieras y pasara a la demo)\n")
    entrenar_tensorflow(modelo, episodios=1000)

    # Modo "entrenar": solo entrena y guarda; "todo" (por defecto) tambien juega
    if modo != "entrenar":
        jugar_demo_tensorflow(modelo)


if __name__ == "__main__":
    main()
