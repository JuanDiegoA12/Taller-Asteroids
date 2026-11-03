import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def crear_modelo(n_entradas, n_acciones):
    entradas = layers.Input(shape=(n_entradas,), batch_size=None)
    
    x = layers.Dense(128, activation="relu")(entradas)
    x = layers.Dense(128, activation="relu")(x)
    
    actor = layers.Dense(n_acciones, activation="softmax")(x)
    
    critico = layers.Dense(1)(x)
    
    modelo = tf.keras.Model(inputs=entradas, outputs=[actor, critico])
    
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0))
    
    return modelo

def preprocesar(obs):
    return np.array(obs, dtype=np.float32) / 255.0

def entrenar_tensorflow(modelo, episodios=1000, gamma=0.99):
    env = gym.make("ALE/Asteroids-v5", obs_type="ram")
    
    optimizador = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    huber_loss = tf.keras.losses.Huber()

    try:
        for ep in range(1, episodios + 1):
            obs, _ = env.reset()
            estado = preprocesar(obs)
            
            recompensa_total = 0
            historial_estados = []
            historial_acciones = []
            historial_recompensas = []
            
            # --- FASE DE RECOLECCIÓN DE EXPERIENCIA ---
            with tf.GradientTape() as tape:
                while True:
                    estado_tf = tf.expand_dims(tf.convert_to_tensor(estado), 0)
                    
                    probabilidades_accion, valores_critico = modelo(estado_tf)
                    
                    accion = np.random.choice(env.action_space.n, p=np.squeeze(probabilidades_accion))
                    
                    obs_siguiente, recompensa, terminado, truncado, _ = env.step(accion)
                    
                    historial_estados.append(estado_tf)
                    historial_acciones.append(accion)
                    historial_recompensas.append(recompensa)
                    
                    estado = preprocesar(obs_siguiente)
                    recompensa_total += recompensa
                    
                    if terminado or truncado:
                        break
                
                recompensas_esperadas = []
                suma_recompensas = 0
                for r in historial_recompensas[::-1]:
                    suma_recompensas = r + gamma * suma_recompensas
                    recompensas_esperadas.insert(0, suma_recompensas)
                
                recompensas_esperadas = np.array(recompensas_esperadas)
                recompensas_esperadas = (recompensas_esperadas - np.mean(recompensas_esperadas)) / (np.std(recompensas_esperadas) + 1e-8)
                
                estados_tensor = tf.concat(historial_estados, axis=0)
                probs, valores = modelo(estados_tensor)
                valores = tf.squeeze(valores)
                
                ventaja = tf.convert_to_tensor(recompensas_esperadas, dtype=tf.float32) - valores
                
                acciones_tensor = tf.convert_to_tensor(historial_acciones, dtype=tf.int32)
                indices = tf.range(len(historial_acciones))
                indices_acciones = tf.stack([indices, acciones_tensor], axis=1)
                probs_elegidas = tf.gather_nd(probs, indices_acciones)
                
                actor_loss = tf.math.reduce_sum(-tf.math.log(probs_elegidas + 1e-8) * tf.stop_gradient(ventaja))
                
                entropia = tf.math.reduce_sum(-probs * tf.math.log(probs + 1e-8))
                actor_loss = actor_loss - (0.01 * entropia)
                critic_loss = huber_loss(tf.convert_to_tensor(recompensas_esperadas, dtype=tf.float32), valores)

                loss_total = actor_loss + critic_loss
            gradientes = tape.gradient(loss_total, modelo.trainable_variables)
            optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
            
            print(f"Episodio {ep}/{episodios} - Recompensa Total: {recompensa_total:.0f}")
            
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente.")
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
            accion = int(np.argmax(probs[0]))

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
    env = gym.make("ALE/Asteroids-v5", obs_type="ram")
    n_entradas = int(env.observation_space.shape[0])
    n_acciones = int(env.action_space.n)
    env.close()

    print(f"Iniciando TensorFlow - Estado: {n_entradas}, Acciones: {n_acciones}")

    modelo = crear_modelo(n_entradas, n_acciones)

    print("\n--- Entrenamiento (TensorFlow A2C) ---")
    entrenar_tensorflow(modelo, episodios=1000)

    jugar_demo_tensorflow(modelo)

if __name__ == "__main__":
    main()