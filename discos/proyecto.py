import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import math

class Particula:
    def __init__(self, pos, rad, masa, rapidez, ang, color):
        '''Constructor que inicializa un disco con una posición, radio, masa, rapidez, angulo y color.
           Además asigana una variable de velocidad v, definida como:

           $$
           v = (rapidez * cos(ang), rapidez * sin(ang))
           $$

           Donde ang es el ángulo en radianes.
           
        Args:
            pos (list): La posición inicial del disco en [x, y]
            rad (float): El radio del disco
            masa (float): La masa de del disco
            rapidez (float): La rapidez del disco
            ang (float): El ángulo de la velocidad del disco en radianes
            color (str): El color del disco para la visualización

        '''
        self.pos = pos  
        self.rad = rad
        self.masa = masa
        self.ang = ang
        self.rapidez = rapidez
        self.v = [rapidez * np.cos(ang), rapidez * np.sin(ang)]
        self.color = color

    def mover(self, dt):
        '''Esta función es un método, actualiza la posición de los discos siguiendo la ecuación
           de posición del movimiento rectilíneo uniforme:

           $$
           x = x_{0} + v_{0}t
           $$

           Donde x es la posición (puede tomar en el eje x y en el eje y), x{0} la posición inicial y t el tiempo.

        Args:
            dt (float): El intervalo de tiempo durante el cual se actualiza la posición

        '''
        self.pos[0] += self.v[0] * dt
        self.pos[1] += self.v[1] * dt

    def display(self, ax):
        '''Función que dibuja el disco con la flecha que representa su velocidad, en el gráfico.
           Mantiene una longitud fija para la flecha y da la dirección de la flecha según el
           ángulo de la velocidad.

        Args:
            ax (matplotlib.axes.Axes): El objeto de los ejes en los que se dibuja los discos
            
        '''
        circulo = plt.Circle(self.pos, self.rad, edgecolor='k', facecolor=self.color)
        ax.add_patch(circulo)
        # Longitud fija para la flecha
        longitud_flecha = 0.1
        # Dirección de la flecha según el ángulo de la velocidad
        direccion = np.arctan2(self.v[1], self.v[0])
        final_pos = [self.pos[0] + longitud_flecha * np.cos(direccion), self.pos[1] + longitud_flecha * np.sin(direccion)]
        ax.quiver(self.pos[0], self.pos[1], final_pos[0] - self.pos[0], final_pos[1] - self.pos[1], angles='xy', scale_units='xy', scale=1, color='k')

    def tiempo_a_pared(self, size):
        '''Esta función calcula el mínimo tiempo de colisión entre un disco y las paredes de la caja.
           Para esto calcula los tiempos de colision con cada pared y toma el menor de estos tiempos.

        Args:
            size (list): Tamaño de la caja en [x, y]

        Returns:
            float (float): El tiempo hasta la colisión con la pared más cercana
            
        '''
        tx_derecha = (size[0] - self.rad - self.pos[0]) / self.v[0] if self.v[0] > 0 else float('inf')
        tx_izquierda = (self.rad - self.pos[0]) / self.v[0] if self.v[0] < 0 else float('inf')
        ty_arriba = (size[1] - self.rad - self.pos[1]) / self.v[1] if self.v[1] > 0 else float('inf')
        ty_abajo = (self.rad - self.pos[1]) / self.v[1] if self.v[1] < 0 else float('inf')
        return min(tx_derecha, tx_izquierda, ty_arriba, ty_abajo)

    @staticmethod
    def tiempo_a_pares(part_a, part_b):
        '''Es una función que se utiliza para declarar un método estático dentro de una clase.
           Calcula el tiempo hasta la colisión entre dos discos.
           En este caso indica que pair_time es un método estático,
           lo que significa que no requiere una instancia de la clase para ser llamado.
           Sabemos que una colisión ocurre cuando la distancia del centro de cada partícula
           a la otra es la suma de ambos radios R:

           $$
           R = r_{a} + r_{b}
           $$

           Ahora para encontrar la ecuación cuadrática que determina el tiempo de colisión hacemos lo siguiente:
           Como estudiamos esto con movimiento relativo, analizamos que este R como la suma de los radios, se cumple cuando:

           $$
           r(0) + vt 
           $$

           Entonces ahora podemos partir de:

           $$
	   \|r(0) + v t \|^2 =  R^2 
           $$

           Por lo que:
           
           $$
           (r(0) + v t)^2 = R^2
           $$

           $$
           \|r(0)\|^2 + 2 r(0) v t + \|v\|^2 t^2 = R^2
           $$

           $$
           \| v \|^2 t^2 + 2 r(0) v t + \| r(0) \|^2 - R^2 = 0
	   $$

           Así, la ecuación cuadrática que determina el tiempo de colisión es:

           $$
           a t^2 + b t + c = 0
           $$

           donde 

           $$
           a = \|v\|^2 
           $$

           $$
           b = 2 (v * r(0))
           $$

           $$
           c = \| r(0) \|^2 - R^2
           $$

           Ahora, aplicando esta física al código tenemos:
 
           det es el discriminante de la ecuación cuadrática que se resolverá para encontrar los tiempos de colisión.
           Este valor determina si hay soluciones reales (y por lo tanto colisiones):

           $$
           det = (2 * b)^2 - 4 * \|v\|^2  * (\|r(0)\|^2  - R^2)
           $$

           Verificamos si hay soluciones reales (o colisión posible) y si las partículas se están acercando una a la otra (b < 0.0).
           sqrt_det calcula la raíz cuadrada del discriminante. Esto para poder resolver la ecuación cuadrática por fórmula general.
           del_t1 y del_t2 son las dos soluciones de la ecuación cuadrática que representan los tiempos en los que las partículas podrían colisionar.

           Luego se selecciona el primer momento futuro en el que ocurre la colisión (el menor tiempo no negativo).
           Si ambos tiempos son negativos o si no hay soluciones reales, del_t se mantiene como infinito (float('inf')),
           indicando que no hay colisión en el futuro.

        Args:
            part_a (Particula): El primer disco
            part_b (Particula): El segundo disco

        Returns:
            float (float): El tiempo hasta la colisión entre dos discos, o infinito si no colisionan
            
        '''
        del_x = [part_b.pos[0] - part_a.pos[0], part_b.pos[1] - part_a.pos[1]]
        del_x_sq = del_x[0]**2 + del_x[1]**2
        del_v = [part_b.v[0] - part_a.v[0], part_b.v[1] - part_a.v[1]]
        del_v_sq = del_v[0]**2 + del_v[1]**2
        b = del_v[0] * del_x[0] + del_v[1] * del_x[1]
        sum_rad = part_a.rad + part_b.rad
        det = (2 * b)**2 - 4 * del_v_sq * (del_x_sq - sum_rad**2)

        if det >= 0.0 and b < 0.0:
            sqrt_det = math.sqrt(det)
            del_t1 = ((-2 * b) + sqrt_det) / (2 * del_v_sq)
            del_t2 = ((-2 * b) - sqrt_det) / (2 * del_v_sq)
            
            # Seleccionar el tiempo mínimo no negativo
            del_t = float('inf')
            if del_t1 >= 0:
                del_t = del_t1
            if del_t2 >= 0 and del_t2 < del_t:
                del_t = del_t2
        else:
            del_t = float('inf')
        return del_t

def inicializar_particulas(posiciones, radios, velocidades, angulos, colores):
    '''Esta función inicializa los atritibutos: posiciones, radios, velocidades, ángulos y colores, de los discos.

    Args:
        posiciones (list): Lista de las posiciones de cada disco
        radios (list): Lista de los radios de cada disco
        velocidades (list): Lista de las velocidades de cada disco
        angulos (list): Lista de los ángulos de cada disco
        colores (list): Lista de los colores de cada disco

    Returns:
        particulas (list): Los atritibutos: posiciones, radios, masas (se definió una masa unitaria), velocidades, angulos y colores, de los discos
    '''
    particulas = []
    for i in range(len(posiciones)):
        pos = posiciones[i]
        radio = radios[i]
        rapidez = velocidades[i]
        ang = angulos[i]
        color = colores[i % len(colores)]
        masa = 1  # Asumimos masa unitaria para simplicidad
        particulas.append(Particula(pos, radio, masa, rapidez, ang, color))
    return particulas

def rebotar_pared(particula, size):
    '''Esta función describe los choques de las partículas con la pared.
       Se asegura de que las partículas reboten elásticamente contra las paredes de la caja.
       Verifica cada componente de la posición respecto a los límites de la caja y
       ajusta tanto la velocidad como la posición de la partícula en caso de una colisión con una pared.

       La condición de si la posiciõn del disco en x es mayor a la longitud de una de las caras de la caja menos el radio
       del disco verifica si la posición en x de la partícula + su radio
       excede el límite derecho de la caja. Si ese es el caso, entonces choca con la pared derecha y entonces

       $$
       v_{x} = -v{x}
       $$

       Y
       
       $$
       x_{0} = size[0] - r
       $$

       Donde v es la velocidad del disco, size[0] la longitud de la caja en x
       y r el radio del disco.
       Esta función invierte la componente x de la velocidad (simulando un rebote elástico) y
       ajusta la posición x para que la partícula no se salga de la caja. 

       Esta misma dinámica cumple con todos los condicionales, cada uno analizando el choque con una pared en específico.

       Args: 
           partícula (class): representa a una partícula individual dentro de la simulación. Se utiliza para poder modificar las propiedades de una partícula (en este caso su posición y velocidad) al detectar colisiones con las paredes de la caja.
           size (tuple): Tupla de dos elementos ´(ancho, alto)´ que representa las dimensiones de la caja en la que se están moviendo las partículas. (´size[0]´ es el ancho de la caja y ´size[1]´ es la altura de la caja, en este caso nuestra caja es de medida 1x1). Se utiliza para delimitar las paredes de la caja y así poder verificar las colisiones con ésta.
    '''
    if particula.pos[0] >= size[0] - particula.rad:
        particula.v[0] = -particula.v[0]
        particula.pos[0] = size[0] - particula.rad
    elif particula.pos[0] <= particula.rad:
        particula.v[0] = -particula.v[0]
        particula.pos[0] = particula.rad
    if particula.pos[1] >= size[1] - particula.rad:
        particula.v[1] = -particula.v[1]
        particula.pos[1] = size[1] - particula.rad
    elif particula.pos[1] <= particula.rad:
        particula.v[1] = -particula.v[1]
        particula.pos[1] = particula.rad

def choque_elastico(part1, part2):
    '''
    Esta función determina las nuevas velocidades de los discos A y B, si dos de ellos chocan entre sí.
    Se tiene la distancia d entre los discos: C_A - C_B, y esta es menor o igual a la suma de sus radios
    entonces es que hay un choque, y se realiza todo el siguiente procedimiento para determina las nuevas
    velocidades de los discos despues del choque:
    La norma de esta distancia d es:

    $$
    \|d\| = \sqrt{\Delta{x}^2 + \Delta{y}^2}
    $$

    Si la distancia d, entre ellos es menor o igual a la suma de sus radios, entonces:
    Se normaliza el vector d, para tener un vector unitario perpendicular $d_{n}$, a la superficie
    de choque entre discos:

    $$
    d_{n} = d/(\|d\|)
    $$
    
    Se determina el vector unitario tangencial $t_{n}$, a la superficie de choque entre los discos,
    como ya se tenia normalizado al vector perpendicular $d_{n}$ entonces:

    $$
    t_{n} = (-\Delta{y},\Delta{x}) / \sqrt{\Delta{x}^2 + \Delta{y}^2}.
    $$

    Ahora se proyectan las velocidades en los vectores unitarios perpendiculares y tangenciales:

    $$
    v_{An1} = v_{A} d_{n}
    $$

    $$
    v_{At1} = v_{A} t_{n} = v_{At2}
    $$

    $$
    v_{Bn1} = v_{B} d_{n}
    $$


    $$
    v_{Bt1} = v_{B} t_{n} = v_{At2}
    $$

    Debido a que la velocidad tangencial es exactamente solo se calcularán la nueva velocidad perpendicular,
    las nuevas velocidades perpendiculares a la superficie de choque unidimensional serían:

    $$
    v_{An2} = (v_{An1} (m_{A} - m_{B}) + 2m_{B}v_{Bn1}) / (m_{A} + m_{B})
    $$

    $$
    v_{Bn2} = (v_{Bn1} (m_{B} - m_{A}) + 2m_{A}v_{An1}) / (m_{A} + m_{B})
    $$


    Estas serían las componentes normales del vector velocidad tras la colisión de ambos cuerpos,
    para hacerlos de nuevo como vectores se toma multiplican por el vector correspondiente que les da su dirección:

    $$
    v_{An2} = v_{An2} d_{n}
    $$


    $$
    v_{At2} = v_{At2} t_{t}
    $$
    
    $$
    v_{Bn2} = v_{Bn2} d_{n}
    $$

    $$
    v_{Bt2} = v_{Bt2} t_{t}
    $$

    Finalmente, las nuevas velocidades tras la colisión en el sistema de coordenadas original vendrian dadas por:

    $$
    v_{A2} = v_{An2} + v_{At2}
    $$

    $$
    v_{B2} = v_{Bn2} + v_{Bt2}
    $$

    Además esta función mueve ligeramente a los discos para que no se queden pegados en la simulación.
    Para esto, mueve al disco A una distancia d_{m} dada por:

    $$
    d_{m} = d_{n} (rad_{1} + rad_{1} - \|d\|) / 2
    $$

    Y mueve al disco B en la dirección opuesta a A:

    $$
    d_{m} = -d_{n} (rad_{1} + rad_{1} - \|d\|) / 2
    $$

    Args:
        part1 (Particula): El disco A del choque
        part2 (Particula): El disco B del choque

    '''
    delta_pos = [part1.pos[0] - part2.pos[0], part1.pos[1] - part2.pos[1]]
    dist = np.linalg.norm(delta_pos)
    if dist <= (part1.rad + part2.rad):
        # Normalización del vector de diferencia de posición
        delta_pos = [delta_pos[0] / dist, delta_pos[1] / dist]
        
        # Vector unitario tangencial
        delta_tan = [-delta_pos[1], delta_pos[0]]
        
        # Proyecciones de velocidad
        v1n = part1.v[0] * delta_pos[0] + part1.v[1] * delta_pos[1]
        v1t = part1.v[0] * delta_tan[0] + part1.v[1] * delta_tan[1]
        v2n = part2.v[0] * delta_pos[0] + part2.v[1] * delta_pos[1]
        v2t = part2.v[0] * delta_tan[0] + part2.v[1] * delta_tan[1]
        
        # Nuevas velocidades normales usando colisión elástica unidimensional
        v1n_prima = (v1n * (part1.masa - part2.masa) + 2 * part2.masa * v2n) / (part1.masa + part2.masa)
        v2n_prima = (v2n * (part2.masa - part1.masa) + 2 * part1.masa * v1n) / (part2.masa + part1.masa)
        
        # Conversión a vectores
        v1n_prima_vec = [v1n_prima * delta_pos[0], v1n_prima * delta_pos[1]]
        v1t_vec = [v1t * delta_tan[0], v1t * delta_tan[1]]
        v2n_prima_vec = [v2n_prima * delta_pos[0], v2n_prima * delta_pos[1]]
        v2t_vec = [v2t * delta_tan[0], v2t * delta_tan[1]]
        
        # Nuevas velocidades
        part1.v = [v1n_prima_vec[0] + v1t_vec[0], v1n_prima_vec[1] + v1t_vec[1]]
        part2.v = [v2n_prima_vec[0] + v2t_vec[0], v2n_prima_vec[1] + v2t_vec[1]]
        
        # Mover ligeramente las partículas para evitar que se queden pegadas
        translape = part1.rad + part2.rad - dist
        mover = translape / 2.0
        part1.pos[0] += delta_pos[0] * mover
        part1.pos[1] += delta_pos[1] * mover
        part2.pos[0] -= delta_pos[0] * mover
        part2.pos[1] -= delta_pos[1] * mover

def actualizar(frame, particulas, size, ax, posiciones_x,dt):
    '''Esta función va creando y actualizando los frames para hacer evolucionar la simulación en el tiempo.
       Primero lmpia el gráfico, luego determina el tiempo mínimo hasta el proximo evento.
       Despúes mueve las partículas por el tiempo mínimo o dt, el que sea menor. 
       
       Luego verifica y maneja las colisiones con las paredes. Si el tiempo mínimo es menor que el dt, entonces se
       da una colisión entre el disco y una pared. Si no es así, entonces el frame continua para que la simulación
       sea continua. Llama a la función rebotar_pared si hay un choque con una pared.
       
       Despúes verifica y maneja las colisiones entre discos y captura las posiciones en x de los discos.
       Si hay colisión entre discos entonces se llama a la función choque_elastico.
       Por último, dibuja los discos en la gráfica.

       Args:
           frame (int): el número de frame
           particulas (list): lista con los atributos de cada disco
           size (tuple): el tamaño de la caja en x y y
           ax (matplotlib.axes.Axes): El objeto de los ejes en los que se dibuja los discos
           posiciones_x (list): las posisciones en x de cada disco
           dt (float): diferencial de tiempo, es la velocidad de evolución de la simulación

    '''
    # Limpiar el gráfico
    ax.clear()
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    
    # Determinar el tiempo mínimo hasta el próximo evento
    tiempos_pared = [p.tiempo_a_pared(size) for p in particulas]
    tiempos_pareja = [Particula.tiempo_a_pares(particulas[i], particulas[j]) for i in range(len(particulas)) for j in range(i+1, len(particulas))]
    all_tiempos = tiempos_pared + tiempos_pareja
    all_tiempos.sort()
    min_tiempo = next(t for t in all_tiempos if t > 1e-14 )  # Umbral de tiempo mínimo

    tiempo_mover= min(min_tiempo, dt)
    # Mover las partículas
        
    for p in particulas:
        p.mover(tiempo_mover+1e-15)
    if min_tiempo <= dt:
        # Verificar colisiones con las paredes
        for p in particulas:
            rebotar_pared(p, size)
        
        # Verificar colisiones entre partículas
        for i in range(len(particulas)):
            for j in range(i+1, len(particulas)):
                choque_elastico(particulas[i], particulas[j])
    
    # Capturar las posiciones en x de las partículas
    posiciones_x.extend([p.pos[0] for p in particulas])
    
    # Dibujar las partículas
    for p in particulas:
        p.display(ax)

def generar_densidad_probabilidad(posiciones_x):
    '''Esta función plotea la distribucion de la densidad de probabilidad de las posiciones en x y plotea un histograma de las posiciones en el eje x.

    Args:
        posiciones_x (list):  Primer argumento (una lista con el valor de las posiciones en x)
        
    ''' 
    plt.figure(figsize=(12, 6))

    # Subplot 1: Distribución de Densidad de Probabilidad
    plt.subplot(1, 2, 1)
    sns.kdeplot(posiciones_x, bw_adjust=0.5, fill=True)
    plt.title('Distribución de Densidad de Probabilidad de las Posiciones en x')
    plt.xlabel('Posición en x')
    plt.ylabel('Densidad de Probabilidad')

    # Subplot 2: Histograma de las Posiciones en x
    plt.subplot(1, 2, 2)
    plt.hist(posiciones_x, bins=100, alpha=0.6, color='g', edgecolor='black')
    plt.title('Histograma de las Posiciones en x')
    plt.xlabel('Posición en x')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()

# Parámetros de simulación
size = (1, 1)  # Tamaño de la caja normalizado
num_particulas = 4  # Se puede cambiar para aumentar el número de discos
radio = 0.1  # Se puede cambiar para variar el radio de los discos
dt = 0.02
velocidad_min = 0.1
velocidad_max = 0.5

# Datos conocidos
posiciones = [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]] #
radios = [radio] * num_particulas
velocidades = [np.random.uniform(velocidad_min, velocidad_max) for _ in range(num_particulas)]
angulos = [np.random.rand() * 2 * np.pi for _ in range(num_particulas)]
colores = ['green', 'orange', 'red', 'blue']

particulas = inicializar_particulas(posiciones, radios, velocidades, angulos, colores)
fig, ax = plt.subplots()
posiciones_x = []

# Configuración de la animación
ani = animation.FuncAnimation(fig, actualizar, frames=120000, fargs=(particulas, size, ax, posiciones_x,dt), interval=5, repeat=False)

# Mostrar la animación
plt.show()

# Generar distribución de densidad de probabilidad después de la simulación
generar_densidad_probabilidad(posiciones_x)
