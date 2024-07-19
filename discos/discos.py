import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import math

class Particula:
    def __init__(self, pos, rad, masa, rapidez, ang, color):
        '''Constructor que inicializa un disco con una posición, radio, masa, rapidez, angulo y color.

        Args:
            pos (list): La posición inicial del disco en [x, y]
            rad (float): El radio del disco
            masa (float): La masa de del disco
            rapidez (float): La rapidez del disco
            ang (float): El ángulo de la velocidad del disco en radianes
            color (str): El color del disco para la visualización

        '''
        self.pos = pos  # Usamos una lista en lugar de un array de NumPy
        self.rad = rad
        self.masa = masa
        self.ang = ang
        self.rapidez = rapidez
        self.v = [rapidez * np.cos(ang), rapidez * np.sin(ang)]
        self.color = color

    def mover(self, dt):
        '''Esta función es un método, actualiza la posición de los discos siguiendo la ecuación de de posición del movimiento rectilíneo uniforme.

        Args:
            dt (float): El intervalo de tiempo durante el cual se actualiza la posición
            
        '''
        self.pos[0] += self.v[0] * dt
        self.pos[1] += self.v[1] * dt

    def display(self, ax):
        '''Función que dibuja el disco con la flecha que representa su velocidad, en el gráfico. Mantiene una longitud fija para la flecha y da la dirección de la flecha según el ángulo de la velocidad.

        Args:
            ax (matplotlib.axes.Axes): El objeto de los ejes en los que se dibuja la partícula
            
        '''
        circle = plt.Circle(self.pos, self.rad, edgecolor='k', facecolor=self.color)
        ax.add_patch(circle)
        # Longitud fija para la flecha
        arrow_length = 0.1
        # Dirección de la flecha según el ángulo de la velocidad
        direction = np.arctan2(self.v[1], self.v[0])
        end_pos = [self.pos[0] + arrow_length * np.cos(direction), self.pos[1] + arrow_length * np.sin(direction)]
        ax.quiver(self.pos[0], self.pos[1], end_pos[0] - self.pos[0], end_pos[1] - self.pos[1], angles='xy', scale_units='xy', scale=1, color='k')

    def wall_time_2d(self, size):
        '''Esta función calcula el minimo tiempo de colisión entre un disco y las paredes de la caja.

        Args:
            size (list of float): Tamaño de la caja en [x, y]

        Returns:
            float (float): El tiempo hasta la colisión con la pared más cercana
            
        '''
        tx_right = (size[0] - self.rad - self.pos[0]) / self.v[0] if self.v[0] > 0 else float('inf')
        tx_left = (self.rad - self.pos[0]) / self.v[0] if self.v[0] < 0 else float('inf')
        ty_top = (size[1] - self.rad - self.pos[1]) / self.v[1] if self.v[1] > 0 else float('inf')
        ty_bottom = (self.rad - self.pos[1]) / self.v[1] if self.v[1] < 0 else float('inf')
        return min(tx_right, tx_left, ty_top, ty_bottom)

    @staticmethod
    def pair_time(part_a, part_b):
        '''Es una función que se utiliza para declarar un método estático dentro de una clase. Calcula el tiempo hasta la colisión entre dos discos.
           En este caso indica que pair_time es un método estático, lo que significa que no requiere una instancia de la clase para ser llamado.
           

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
        scal = del_v[0] * del_x[0] + del_v[1] * del_x[1]
        sigma = part_a.rad + part_b.rad
        Upsilon = (2 * scal)**2 - 4 * del_v_sq * (del_x_sq - sigma**2)

        if Upsilon >= 0.0 and scal < 0.0:
            sqrt_Upsilon = math.sqrt(Upsilon)
            del_t1 = ((-2 * scal) + sqrt_Upsilon) / (2 * del_v_sq)
            del_t2 = ((-2 * scal) - sqrt_Upsilon) / (2 * del_v_sq)
            
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
        particulas (list): Los atritibutos: posiciones, radios, masas (masa unitaria), velocidades, angulos y colores, de los discos
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
    if particula.pos[0] > size[0] - particula.rad:
        particula.v[0] = -particula.v[0]
        particula.pos[0] = size[0] - particula.rad
    elif particula.pos[0] < particula.rad:
        particula.v[0] = -particula.v[0]
        particula.pos[0] = particula.rad
    if particula.pos[1] > size[1] - particula.rad:
        particula.v[1] = -particula.v[1]
        particula.pos[1] = size[1] - particula.rad
    elif particula.pos[1] < particula.rad:
        particula.v[1] = -particula.v[1]
        particula.pos[1] = particula.rad

def choque_elastico(part1, part2):
    """
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
    
    Args:
        part1 (Particula): El disco A del choque
        part2 (Particula): El disco B del choque
    """
    
    delta_pos = [part1.pos[0] - part2.pos[0], part1.pos[1] - part2.pos[1]]
    dist = np.linalg.norm(delta_pos)
    if dist <= (part1.rad + part2.rad):
        #Normalización del vector de diferencia de posición
        delta_pos = [delta_pos[0] / dist, delta_pos[1] / dist]
        
        #Vector unitario tangencial
        delta_tan = [-delta_pos[1], delta_pos[0]]
        
        #Proyecciones de velocidad
        v1n = part1.v[0] * delta_pos[0] + part1.v[1] * delta_pos[1]
        v1t = part1.v[0] * delta_tan[0] + part1.v[1] * delta_tan[1]
        v2n = part2.v[0] * delta_pos[0] + part2.v[1] * delta_pos[1]
        v2t = part2.v[0] * delta_tan[0] + part2.v[1] * delta_tan[1]
        
        #Nuevas velocidades normales usando colisión elástica unidimensional
        v1n_prime = (v1n * (part1.masa - part2.masa) + 2 * part2.masa * v2n) / (part1.masa + part2.masa)
        v2n_prime = (v2n * (part2.masa - part1.masa) + 2 * part1.masa * v1n) / (part1.masa + part2.masa)
        
        #Conversión a vectores
        v1n_prime_vec = [v1n_prime * delta_pos[0], v1n_prime * delta_pos[1]]
        v1t_vec = [v1t * delta_tan[0], v1t * delta_tan[1]]
        v2n_prime_vec = [v2n_prime * delta_pos[0], v2n_prime * delta_pos[1]]
        v2t_vec = [v2t * delta_tan[0], v2t * delta_tan[1]]
        
        #Nuevas velocidades
        part1.v = [v1n_prime_vec[0] + v1t_vec[0], v1n_prime_vec[1] + v1t_vec[1]]
        part2.v = [v2n_prime_vec[0] + v2t_vec[0], v2n_prime_vec[1] + v2t_vec[1]]
        
        #Corrección del solapamiento
        overlap = 0.5 * (part1.rad + part2.rad - dist)
        part1.pos[0] += overlap * delta_pos[0]
        part1.pos[1] += overlap * delta_pos[1]
        part2.pos[0] -= overlap * delta_pos[0]
        part2.pos[1] -= overlap * delta_pos[1]
        
def actualizar(frame, particulas, ax, size, posiciones_x,dt):
    '''Esta función va creando y actualizando los frames para hacer evolucionar la simulación en el tiempo.

    '''
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    for part in particulas:
        part.mover(dt)
        rebotar_pared(part, size)
        part.display(ax)
    for i in range(len(particulas)):
        for j in range(i + 1, len(particulas)):
            choque_elastico(particulas[i], particulas[j])
    ax.set_title(f't = {frame * dt:.2f}')
    # Almacenar las posiciones en x de las partículas
    for part in particulas:
        posiciones_x.append(part.pos[0])

##def actualizar(frame, particulas, ax, size, posiciones_x, dt):
##    ax.clear()
##    ax.set_xlim(0, size[0])
##    ax.set_ylim(0, size[1])
##    ax.set_aspect('equal')
##
##    # Encuentra el tiempo mínimo hasta la próxima colisión
##    min_t = float('inf')
##    colisiones = []
##
##    for i, part in enumerate(particulas):
##        # Tiempo hasta la colisión con la pared
##        wall_t = part.wall_time_2d(size)
##        if wall_t < min_t:
##            min_t = wall_t
##            colisiones = [(part, None)]  # Partícula y pared
##
##        for j in range(i + 1, len(particulas)):
##            # Tiempo hasta la colisión entre dos partículas
##            pair_t = Particula.pair_time(part, particulas[j])
##            if pair_t < min_t:
##                min_t = pair_t
##                colisiones = [(part, particulas[j])]  # Dos partículas
##
##    # Mover partículas por el tiempo mínimo de colisión
##    for part in particulas:
##        part.mover(min_t)
##
##    # Procesar colisiones
##    for part, other in colisiones:
##        if other is None:
##            # Colisión con la pared
##            rebotar_pared(part, size)
##        else:
##            # Colisión entre dos partículas
##            choque_elastico(part, other)
##
##    # Mover partículas por el tiempo restante después de la colisión
##    remaining_time = dt - min_t
##    for part in particulas:
##        part.mover(remaining_time)
##
##    # Procesar colisiones restantes después del movimiento adicional
##    for part in particulas:
##        rebotar_pared(part, size)
##    for i in range(len(particulas)):
##        for j in range(i + 1, len(particulas)):
##            choque_elastico(particulas[i], particulas[j])
##
##    # Dibujar las partículas y sus direcciones
##    for part in particulas:
##        part.display(ax)
##
##    ax.set_title(f't = {frame * dt:.2f}')
##    for part in particulas:
##        posiciones_x.append(part.pos[0])
    
def generar_densidad_probabilidad(posiciones_x):
    '''Esta función plotea la distribucion de la densidad de probabilidad de las posiciones en x y plotea un histograma de las posiciones en el eje x.

    Args:
        posiciones_x (array):  Primer argumento (un array con el valor de las posiciones en x)
        
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
posiciones = [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]]
radios = [radio] * num_particulas
velocidades = [np.random.uniform(velocidad_min, velocidad_max) for _ in range(num_particulas)]
angulos = [np.random.rand() * 2 * np.pi for _ in range(num_particulas)]
colores = ['green', 'orange', 'red', 'blue']

particulas = inicializar_particulas(posiciones, radios, velocidades, angulos, colores)
fig, ax = plt.subplots()
posiciones_x = []

ani = animation.FuncAnimation(fig, actualizar, fargs=(particulas, ax, size, posiciones_x,dt), frames=120000, interval=10, repeat=False)

# Mostrar la animación
plt.show()

#Generar distribución de densidad de probabilidad después de la simulación
generar_densidad_probabilidad(posiciones_x)


