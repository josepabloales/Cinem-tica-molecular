# Marco Teórico
## Proyecto final del curso Física Computacional, Universidad de Costa Rica, Semestre I 2024.
Estudiantes: Keylor Rivera, Kevin Vega, José Pablo Alvarado, Mateo Heinz, Ana Laura Víquez

### Cinemática
La cinemática es la rama de la mecánica que estudia el movimiento de los cuerpos. Se enfoca en describir cómo se mueven los objetos en el espacio y el tiempo, utilizando conceptos como la posición, la velocidad y la aceleración.

Cuando el movimiento de una particula es con velocidad constante, es decir cuando la aceleración es cero, este movimiento se denomina movimiento rectilíneo uniforme y la ecuación que describe la posición de una partícula esta descrita por:

$$ 
x = x_0 + v_0t 
$$

Donde x es la posición, x_0 la posición inicial, v_0 la velocidad inicial y t el tiempo.

El momento p, esta definido como:
$$ 
p = mv
$$
Donde m es la masa de la particula.

La energía cinética K, de la partícula es descrita por: 
$$
K = \frac{1}{2}mv^2
$$


### Colisiones elásticas en una dimensión
Cuando 2 partículas A y B chocan, y la fuerza entre los dos es conservativa, se dice que el choque es elástico. En este tipo de choques la energía cinética del sistema es la misma antes y después del choque, por lo que el momento y la energía cinética de cada partícula se conserva:

$$
m_Av_{A1} + m_Bv_{B1} = m_Av_{A2}+ m_Bv_{B2}
$$

$$
\frac{1}{2}m_Av_{A1}^2 + \frac{1}{2}m_Bv_{B1}^2 = \frac{1}{2}m_Av_{A2}^2 + \frac{1}{2}m_Bv_{B2}^2
$$

Combinando estas dos ecuaciones se obtienen las velocidades v_A2 y v_B2:

$$
v_{A2} = \frac{v_{A1}(m_A - m_B) + 2m_Bv_{B1}}{m_A + m_B}
$$

$$
v_{B2} = \frac{v_{B1}(m_B - m_A) + 2m_Av_{A1}}{m_A + m_B}
$$


### Colisiones elásticas en dos dimensiones
Si se considera la colisión elástica entre dos discos sólidos, hay que considerar 2 componentes para la velocidad de cada disco. Ahora si proyectamos los componentes de velocidad de los discos sobre los vectores perpendiculares y tangentes a la superficie de colisión, podemos utilizar la componente de velocidad perpendicular y tratarla como en una colisión de una dimensión.

Para calcular las velocidades de dos entre dos discos sólidos tras una colisión elástica, sin el uso de trigonometría, se puede aplicar lo siguiente. Básicamente se hace a partir de un cambio de coordenadas, los cuales están determinados por los vectores normales y tangenciales a la superficie de choque de los discos. Para el eje tangencial no hay ninguna fuerza involucrada, por lo que su rapidez en ese eje se mantiene constante. Para el eje normal, donde está la fuerza del choque, tan solo se analiza como una colisión en una dimensión, lo que facilita muchísimo el cálculo. Este es el proceso, inicialmente hay que calcular el vector normal a la colisión, este será dado por la distancia entre los centros de cada disco, si tenemos el disco A y el disco B, y sus centros C están dados en coordenadas de la siguiente manera:

$$
C_A = (x_a,y_a) 
$$

$$
C_B = (x_b,y_b)
$$

Entonces, el vector normal será:

$$
\vec{n} = C_A - C_B = (x_a - x_b,y_a - y_b) = (n_x,n_y)
$$

Y de forma unitaria sería: 

$$
\hat{n} = \frac{\vec{n}}{||\vec{n}||} = \frac{(n_x,n_y)}{\sqrt{n_x^2 + n_y^2}}
$$

Ahora es necesario el vector tangente a las superficies, el cual es simplemente una rotación de 90 grados del vector normal, o que básicamente la primera componente del vector tangente es el negativo de la segunda componente del vector normal, y la segunda componente del tangente es igual a la primera del normal. 

$$
\hat{t} = \frac{\vec{t}}{||\vec{n}||} = \frac{(-n_y,n_x)}{\sqrt{n_x^2 + n_y^2}}
$$

Ahora se establecen los vectores de velocidad con respecto al eje de coordenadas original.

$$
\vec{v}_{A1} = (v_{ax},v_{ay})
$$

$$
\vec{v}_{B1}= (v_{bx},v_{by})
$$

A continuación se deben adaptar las componentes de estos vectores en las componentes del nuevo sistema de coordenadas a partir de proyecciones sobre los ejes normales y tangentes, o productos punto con estos. Al crear un sistema de coordenadas de esta forma, se obtiene que la velocidad relacionada al eje normal en ambos discos se tratará como si de una colisión unidimensional se tratase, mientras que para la velocidad en el eje tangente, se mantendrá constante pues no hay fuerza actuando en esa dirección. Sea el subíndice $n$ con respecto a la normal y $t$ con respecto a la tangente. Las componentes de las velocidades de $A$ y $B$ antes de la colisión son:

$$
v_{An1} = \vec{v}_A \cdot \hat{n}
$$

$$
v_{At1} = \vec{v}_A \cdot \hat{t} = v_{At2} (constante)
$$

$$
v_{Bn1} = \vec{v}_B \cdot \hat{n}
$$

$$
v_{Bt1} = \vec{v}_B \cdot \hat{t} = v_{Bt2} (constante)
$$

Tras el choque, la velocidad tangencial será exactamente la misma, ya que no hay fuerza que altere su trayectoria. En el caso de la componente normal, la velocidad tras el choque está dada por las siguientes expresiones:

$$
v_{An2} = \frac{v_{An1}(m_A - m_B) + 2m_Bv_{Bn1}}{m_A + m_B}
$$

$$
v_{Bn2} = \frac{v_{Bn1}(m_B - m_A) + 2m_Av_{An1}}{m_A + m_B}
$$

Estas son las componentes normales del vector velocidad tras la colisión de ambos cuerpos, para hacerlos de nuevo como vectores se toma multiplican por el vector correspondiente que les da su dirección:

$$
\vec{v}_{An2} = {v}_{An2} \hat{n}
$$

$$
\vec{v}_{At2} = {v}_{At2} \hat{t}
$$

$$
\vec{v}_{Bn2} = {v}_{Bn2} \hat{n}
$$

$$
\vec{v}_{Bt2} = {v}_{Bt2} \hat{t}
$$

Finalmente, la velocidad tras la colisión en el sistema de coordenadas original será igual a la suma vectorial de la parte normal más la parte tangente:

$$
\vec{v}_{A2} = \vec{v}_{An2} + \vec{v}_{At2}
$$

$$
\vec{v}_{B2} = \vec{v}_{Bn2} + \vec{v}_{Bt2}
$$


### Bibliografía
Resnick R, Halliday D, Krane KS. Física Volumen 1. Continental; 2001.




