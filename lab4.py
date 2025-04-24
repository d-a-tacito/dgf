import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 1) Целевая функция и её градиент
def F(x, y):
    return -x**2 + 2*x*y + y**2 + np.exp(-x - y)

def gradF(x, y):
    return np.array([
        -2*x + 2*y - np.exp(-x - y),
        2*x + 2*y - np.exp(-x - y)
    ])

# 2) Векторные градиенты ограничений (для g_i(x)<=0):
#    g1 = -x1       → ∇g1 = (-1, 0)
#    g2 = -x2       → ∇g2 = ( 0,-1)
#    g3 = -x1 + x2 - 3 → ∇g3 = (-1, 1)
#    g4 =  x1 + x2 - 5 → ∇g4 = ( 1, 1)
grads = {
    'g1': np.array([-1.0,  0.0]),
    'g2': np.array([ 0.0, -1.0]),
    'g3': np.array([-1.0,  1.0]),
    'g4': np.array([ 1.0,  1.0]),
}

# 3) Вершины допустимого множества и какие ограничения там активны:
#    (0,0): g1=0, g2=0
#    (5,0): g2=0, g4=0
#    (1,4): g3=0, g4=0
#    (0,3): g1=0, g3=0
vertices = [
    ((0.0, 0.0), ['g1','g2']),
    ((5.0, 0.0), ['g2','g4']),
    ((1.0, 4.0), ['g3','g4']),
    ((0.0, 3.0), ['g1','g3']),
]

# 4) Задаём широкую сетку для контуров
x_min, x_max = -2, 8
y_min, y_max = -2, 8
nx, ny = 600, 600
xx = np.linspace(x_min, x_max, nx)
yy = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(xx, yy)
Z = F(X, Y)

# 5) Рисуем контурный график всюду
fig, ax = plt.subplots(figsize=(8, 8))
cont = ax.contour(X, Y, Z, levels=30, cmap='viridis')
ax.clabel(cont, inline=True, fontsize=8)

# 6) Рисуем допустимый многоугольник
poly_verts = np.array([[0,0],[5,0],[1,4],[0,3]])
poly = Polygon(poly_verts, facecolor='none', edgecolor='black', linewidth=2)
ax.add_patch(poly)
ax.plot(*np.vstack((poly_verts, poly_verts[0])).T, 'k-')

# 7) Отмечаем вершины и рисуем векторы
arrow_len = 0.6   # длина стрелок после нормировки

for (x, y), active in vertices:
    ax.plot(x, y, 'ko', markersize=5)
    ax.text(x+0.1, y+0.1, f'({x:.0f},{y:.0f})', fontsize=9)

    # 7.1) градиент функции
    vF = gradF(x, y)
    vF = vF / np.linalg.norm(vF) * arrow_len
    ax.arrow(x, y, vF[0], vF[1], head_width=0.15, color='red',
             length_includes_head=True)

    # 7.2) градиенты активных ограничений
    colors = ['blue','magenta']
    for ci, c in enumerate(active):
        vg = grads[c]
        vg = vg / np.linalg.norm(vg) * arrow_len
        ax.arrow(x, y, vg[0], vg[1], head_width=0.15,
                 color=colors[ci], length_includes_head=True)

# 8) Оформление
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Контуры $F(x_1,x_2)$ на большой области\n'
             'красная стрелка – ∇F, синяя/магента – ∇g_i\n'
             'векторы в вершинах допустимой области', fontsize=14)
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.show()
