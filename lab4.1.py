import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

# --- 1) Целевая функция и её градиент ---
def F(x, y):
    return -x**2 + 2*x*y + y**2 + np.exp(-x - y)

def gradF(x, y):
    return np.array([
        -2*x + 2*y - np.exp(-x - y),
        2*x + 2*y - np.exp(-x - y)
    ])

# --- 2) Градиенты ограничений ---
grads = {
    'g1': np.array([-1.0,  0.0]),  # -x1 ≤ 0
    'g2': np.array([ 0.0, -1.0]),  # -x2 ≤ 0
    'g3': np.array([-1.0,  1.0]),  # -x1 + x2 ≤ 3
    'g4': np.array([ 1.0,  1.0]),  #  x1 + x2 ≤ 5
}

# --- 3) Вершины области и активные ограничения в них ---
vertices = [
    ((0.0, 0.0), ['g1','g2']),
    ((5.0, 0.0), ['g2','g4']),
    ((1.0, 4.0), ['g3','g4']),
    ((0.0, 3.0), ['g1','g3']),
]

# --- 4) Предвычислим минимума на большой сетке для меток ---
xg = np.linspace(-2, 8, 500)
yg = np.linspace(-2, 8, 500)
Xg, Yg = np.meshgrid(xg, yg)
Zg = F(Xg, Yg)
idx_glob = np.unravel_index(np.argmin(Zg), Zg.shape)
xmin_glob, ymin_glob = Xg[idx_glob], Yg[idx_glob]

maskg = (Yg <= Xg + 3) & (Xg + Yg <= 5) & (Xg >= 0) & (Yg >= 0)
Zg_reg = np.ma.array(Zg, mask=~maskg)
idx_reg = np.unravel_index(np.argmin(Zg_reg), Zg_reg.shape)
xmin_reg, ymin_reg = Xg[idx_reg], Yg[idx_reg]

# --- 5) Соберём текст со значениями F в вершинах ---
f_vals = {pt: F(pt[0], pt[1]) for pt, _ in vertices}
text_vals = "Значения $F$ в вершинах:\n\n" + "\n".join(
    f"({int(x)},{int(y)}) → {f_vals[(x,y)]:.4f}" for (x,y),_ in vertices
)

# --- 6) Рисунок, отступ для текста справа ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(right=0.75)
fig.text(0.78, 0.5, text_vals, fontsize=10, va='center')

# --- 7) Функция отрисовки ---
def redraw(ax):
    # --- ЗАПОМНИТЬ текущие лимиты до очистки ---
    curr_xlim = ax.get_xlim()
    curr_ylim = ax.get_ylim()

    ax.clear()

    x0, x1 = curr_xlim
    y0, y1 = curr_ylim

    # 7.1) Контуры F
    xx = np.linspace(x0, x1, 200)
    yy = np.linspace(y0, y1, 200)
    X, Y = np.meshgrid(xx, yy)
    Z = F(X, Y)

    CS = ax.contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.7)
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.1f")
    # (опционально) цветовая заливка и цветовая шкала
    # CF = ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.3)
    # fig.colorbar(CF, ax=ax, shrink=0.8, label='$F(x,y)$')


    mask = (Y <= X + 3) & (X + Y <= 5) & (X >= 0) & (Y >= 0)
    Zr = np.ma.array(Z, mask=~mask)
    ax.contour(X, Y, Zr, levels=25, colors='black',
               linestyles='dashed', alpha=0.5)

    # 7.2) Заштрихованная область
    poly_verts = np.array([v for v,_ in vertices])
    poly = Polygon(poly_verts, facecolor='lightgreen',
                   alpha=0.3, edgecolor='black', linewidth=1.5)
    ax.add_patch(poly)
    ax.plot(*np.vstack((poly_verts, poly_verts[0])).T, 'k-', lw=1.5)

    # 7.3) Вершины и градиенты
    arrow_len = 0.6
    for (x, y), active in vertices:
        ax.plot(x, y, 'ko', markersize=5)
        ax.text(x+0.1, y+0.1, f'({x:.0f},{y:.0f})', fontsize=9)

        vF = gradF(x, y)
        vF = vF/np.linalg.norm(vF)*arrow_len
        ax.arrow(x, y, *vF, head_width=0.15, color='red', length_includes_head=True)

        cols = ['blue','magenta']
        for ci, gi in enumerate(active):
            vg = grads[gi]
            vg = vg/np.linalg.norm(vg)*arrow_len
            ax.arrow(x, y, *vg, head_width=0.15, color=cols[ci], length_includes_head=True)

    # 7.4) Минимумы
    ax.plot(xmin_reg, ymin_reg, 'r*', markersize=15, label='мин в обл.')
    ax.plot(xmin_glob, ymin_glob, 'g*', markersize=15, label='глоб. мин $F$')

    # 7.5) Оформление и легенда
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_aspect('equal', 'box')

    legend_elems = [
        Patch(facecolor='lightgreen', edgecolor='black', label='допустим. обл.'),
        Line2D([0],[0], color='navy', lw=2, label='контуры $F$'),
        Line2D([0],[0], color='black', ls='--', label='контуры в обл.'),
        Line2D([0],[0], marker='*', color='red',   linestyle='None', markersize=10, label='мин в обл.'),
        Line2D([0],[0], marker='*', color='green', linestyle='None', markersize=10, label='глоб. мин $F$'),
        Line2D([0],[0], lw=2, color='red',    label=r'$\nabla F$'),
        Line2D([0],[0], lw=2, color='blue',   label=r'$\nabla g_i$'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.02, 1.0))

    ax.set_title('Контуры $F$, область допустимых значений и градиенты', pad=20)
    fig.canvas.draw_idle()

# --- 8) Начальный вид до привязки колбэков ---
ax.set_xlim(-2, 8)
ax.set_ylim(-3, 5)

# --- 9) Привязываем перерисовку к зуму/пану и разово рисуем ---
ax.callbacks.connect('xlim_changed', lambda ax: redraw(ax))
ax.callbacks.connect('ylim_changed', lambda ax: redraw(ax))
redraw(ax)

plt.show()
