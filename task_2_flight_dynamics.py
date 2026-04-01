import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider, Button
from math import atan2, sqrt, degrees, radians
import os
import sys

#   CONSTANTES
G = 9.81  # Aceleración gravitacional [m/s²]

# Perfil de misión (fases)
MISSION_PHASES = [
    {"name": "Recto y Nivelado",   "t_start": 0,  "t_end": 15, "color": "#45A29E", "alpha": 0.15},
    {"name": "Viraje Coordinado",  "t_start": 15, "t_end": 30, "color": "#5C7A8C", "alpha": 0.15},
    {"name": "Ascenso (Climb)",    "t_start": 30, "t_end": 40, "color": "#C05A35", "alpha": 0.15},
    {"name": "Descenso (Descent)", "t_start": 40, "t_end": 50, "color": "#8B2500", "alpha": 0.15},
    {"name": "Nivelación",         "t_start": 50, "t_end": 60, "color": "#D4AF37", "alpha": 0.15},
]

#   MÓDULO 1: LECTURA DE DATOS

def load_imu_data(filepath):
    df = pd.read_csv(filepath)
    return {
        'time': df['time_s'].values,
        'p':    df['gyro_p_rad_s'].values,    # Roll rate  [rad/s]
        'q':    df['gyro_q_rad_s'].values,    # Pitch rate [rad/s]
        'r':    df['gyro_r_rad_s'].values,    # Yaw rate   [rad/s]
        'ax':   df['accel_x_m_s2'].values,    # Aceleración body-x [m/s²]
        'ay':   df['accel_y_m_s2'].values,    # Aceleración body-y [m/s²]
        'az':   df['accel_z_m_s2'].values,    # Aceleración body-z [m/s²]
    }

def load_ground_truth(filepath):
    df = pd.read_csv(filepath)
    return {
        'time': df['time_s'].values,
        'x':    df['gt_pos_x_m'].values,   # Posición North [m]
        'y':    df['gt_pos_y_m'].values,   # Posición East  [m]
        'z':    df['gt_pos_z_m'].values,   # Posición Down  [m]
    }

#   MÓDULO 2: PROPAGACIÓN DE ACTITUD (Cinemática de Euler)

def calculate_H(phi, theta):
    sp, cp = np.sin(phi), np.cos(phi)
    tt = np.tan(theta)
    st = 1.0 / np.cos(theta)  # sec(theta)
    
    H = np.array([
        [1.0,  sp * tt,  cp * tt],
        [0.0,  cp,      -sp     ],
        [0.0,  sp * st,  cp * st]
    ])
    return H


def propagate_attitude(time, p, q, r, phi0=0.0, theta0=0.0, psi0=0.0):
    n = len(time)
    phi   = np.zeros(n)
    theta = np.zeros(n)
    psi   = np.zeros(n)
    
    # Condiciones iniciales
    phi[0]   = phi0
    theta[0] = theta0
    psi[0]   = psi0
    
    for i in range(1, n):
        dt = time[i] - time[i-1]
        
        # Vector de velocidades angulares del body
        omega_b = np.array([p[i-1], q[i-1], r[i-1]])
        
        # Calcular Matriz H con los ángulos actuales
        H = calculate_H(phi[i-1], theta[i-1])
        
        # Tasas de cambio de ángulos de Euler: Φ̇ = H × ω_B
        euler_rates = H @ omega_b
        
        # Integración numérica (Euler forward)
        phi[i]   = phi[i-1]   + euler_rates[0] * dt
        theta[i] = theta[i-1] + euler_rates[1] * dt
        psi[i]   = psi[i-1]   + euler_rates[2] * dt
    
    return phi, theta, psi

#   MÓDULO 3: NAVEGACIÓN (Velocidad y Posición en NED)

def body_to_ned_dcm(phi, theta, psi):
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    
    # Rz(ψ) × Ry(θ) × Rx(φ) — secuencia aeronáutica ZYX
    DCM = np.array([
        [ct*cy,                  ct*sy,                  -st    ],
        [sp*st*cy - cp*sy,       sp*st*sy + cp*cy,       sp*ct  ],
        [cp*st*cy + sp*sy,       cp*st*sy - sp*cy,       cp*ct  ]
    ])
    return DCM


def propagate_navigation(time, ax, ay, az, phi, theta, psi):
    n = len(time)
    vel_ned = np.zeros((n, 3))  # [V_N, V_E, V_D]
    pos_ned = np.zeros((n, 3))  # [P_N, P_E, P_D]
    
    # Vector de gravedad en NED [m/s²]
    gravity_ned = np.array([0.0, 0.0, G])
    
    for i in range(1, n):
        dt = time[i] - time[i-1]
        
        # 1. Aceleración en body frame
        a_body = np.array([ax[i-1], ay[i-1], az[i-1]])
        
        # 2. Rotar body → NED usando DCM
        C_eb = body_to_ned_dcm(phi[i-1], theta[i-1], psi[i-1])
        a_ned = C_eb @ a_body
        
        # 3. Restar gravedad (obtener aceleración cinemática neta)
        a_net = a_ned - gravity_ned
        
        # 4. Integrar aceleración → velocidad (Euler forward)
        vel_ned[i] = vel_ned[i-1] + a_net * dt
        
        # 5. Integrar velocidad → posición (Euler forward)
        pos_ned[i] = pos_ned[i-1] + vel_ned[i] * dt
    
    return vel_ned, pos_ned

#   MÓDULO 4: CUATERNIONES

def dcm_to_quaternion(C):
    T = np.trace(C)  # T = C[0,0] + C[1,1] + C[2,2]
    
    # Candidatos para encontrar el máximo (evitar raíz de número negativo)
    candidates = [T, C[0,0], C[1,1], C[2,2]]
    max_idx = np.argmax(candidates)
    
    if max_idx == 0:
        # T es máximo
        q0 = 0.5 * np.sqrt(1 + T)
        s = 1.0 / (4.0 * q0)
        q1 = (C[1,2] - C[2,1]) * s
        q2 = (C[2,0] - C[0,2]) * s
        q3 = (C[0,1] - C[1,0]) * s
    elif max_idx == 1:
        # C[0,0] es máximo
        q1 = 0.5 * np.sqrt(1 + 2*C[0,0] - T)
        s = 1.0 / (4.0 * q1)
        q0 = (C[1,2] - C[2,1]) * s
        q2 = (C[0,1] + C[1,0]) * s
        q3 = (C[2,0] + C[0,2]) * s
    elif max_idx == 2:
        # C[1,1] es máximo
        q2 = 0.5 * np.sqrt(1 + 2*C[1,1] - T)
        s = 1.0 / (4.0 * q2)
        q0 = (C[2,0] - C[0,2]) * s
        q1 = (C[0,1] + C[1,0]) * s
        q3 = (C[1,2] + C[2,1]) * s
    else:
        # C[2,2] es máximo
        q3 = 0.5 * np.sqrt(1 + 2*C[2,2] - T)
        s = 1.0 / (4.0 * q3)
        q0 = (C[0,1] - C[1,0]) * s
        q1 = (C[2,0] + C[0,2]) * s
        q2 = (C[1,2] + C[2,1]) * s
    
    q = np.array([q0, q1, q2, q3])
    # Normalizar para garantizar cuaternión unitario
    q = q / np.linalg.norm(q)
    # Convención: q0 siempre positivo
    if q[0] < 0:
        q = -q
    return q


def euler_to_quaternion(phi, theta, psi):
    cp2 = np.cos(phi / 2)
    sp2 = np.sin(phi / 2)
    ct2 = np.cos(theta / 2)
    st2 = np.sin(theta / 2)
    cy2 = np.cos(psi / 2)
    sy2 = np.sin(psi / 2)
    
    q0 = cp2 * ct2 * cy2 + sp2 * st2 * sy2
    q1 = sp2 * ct2 * cy2 - cp2 * st2 * sy2
    q2 = cp2 * st2 * cy2 + sp2 * ct2 * sy2
    q3 = cp2 * ct2 * sy2 - sp2 * st2 * cy2
    
    q = np.array([q0, q1, q2, q3])
    if q[0] < 0:
        q = -q
    return q


def quaternion_angle_axis(q):
    # Asegurar que q0 está en rango [-1, 1] para arccos
    q0_clamped = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(q0_clamped)
    
    # Vector parte del cuaternión
    v = q[1:4]
    
    sin_half = np.sin(angle / 2.0)
    if abs(sin_half) > 1e-10:
        axis = v / sin_half
    else:
        axis = np.array([1.0, 0.0, 0.0])  # Rotación nula → eje arbitrario
    
    return angle, axis


def compute_all_quaternions(phi_arr, theta_arr, psi_arr):
    n = len(phi_arr)
    quats  = np.zeros((n, 4))
    angles = np.zeros(n)
    axes   = np.zeros((n, 3))
    
    for i in range(n):
        q = euler_to_quaternion(phi_arr[i], theta_arr[i], psi_arr[i])
        quats[i] = q
        ang, ax = quaternion_angle_axis(q)
        angles[i] = ang
        axes[i]   = ax
    
    return quats, angles, axes

#   MÓDULO 5: VISUALIZACIÓN / INTERFAZ GRÁFICA

# Colores y nombres de fases en inglés
PHASE_COLORS_DASH = ['#45A29E', '#5C7A8C', '#C05A35', '#8B2500', '#D4AF37'] 
PHASE_NAMES_EN = ['Straight & Level', 'Coordinated Turn', 'Climb', 'Descent', 'Level Again']

# Fondo oscuro global
DARK_BG  = '#1E1A18' # Very dark charcoal brown (tarnished iron)
DARK_AX  = '#2C2520' # Gunmetal / dark rust
GRID_CLR = '#A67C52' # Bronze / brass
TXT_CLR  = '#E6C280' # Antique gold


def _get_phase_indices(time):
    """Retorna lista de (mask, color, name) por cada fase de misión."""
    phases = []
    for i, ph in enumerate(MISSION_PHASES):
        mask = (time >= ph['t_start']) & (time < ph['t_end'])
        phases.append((mask, PHASE_COLORS_DASH[i], PHASE_NAMES_EN[i]))
    return phases


def _style_2d_ax(ax, xlabel='', ylabel='', title=''):
    """Aplica tema oscuro a un subplot 2D."""
    ax.set_facecolor(DARK_AX)
    ax.set_xlabel(xlabel, color=TXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=TXT_CLR, fontsize=9)
    if title:
        ax.set_title(title, color=TXT_CLR, fontsize=10, fontweight='bold', pad=6)
    ax.tick_params(colors=TXT_CLR, labelsize=8)
    ax.grid(True, color=GRID_CLR, alpha=0.5, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)


def _style_3d_ax(ax, title=''):
    """Aplica tema oscuro a un subplot 3D."""
    ax.set_facecolor(DARK_AX)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_CLR); ax.yaxis.pane.set_edgecolor(GRID_CLR)
    ax.zaxis.pane.set_edgecolor(GRID_CLR)
    ax.tick_params(colors=TXT_CLR, labelsize=7)
    ax.xaxis.label.set_color(TXT_CLR); ax.yaxis.label.set_color(TXT_CLR)
    ax.zaxis.label.set_color(TXT_CLR)
    if title:
        ax.set_title(title, color=TXT_CLR, fontsize=10, fontweight='bold', pad=4)
    ax.grid(True, color=GRID_CLR, alpha=0.3)


def _plot_phased_3d(ax, time, x, y, z, phases):
    """Dibuja trayectoria 3D coloreada por fase de misión."""
    for mask, color, name in phases:
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        # Extender 1 punto para continuidad visual
        i0, i1 = idx[0], min(idx[-1] + 2, len(time))
        ax.plot(x[i0:i1], y[i0:i1], z[i0:i1], color=color, linewidth=1.5,
                label=name, alpha=0.9)


def _plot_phased_2d(ax, time, x, y, phases):
    """Dibuja trayectoria 2D coloreada por fase de misión."""
    for mask, color, name in phases:
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        i0, i1 = idx[0], min(idx[-1] + 2, len(time))
        ax.plot(x[i0:i1], y[i0:i1], color=color, linewidth=1.5, label=name, alpha=0.9)


def _add_phase_bands_dark(ax, time):
    """Bandas verticales semitransparentes de fases de misión (tema oscuro)."""
    for i, ph in enumerate(MISSION_PHASES):
        t0 = max(ph['t_start'], time[0])
        t1 = min(ph['t_end'], time[-1])
        if t0 < t1:
            ax.axvspan(t0, t1, alpha=0.10, color=PHASE_COLORS_DASH[i])
    # Línea divisoria punteada en t=15 (inicio del viraje)
    ax.axvline(x=15, color='white', linestyle='--', linewidth=0.6, alpha=0.3)


import matplotlib.animation as animation

def plot_dashboard(results):
    """
    Dashboard unificado con tema oscuro, animado con slider y autoplayer.
    """
    time    = results['time']
    phi     = results['phi']
    theta   = results['theta']
    psi     = results['psi']
    vel_ned = results['vel_ned']
    pos_ned = results['pos_ned']
    gt      = results['gt']
    
    phases = _get_phase_indices(time)
    
    # Altitud: -Down para que positivo = arriba
    alt_ins = -pos_ned[:, 2]
    alt_gt  = -gt['z']
    
    # ---------- Crear figura con tema oscuro ----------
    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig.canvas.manager.set_window_title("AHRS Dashboard — Animado")
    
    # GridSpec con mayor hspace para evitar choques
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.1],
                          hspace=0.6, wspace=0.22,
                          left=0.05, right=0.97, top=0.92, bottom=0.12)
    
    ax1 = fig.add_subplot(gs[0:2, 0], projection='3d')
    _style_3d_ax(ax1, 'INS puro – drift acumulado')
    
    ax2 = fig.add_subplot(gs[2:4, 0], projection='3d')
    _style_3d_ax(ax2, 'Ground Truth – trayectoria real')
    
    ax3 = fig.add_subplot(gs[0, 1])
    _style_2d_ax(ax3, 'Tiempo [s]', 'Ángulo [°]', 'Ángulos de Euler – Perfil de Misión')
    
    ax4 = fig.add_subplot(gs[1, 1])
    _style_2d_ax(ax4, 'Tiempo [s]', 'Vel [m/s]', 'Velocidades NED')
    
    ax5 = fig.add_subplot(gs[2, 1])
    _style_2d_ax(ax5, 'East [m]', 'North [m]', 'Trayectoria 2D – INS (drift acumulado)')
    
    ax6 = fig.add_subplot(gs[3, 1])
    _style_2d_ax(ax6, 'East [m]', 'North [m]', 'Trayectoria 2D – Ground Truth (escala real)')

    # ── Fondos Pálidos (Trayectorias completas semitransparentes) ──
    for mask, color, name in phases:
        idx = np.where(mask)[0]
        if len(idx) < 2: continue
        i0, i1 = idx[0], min(idx[-1] + 2, len(time))
        ax1.plot(pos_ned[i0:i1,0], pos_ned[i0:i1,1], alt_ins[i0:i1], color=color, linewidth=1.0, alpha=0.15)
        ax2.plot(gt['x'][i0:i1], gt['y'][i0:i1], alt_gt[i0:i1], color=color, linewidth=1.0, alpha=0.15)
        ax5.plot(pos_ned[i0:i1,1], pos_ned[i0:i1,0], color=color, linewidth=1.0, alpha=0.15)
        ax6.plot(gt['y'][i0:i1], gt['x'][i0:i1], color=color, linewidth=1.0, alpha=0.15)

    _add_phase_bands_dark(ax3, time)
    _add_phase_bands_dark(ax4, time)

    # ── Líneas Principales Animadas ──
    ax1_lines = [ax1.plot([], [], [], color=color, linewidth=2.0)[0] for _, color, _ in phases]
    ax2_lines = [ax2.plot([], [], [], color=color, linewidth=2.0)[0] for _, color, _ in phases]
    ax5_lines = [ax5.plot([], [], color=color, linewidth=2.0)[0] for _, color, _ in phases]
    ax6_lines = [ax6.plot([], [], color=color, linewidth=2.0)[0] for _, color, _ in phases]

    dot_ins_3d = ax1.scatter([], [], [], color='white', s=60, edgecolors='#e74c3c', linewidth=2, zorder=10)
    dot_gt_3d  = ax2.scatter([], [], [], color='white', s=60, edgecolors='#2ecc71', linewidth=2, zorder=10)
    dot_ins_2d = ax5.scatter([], [], color='white', s=60, edgecolors='#e74c3c', linewidth=2, zorder=10)
    dot_gt_2d  = ax6.scatter([], [], color='white', s=60, edgecolors='#2ecc71', linewidth=2, zorder=10)

    line_phi,   = ax3.plot([], [], color='#e74c3c', linewidth=1.5, label='Roll φ')
    line_theta, = ax3.plot([], [], color='#2ecc71', linewidth=1.5, label='Pitch θ')
    line_psi,   = ax3.plot([], [], color='#f1c40f', linewidth=1.5, label='Yaw ψ')
    
    line_vn, = ax4.plot([], [], color='#e74c3c', linewidth=1.5, label='Vn (Norte)')
    line_ve, = ax4.plot([], [], color='#3498db', linewidth=1.5, label='Ve (Este)')
    line_vd, = ax4.plot([], [], color='#2ecc71', linewidth=1.5, label='Vd (Abajo)')

    ax1.set_xlim(pos_ned[:,0].min(), pos_ned[:,0].max()); ax1.set_ylim(pos_ned[:,1].min(), pos_ned[:,1].max())
    ax1.set_zlim(alt_ins.min(), alt_ins.max())
    ax2.set_xlim(gt['x'].min(), gt['x'].max()); ax2.set_ylim(gt['y'].min(), gt['y'].max())
    ax2.set_zlim(alt_gt.min(), alt_gt.max())
    ax3.set_xlim(time[0], time[-1]); ax3.set_ylim(-180, 180)
    ax4.set_xlim(time[0], time[-1]); ax4.set_ylim(vel_ned.min()*1.1, vel_ned.max()*1.1)
    ax5.set_xlim(pos_ned[:,1].min(), pos_ned[:,1].max()); ax5.set_ylim(pos_ned[:,0].min(), pos_ned[:,0].max())
    ax6.set_xlim(gt['y'].min(), gt['y'].max()); ax6.set_ylim(gt['x'].min(), gt['x'].max())

    ax3.legend(fontsize=8, loc='upper left', facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TXT_CLR)
    ax4.legend(fontsize=8, loc='upper left', facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TXT_CLR)

    ax_slider = fig.add_axes([0.15, 0.02, 0.65, 0.03], facecolor=DARK_AX)
    from matplotlib.widgets import Slider, Button
    slider = Slider(ax_slider, 'Tiempo [s]', time[0], time[-1], valinit=time[0], color='#C05A35')
    slider.label.set_color(TXT_CLR); slider.valtext.set_color(TXT_CLR)

    ax_play = fig.add_axes([0.85, 0.015, 0.08, 0.04])
    btn_play = Button(ax_play, '▶ Play', color='#5C7A8C', hovercolor='#45A29E')
    btn_play.label.set_fontweight('bold'); btn_play.label.set_color(DARK_BG)

    def draw_lines_truncated(x, y, z=None, idx_cut=0):
        """Devuelve datos truncados separados por fases para actualizar las líneas (x, y, z o x, y)."""
        data = []
        for (mask, _, _) in phases:
            valid_idx = np.where(mask)[0]
            valid_idx = valid_idx[valid_idx <= idx_cut]
            if len(valid_idx) < 2:
                data.append(([], [], [])) if z is not None else data.append(([], []))
                continue
            i0, i1 = valid_idx[0], valid_idx[-1] + 1
            if z is not None:
                data.append((x[i0:i1], y[i0:i1], z[i0:i1]))
            else:
                data.append((x[i0:i1], y[i0:i1]))
        return data

    def update(val):
        t_current = slider.val
        idx = np.argmin(np.abs(time - t_current))
        
        # Actualizar líneas de 3D INS
        for ln, d in zip(ax1_lines, draw_lines_truncated(pos_ned[:,0], pos_ned[:,1], alt_ins, idx)):
            ln.set_data_3d(d[0], d[1], d[2])
        # Actualizar líneas de 3D GT
        for ln, d in zip(ax2_lines, draw_lines_truncated(gt['x'], gt['y'], alt_gt, idx)):
            ln.set_data_3d(d[0], d[1], d[2])

        dot_ins_3d._offsets3d = ([pos_ned[idx,0]], [pos_ned[idx,1]], [alt_ins[idx]])
        dot_gt_3d._offsets3d  = ([gt['x'][idx]], [gt['y'][idx]], [alt_gt[idx]])

        # Actualizar 2D INS
        for ln, d in zip(ax5_lines, draw_lines_truncated(pos_ned[:,1], pos_ned[:,0], idx_cut=idx)):
            ln.set_data(d[0], d[1])
        # Actualizar 2D GT
        for ln, d in zip(ax6_lines, draw_lines_truncated(gt['y'], gt['x'], idx_cut=idx)):
            ln.set_data(d[0], d[1])

        dot_ins_2d.set_offsets([pos_ned[idx,1], pos_ned[idx,0]])
        dot_gt_2d.set_offsets([gt['y'][idx], gt['x'][idx]])
        
        # Series de Tiempo
        line_phi.set_data(time[:idx+1], np.degrees(phi[:idx+1]))
        line_theta.set_data(time[:idx+1], np.degrees(theta[:idx+1]))
        line_psi.set_data(time[:idx+1], np.degrees(psi[:idx+1]))
        
        line_vn.set_data(time[:idx+1], vel_ned[:idx+1, 0])
        line_ve.set_data(time[:idx+1], vel_ned[:idx+1, 1])
        line_vd.set_data(time[:idx+1], vel_ned[:idx+1, 2])
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    is_playing = False
    anim_data = {'anim': None}

    def play_step(frame):
        if is_playing:
            current_val = slider.val
            step = (time[-1] - time[0]) / 300.0
            new_val = current_val + step
            if new_val >= time[-1]:
                new_val = time[0]
            slider.set_val(new_val)

    def toggle_play(event):
        nonlocal is_playing
        is_playing = not is_playing
        if is_playing:
            btn_play.color = '#8B2500'
            btn_play.hovercolor = '#C05A35'
            btn_play.label.set_text('⏸ Pause')
            anim_data['anim'] = animation.FuncAnimation(fig, play_step, interval=40, blit=False, cache_frame_data=False)
        else:
            btn_play.color = '#5C7A8C'
            btn_play.hovercolor = '#45A29E'
            btn_play.label.set_text('▶ Play')
            if anim_data['anim']:
                anim_data['anim'].event_source.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    update(time[0])
    plt.show()

def add_mission_phase_bands(ax, time_array):
    for phase in MISSION_PHASES:
        t0 = max(phase["t_start"], time_array[0])
        t1 = min(phase["t_end"], time_array[-1])
        if t0 < t1:
            ax.axvspan(t0, t1, alpha=phase["alpha"], color=phase["color"],
                      label=phase["name"])


def plot_quaternions(time, quats, rot_angles):
    """Cuaterniones y ángulo de rotación vs tiempo."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, facecolor=DARK_BG)
    fig.canvas.manager.set_window_title("AHRS - Cuaterniones")
    fig.suptitle("Representación en Cuaterniones", fontsize=14,
                 fontweight='bold', color=TXT_CLR)
    q_labels = ['q₀ (Escalar)', 'q₁ (Roll)', 'q₂ (Pitch)', 'q₃ (Yaw)']
    q_colors = ['#8E44AD', '#E74C3C', '#27AE60', '#2980B9']
    for i in range(4):
        _style_2d_ax(axes[i], '', q_labels[i])
        _add_phase_bands_dark(axes[i], time)
        axes[i].plot(time, quats[:, i], color=q_colors[i], linewidth=1.0,
                     label=q_labels[i])
        axes[i].legend(fontsize=8, loc='upper left', facecolor=DARK_AX,
                       edgecolor=GRID_CLR, labelcolor=TXT_CLR)
    _style_2d_ax(axes[4], 'Tiempo [s]', 'θ rot [°]')
    _add_phase_bands_dark(axes[4], time)
    axes[4].plot(time, np.degrees(rot_angles), color='#F39C12', linewidth=1.0,
                 label='θ Rotación')
    axes[4].legend(fontsize=8, loc='upper left', facecolor=DARK_AX,
                   edgecolor=GRID_CLR, labelcolor=TXT_CLR)
    plt.tight_layout()
    plt.show()


def plot_position_error(time, pos_ned, gt):
    """Error de posición INS vs GT  - muestra drift."""
    err_n = pos_ned[:, 0] - gt['x']
    err_e = pos_ned[:, 1] - gt['y']
    err_d = pos_ned[:, 2] - gt['z']
    err_total = np.sqrt(err_n**2 + err_e**2 + err_d**2)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, facecolor=DARK_BG)
    fig.canvas.manager.set_window_title("AHRS - Error de Posición (Drift)")
    fig.suptitle("Error de Posición — Drift Acumulativo", fontsize=14,
                 fontweight='bold', color='#e74c3c')
    components = [
        (err_n, 'Error North', '#E74C3C'), (err_e, 'Error East', '#27AE60'),
        (err_d, 'Error Down', '#2980B9'), (err_total, 'Error Total 3D', '#8E44AD'),
    ]
    for i, (ax, (err, label, color)) in enumerate(zip(axes, components)):
        _style_2d_ax(ax, '' if i < 3 else 'Tiempo [s]', f'{label} [m]')
        _add_phase_bands_dark(ax, time)
        ax.plot(time, err, color=color, linewidth=1.0, label=label)
        ax.legend(fontsize=8, loc='upper left', facecolor=DARK_AX,
                  edgecolor=GRID_CLR, labelcolor=TXT_CLR)
    plt.tight_layout()
    plt.show()


#   GRÁFICA INTERACTIVA 3D (Aeronave + Slider)

def set_axes_equal(ax):
    """Ajusta la escala de los ejes 3D para que no se deforme, preservando la inversión del eje Z si existe."""
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    
    # Check if Z axis is inverted originally
    z_inverted = z0 > z1
    
    mx = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)) / 2
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    
    ax.set_xlim3d(cx - mx, cx + mx)
    ax.set_ylim3d(cy - mx, cy + mx)
    if z_inverted:
        ax.set_zlim3d(cz + mx, cz - mx)
    else:
        ax.set_zlim3d(cz - mx, cz + mx)


def build_aircraft_polys():
    """ Construye la geometría detallada de la aeronave comercial. """
    L  = 2.4; R  = 0.18; ns = 1.0
    ws = 1.5; wc = 0.55; wt = 0.20; sw = 0.35
    hs = 0.55; hc = 0.30; vh = 0.55; vc = 0.45
    er = 0.08; el = 0.35
    
    x_nose = L / 2; x_tail = -L / 2
    x_wing = 0.05; x_htail = x_tail + 0.15; x_vtail = x_tail + 0.10
    
    polys = []
    
    # ── FUSELAJE (prisma octogonal) ──
    n_sec = 8
    angles_sec = np.linspace(0, 2 * np.pi, n_sec, endpoint=False)
    x_stations = [x_nose, x_nose - 0.25, 0.4, 0.0, -0.4, x_tail + 0.35, x_tail]
    r_stations = [0.0, R * 0.6, R, R, R, R * 0.75, R * 0.35]
    
    sections = []
    for xs, rs in zip(x_stations, r_stations):
        ring = np.array([[xs, rs * np.cos(a), rs * np.sin(a)] for a in angles_sec])
        sections.append(ring)
    
    for i in range(len(sections) - 1):
        s0, s1 = sections[i], sections[i + 1]
        for j in range(n_sec):
            j1 = (j + 1) % n_sec
            quad = np.array([s0[j], s0[j1], s1[j1], s1[j]])
            fc = '#A37F5A' if j < n_sec // 2 else '#8C6540' # Latón / Brass
            polys.append((quad, fc, '#4A3320'))
    
    tip = np.array([x_nose, 0, 0])
    s_first = sections[1]
    for j in range(n_sec):
        j1 = (j + 1) % n_sec
        tri = np.array([tip, s_first[j], s_first[j1]])
        polys.append((tri, '#A37F5A', '#4A3320'))
    
    # ── ALAS ──
    for sign in [1, -1]:
        root_le = np.array([x_wing, 0, 0])
        root_te = np.array([x_wing - wc, 0, 0])
        tip_le  = np.array([x_wing - sw, sign * ws, -sign * 0.04])
        tip_te  = np.array([x_wing - sw - wt, sign * ws, -sign * 0.04])
        polys.append((np.array([root_le, tip_le, tip_te, root_te]), '#8E402A', '#401B12')) # Cobre Oxidado
    
    # ── GÓNDOLAS DE MOTOR ──
    for sign in [1, -1]:
        eng_x = x_wing - sw * 0.45
        eng_y = sign * ws * 0.40
        eng_z = 0.12
        n_eng = 6
        a_eng = np.linspace(0, 2 * np.pi, n_eng, endpoint=False)
        front = np.array([[eng_x + el / 2, eng_y + er * np.cos(a), eng_z + er * np.sin(a)] for a in a_eng])
        back  = np.array([[eng_x - el / 2, eng_y + er * np.cos(a), eng_z + er * np.sin(a)] for a in a_eng])
        for j in range(n_eng):
            j1 = (j + 1) % n_eng
            polys.append((np.array([front[j], front[j1], back[j1], back[j]]), '#3B3833', '#1C1917')) # Gunmetal/Hierro
        polys.append((front, '#2C2825', '#1C1917'))
        polys.append((back, '#4A4641', '#1C1917'))
    
    # Pilones
    for sign in [1, -1]:
        eng_x = x_wing - sw * 0.45
        eng_y = sign * ws * 0.40
        pylon = np.array([
            [eng_x + el * 0.3, eng_y, 0],
            [eng_x - el * 0.3, eng_y, 0],
            [eng_x - el * 0.3, eng_y, 0.12 - er],
            [eng_x + el * 0.3, eng_y, 0.12 - er],
        ])
        polys.append((pylon, '#6E5A47', '#4A3320'))
    
    # ── ESTABILIZADOR HORIZONTAL ──
    for sign in [1, -1]:
        polys.append((np.array([
            [x_htail, 0, 0], [x_htail - 0.12, sign * hs, 0],
            [x_htail - hc + 0.05, sign * hs, 0], [x_htail - hc, 0, 0]
        ]), '#8E402A', '#401B12')) # Cobre
    
    # ── ALETA VERTICAL ──
    polys.append((np.array([
        [x_vtail, 0, 0], [x_vtail - 0.12, 0, -vh],
        [x_vtail - vc, 0, -vh * 0.85], [x_vtail - vc - 0.05, 0, 0]
    ]), '#A37F5A', '#4A3320')) # Latón
    
    polys.append((np.array([
        [x_vtail - vc + 0.05, 0, -vh * 0.85], [x_vtail - vc - 0.05, 0, -vh * 0.85],
        [x_vtail - vc - 0.08, 0, -vh * 0.15], [x_vtail - vc + 0.02, 0, -vh * 0.15]
    ]), '#712F21', '#401B12')) # Óxido oscuro
    
    # Franjas decorativas
    stripe_pts = np.array([
        [0.5, R * 0.97, -R * 0.15], [-0.7, R * 0.97, -R * 0.15],
        [-0.7, R * 0.97, R * 0.05], [0.5, R * 0.97, R * 0.05]
    ])
    polys.append((stripe_pts, '#B28238', '#B28238')) # Oro envejecido/Gold
    stripe_l = stripe_pts.copy(); stripe_l[:, 1] *= -1
    polys.append((stripe_l, '#B28238', '#B28238'))
    
    return polys


def plot_interactive_3d(time, phi, theta, psi, vel_ned, quats, rot_angles, rot_axes):
    """
    Gráfica 6: Visualización interactiva 3D con slider de tiempo.
    Muestra la aeronave orientada según los ángulos de Euler en cada instante.
    """
    aircraft_polys = build_aircraft_polys()
    
    fig = plt.figure(figsize=(16, 9), constrained_layout=False, facecolor=DARK_BG)
    fig.canvas.manager.set_window_title("AHRS - Visualización Interactiva 3D")
    
    # Layout: aeronave 3D a la izquierda, paneles de info a la derecha
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1.2],
                          left=0.05, right=0.95, top=0.92, bottom=0.12,
                          wspace=0.15, hspace=0.25)
    
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_angles = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])
    
    ax_slider = fig.add_axes([0.1, 0.02, 0.65, 0.03], facecolor=DARK_AX)
    slider = Slider(ax_slider, 'Tiempo [s]', time[0], time[-1],
                    valinit=time[0], valstep=time[1]-time[0],
                    color='#C05A35')
    slider.label.set_color(TXT_CLR)
    slider.valtext.set_color(TXT_CLR)
    
    def get_mission_phase(t):
        for phase in MISSION_PHASES:
            if phase["t_start"] <= t < phase["t_end"]:
                return phase["name"]
        return "Fuera de misión"
    
    def update(val):
        t_current = slider.val
        idx = np.argmin(np.abs(time - t_current))
        
        # --- Limpiar y redibujar 3D ---
        ax3d.cla()
        ax3d.set_facecolor(DARK_BG)
        ax3d.xaxis.set_pane_color((0,0,0,0))
        ax3d.yaxis.set_pane_color((0,0,0,0))
        ax3d.zaxis.set_pane_color((0,0,0,0))
        ax3d.grid(color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.5)
        ax3d.xaxis.label.set_color(TXT_CLR); ax3d.yaxis.label.set_color(TXT_CLR); ax3d.zaxis.label.set_color(TXT_CLR)
        ax3d.tick_params(colors=TXT_CLR)
        
        ax3d.set_title(f"Actitud de la Aeronave — t = {time[idx]:.2f} s\n"
                       f"Fase: {get_mission_phase(time[idx])}",
                       fontsize=12, fontweight='bold', color=TXT_CLR)
        ax3d.invert_zaxis()  # Convención NED
        
        # DCM para este instante
        dcm = body_to_ned_dcm(phi[idx], theta[idx], psi[idx])
        
        # Dibujar aeronave rotada
        for verts, fc, ec in aircraft_polys:
            pts = (dcm @ verts.T).T
            poly = Poly3DCollection([pts], facecolor=fc, edgecolor=ec,
                                    alpha=0.85, linewidths=0.5)
            ax3d.add_collection3d(poly)
        
        # Marco NED estático
        for vec, col, lab in zip(
            [np.array([1.5,0,0]), np.array([0,1.5,0]), np.array([0,0,1.5])],
            ['gray', 'gray', 'gray'],
            ['N', 'E', 'D']
        ):
            ax3d.quiver(0,0,0, *vec, color=col, linestyle=':', arrow_length_ratio=0.1, alpha=0.4)
            ax3d.text(*vec*1.1, lab, color='gray', fontsize=8)
        
        # Ejes Body rotados
        for vec, col, lab in zip(np.eye(3), ['#E74C3C','#27AE60','#2980B9'], ['Xb','Yb','Zb']):
            v = dcm @ vec
            ax3d.quiver(0,0,0, *v, length=1.2, color=col, linewidth=2.5, arrow_length_ratio=0.15)
            ax3d.text(*(v*1.3), lab, color=col, fontsize=10, fontweight='bold')
        
        ax3d.set_xlabel('North'); ax3d.set_ylabel('East'); ax3d.set_zlabel('Down')
        ax3d.auto_scale_xyz([-1.5,1.5],[-1.5,1.5],[-1.5,1.5])
        set_axes_equal(ax3d)
        
        # --- Panel de ángulos (tarjetas de color) ---
        ax_angles.cla()
        ax_angles.axis('off')
        ax_angles.set_xlim(0, 1); ax_angles.set_ylim(0, 1)
        ax_angles.set_title("Ángulos Aerodinámicos", fontweight='bold', fontsize=12, color=TXT_CLR)
        
        phi_deg   = np.degrees(phi[idx])
        theta_deg = np.degrees(theta[idx])
        psi_deg   = np.degrees(psi[idx])
        
        cards = [
            ('φ  Roll',  phi_deg,   '#C05A35'), # Óxido de cobre
            ('θ  Pitch', theta_deg, '#5C7A8C'), # Acero
            ('ψ  Yaw',   psi_deg,   '#D4AF37'), # Latón
        ]
        
        card_h = 0.28; gap = 0.04; y_start = 0.92
        for i, (label, val, color) in enumerate(cards):
            y = y_start - i * (card_h + gap)
            fancy = FancyBboxPatch((0.05, y - card_h), 0.9, card_h,
                                   transform=ax_angles.transAxes, clip_on=False,
                                   facecolor=color, edgecolor='white',
                                   linewidth=2, alpha=0.92,
                                   boxstyle='round,pad=0.02')
            ax_angles.add_patch(fancy)
            ax_angles.text(0.5, y - card_h * 0.30, label,
                           transform=ax_angles.transAxes, ha='center', va='center',
                           fontsize=10, color='white', fontweight='bold')
            ax_angles.text(0.5, y - card_h * 0.70, f"{val:.2f}°",
                           transform=ax_angles.transAxes, ha='center', va='center',
                           fontsize=18, color='white', fontweight='bold', family='monospace')
        
        # --- Panel de texto (instrumentos) ---
        ax_text.cla()
        ax_text.axis('off')
        
        q = quats[idx]
        vn, ve, vd = vel_ned[idx]
        ra = np.degrees(rot_angles[idx])
        re = rot_axes[idx]
        
        info = (
            f"--- FLIGHT INSTRUMENTS ---\n\n"
            f"ATTITUDE (Euler):\n"
            f"  Roll (φ):  {phi_deg:8.3f}°\n"
            f"  Pitch (θ): {theta_deg:8.3f}°\n"
            f"  Yaw (ψ):   {psi_deg:8.3f}°\n\n"
            f"QUATERNION:\n"
            f"  q = [{q[0]:.4f}, {q[1]:.4f},\n"
            f"       {q[2]:.4f}, {q[3]:.4f}]\n"
            f"  θ_rot = {ra:.2f}°\n"
            f"  ê = [{re[0]:.3f}, {re[1]:.3f}, {re[2]:.3f}]\n\n"
            f"VELOCITY NED [m/s]:\n"
            f"  V_N: {vn:8.3f}\n"
            f"  V_E: {ve:8.3f}\n"
            f"  V_D: {vd:8.3f}\n"
            f"  |V|: {np.linalg.norm(vel_ned[idx]):8.3f}"
        )
        
        props = dict(boxstyle='round', facecolor=DARK_AX, alpha=0.9, edgecolor=GRID_CLR)
        ax_text.text(0.05, 0.95, info, transform=ax_text.transAxes, fontsize=9,
                     color=TXT_CLR, verticalalignment='top', bbox=props, family='monospace')
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(time[0])  # Dibujar estado inicial

    # --- Autoplayer ---
    ax_play = fig.add_axes([0.80, 0.015, 0.08, 0.04])
    btn_play = Button(ax_play, '▶ Play', color='#5C7A8C', hovercolor='#45A29E')
    btn_play.label.set_fontweight('bold'); btn_play.label.set_color(DARK_BG)

    is_playing = False
    anim_data = {'anim': None}

    def play_step(frame):
        if is_playing:
            current_val = slider.val
            step = (time[-1] - time[0]) / 300.0
            new_val = current_val + step
            if new_val >= time[-1]:
                new_val = time[0]
            slider.set_val(new_val)

    def toggle_play(event):
        nonlocal is_playing
        is_playing = not is_playing
        if is_playing:
            btn_play.color = '#8B2500'
            btn_play.hovercolor = '#C05A35'
            btn_play.label.set_text('⏸ Pause')
            anim_data['anim'] = animation.FuncAnimation(fig, play_step, interval=40, blit=False, cache_frame_data=False)
        else:
            btn_play.color = '#5C7A8C'
            btn_play.hovercolor = '#45A29E'
            btn_play.label.set_text('▶ Play')
            if anim_data['anim']:
                anim_data['anim'].event_source.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    
    plt.show()

#   MÓDULO 6: PROCESAMIENTO COMPLETO Y MENÚ PRINCIPAL

def run_full_processing(imu_path, gt_path):
    print("\n" + "="*60)
    print("  PROCESANDO DATOS DEL IMU...")
    print("="*60)
    
    # 1. Cargar datos
    print("\n  [1/4] Cargando datos IMU y Ground Truth...")
    imu = load_imu_data(imu_path)
    gt  = load_ground_truth(gt_path)
    print(f"        → {len(imu['time'])} muestras IMU cargadas")
    print(f"        → {len(gt['time'])} muestras GT cargadas")
    print(f"        → Duración: {imu['time'][-1]:.1f} s, dt = {imu['time'][1]-imu['time'][0]:.3f} s")
    
    # 2. Propagar actitud
    print("\n  [2/4] Propagando actitud (Cinemática de Euler)...")
    phi, theta, psi = propagate_attitude(
        imu['time'], imu['p'], imu['q'], imu['r']
    )
    print(f"        → φ final: {np.degrees(phi[-1]):.2f}°")
    print(f"        → θ final: {np.degrees(theta[-1]):.2f}°")
    print(f"        → ψ final: {np.degrees(psi[-1]):.2f}°")
    
    # 3. Calcular navegación
    print("\n  [3/4] Calculando navegación (aceleración → velocidad → posición)...")
    vel_ned, pos_ned = propagate_navigation(
        imu['time'], imu['ax'], imu['ay'], imu['az'],
        phi, theta, psi
    )
    print(f"        → Posición final NED: [{pos_ned[-1,0]:.2f}, {pos_ned[-1,1]:.2f}, {pos_ned[-1,2]:.2f}] m")
    print(f"        → Pos. GT final:      [{gt['x'][-1]:.2f}, {gt['y'][-1]:.2f}, {gt['z'][-1]:.2f}] m")
    err_final = np.linalg.norm(pos_ned[-1] - np.array([gt['x'][-1], gt['y'][-1], gt['z'][-1]]))
    print(f"        → Error final 3D:    {err_final:.2f} m")
    
    # 4. Cuaterniones
    print("\n  [4/4] Calculando cuaterniones...")
    quats, rot_angles, rot_axes = compute_all_quaternions(phi, theta, psi)
    print(f"        → q final: [{quats[-1,0]:.4f}, {quats[-1,1]:.4f}, {quats[-1,2]:.4f}, {quats[-1,3]:.4f}]")
    print(f"        → θ_rot final: {np.degrees(rot_angles[-1]):.2f}°")
    
    print("\n" + "="*60)
    print("  ✓ PROCESAMIENTO COMPLETO")
    print("="*60)
    
    return {
        'imu': imu,
        'gt': gt,
        'time': imu['time'],
        'phi': phi,
        'theta': theta,
        'psi': psi,
        'vel_ned': vel_ned,
        'pos_ned': pos_ned,
        'quats': quats,
        'rot_angles': rot_angles,
        'rot_axes': rot_axes,
    }


def print_menu():
    """Imprime el menú principal interactivo."""
    print("\n" + "─"*50)
    print("  MENÚ PRINCIPAL - AHRS Task 2")
    print("─"*50)
    print("  [1] 📊 Dashboard Completo (tema oscuro)")
    print("  [2] ✈️  Visualización Interactiva 3D (Aeronave)")
    print("  [3] 📈 Cuaterniones vs Tiempo")
    print("  [4] 📉 Error de Posición (Drift)")
    print("  [0] Salir")
    print("─"*50)


def main():
    """Función principal con menú interactivo."""
    
    print("\n" + "*"*55)
    print("  ATTITUDE & NAVIGATION REFERENCE SYSTEM (AHRS)  ".center(55, '*'))
    print("  Task 2 - Flight Dynamics                       ".center(55, '*'))
    print("*"*55)
    
    # Rutas de archivos (relativas al directorio del script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    imu_path = os.path.join(script_dir, 'tello_imu_example.csv')
    gt_path  = os.path.join(script_dir, 'tello_ground_truth.csv')
    
    # Verificar existencia de archivos
    for p, name in [(imu_path, 'IMU'), (gt_path, 'Ground Truth')]:
        if not os.path.exists(p):
            print(f"\n  [ERROR] No se encontró el archivo {name}: {p}")
            sys.exit(1)
    
    results = None  # Se llena al ejecutar el procesamiento
    
    while True:
        print_menu()
        choice = input("\n  Opción: ").strip()
        
        if choice == '0':
            print("\n  Cerrando AHRS... ¡Hasta luego!")
            break
        
        elif choice in ('1', '2', '3', '4'):
            if results is None:
                results = run_full_processing(imu_path, gt_path)
            
            r = results
            
            if choice == '1':
                print("\n  Abriendo Dashboard completo...")
                plot_dashboard(r)
            
            elif choice == '2':
                print("\n  Abriendo visualización interactiva 3D...")
                print("  Use el slider para navegar por el tiempo.")
                plot_interactive_3d(r['time'], r['phi'], r['theta'], r['psi'],
                                    r['vel_ned'], r['quats'], r['rot_angles'], r['rot_axes'])
            
            elif choice == '3':
                print("\n  Generando gráfica de cuaterniones...")
                plot_quaternions(r['time'], r['quats'], r['rot_angles'])
            
            elif choice == '4':
                print("\n  Generando gráfica de error de posición...")
                plot_position_error(r['time'], r['pos_ned'], r['gt'])
        
        else:
            print("\n  [ATENCIÓN] Opción inválida. Intente de nuevo.")


if __name__ == "__main__":
    main()
