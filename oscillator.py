import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# assigning constants

g = 9.81
ell = 1

# initial conditions:

theta0 = np.deg2rad(90) # initial angle
theta_dot0 = 0 # angular velocity
mu = 0.5 #damping coefficient
F_0 = 10 # driving force
omega = 3.1305 * 2 # driving force frequency

#defining the system of diffeq.
#y[0] is theta, y[1] is theta_dot

def pendulum_ODE(t, y):
    return (y[1], -mu*y[1] + (-g*np.sin(y[0])/ell) + F_0*np.sin(omega*t))

#solving ODE

time_range = 10
fps = 60
trailing_sec = 5
sol = solve_ivp(
    pendulum_ODE, [0, time_range], (theta0, theta_dot0),
    t_eval=np.linspace(0, time_range, fps*time_range),
    rtol=1e-10, atol=1e-12  #   
)

#output:
theta = sol.y[0]
theta_dot = sol.y[1]
t = sol.t

#rad -> deg
theta_deg = np.rad2deg(theta)
theta_dot_deg = np.rad2deg(theta_dot)

def pendulum_energy(theta, theta_dot, m=1.0, g=g, ell=ell):
    # linear speed of the bob
    v = ell * theta_dot

    # kinetic energy: 1/2 m v^2
    KE = 0.5 * m * v**2

    # potential energy: m g h, with h = ell(1 - cos(theta))
    PE = m * g * ell * (1 - np.cos(theta))

    # total mechanical energy
    E = KE + PE

    return KE, PE, E

KE, PE, E = pendulum_energy(theta, theta_dot)
print(f"Max total energy: {np.max(E):.4f}")
print(f"Min total energy: {np.min(E):.4f}")

ANIMATE_ALL = True
if ANIMATE_ALL:
    
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2,2, width_ratios=[1,2], height_ratios=[1,1])

    #graph
    ax1 = fig.add_subplot(gs[0,0])

    ax1.set_title('Simpel Pendulum: Angular position, velocity vs time')
    ax1.set_xlim(0, time_range)

    y_min = min(np.min(theta_deg), np.min(theta_dot_deg))
    y_max = max(np.max(theta_deg), np.max(theta_dot_deg))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax1.set_ylim(y_min - pad, y_max + pad)

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(r'$\theta$ (deg) $\dot \theta$ (deg/s)')
    ax1.legend([r'$\theta$', r'$\dot \theta$'])
    ax1.grid()

    # mark as animated for blitting
    theta_curve_1, = ax1.plot(t[0], theta_deg[0], 'b', animated=True)
    theta_dot_curve_1, = ax1.plot(t[0], theta_dot_deg[0], 'r', animated=True)

    #phase diagram
    ax2 = fig.add_subplot(gs[1,0])     

    ax2.set_title('Simple Pendulum: Phase Diagram')

    theta_min, theta_max = np.min(theta_deg), np.max(theta_deg)
    theta_dot_min, theta_dot_max = np.min(theta_dot_deg), np.max(theta_dot_deg)
    pad_x = 0.05 * (theta_max - theta_min) if theta_max > theta_min else 1.0
    pad_y = 0.05 * (theta_dot_max - theta_dot_min) if theta_dot_max > theta_dot_min else 1.0
    ax2.set_xlim(theta_min - pad_x, theta_max + pad_x)
    ax2.set_ylim(theta_dot_min - pad_y, theta_dot_max + pad_y)

    ax2.set_xlabel(r'$\theta$ (deg)')
    ax2.set_ylabel(r'$\dot \theta$ (deg/s)')
    ax2.grid()

    # phase lines (animated)
    phase_curve_2, = ax2.plot(theta_deg[0], theta_dot_deg[0], 'b', animated=True)  
    phase_dot_2, = ax2.plot(theta_deg[0], theta_dot_deg[0], 'ro', animated=True) 

    #pendulum
    def pend_pos(theta):
        return (ell*np.sin(theta), -ell*np.cos(theta))

    ax3 = fig.add_subplot(gs[:,1])

    ax3.set_xlim(-1.25, 1.25)
    ax3.set_ylim(-1.25, 1.25)
    ax3.grid()

    # precompute positions to avoid repeated trig calls inside animate
    x_positions = ell * np.sin(theta)
    y_positions = -ell * np.cos(theta)

    x0, y0 = x_positions[0], y_positions[0]
    # make line and patch animated for blitting
    line_3, = ax3.plot([0, x0], [0, y0], lw=2, c='k', animated=True)
    circle_3 = mpatches.Circle((x0, y0), 0.05, fc='r', zorder=3, animated=True)
    ax3.add_patch(circle_3)

    def init_anim():
        # set initial data for blitting
        theta_curve_1.set_data([], [])
        theta_dot_curve_1.set_data([], [])
        phase_curve_2.set_data([], [])
        phase_dot_2.set_data([], [])
        line_3.set_data([0, x_positions[0]], [0, y_positions[0]])
        circle_3.set_center((x_positions[0], y_positions[0]))
        return theta_curve_1, theta_dot_curve_1, phase_curve_2, phase_dot_2, line_3, circle_3

    def animate_all(i):
        # use local refs for speed
        t_local = t
        theta_d = theta_deg
        theta_dot_d = theta_dot_deg
        x_pos = x_positions
        y_pos = y_positions

        a = 0
        TRAIL = True
        if TRAIL:
            trailing_frames = trailing_sec * fps
            if i >= trailing_frames:
                a = i + 1 - trailing_frames

        # update curves (slicing cost is small compared with redraw cost)
        theta_curve_1.set_data(t_local[:i+1], theta_d[:i+1])
        theta_dot_curve_1.set_data(t_local[:i+1], theta_dot_d[:i+1])

        phase_curve_2.set_data(theta_d[a:i+1], theta_dot_d[a:i+1])
        phase_dot_2.set_data([theta_d[i]], [theta_dot_d[i]])

        # pendulum
        x = x_pos[i]
        y = y_pos[i]
        line_3.set_data([0, x], [0, y])
        circle_3.set_center((x, y))

        return theta_curve_1, theta_dot_curve_1, phase_curve_2, phase_dot_2, line_3, circle_3

    ani_all = animation.FuncAnimation(
            fig,
            animate_all,
            frames=len(t),
            interval=1000/fps,
            blit=True,      # use blitting to speed up redraws
            init_func=init_anim,
            repeat=False
        )
    
    plt.show()