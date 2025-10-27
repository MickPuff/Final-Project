import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

# assigning constants

g = 9.81
ell = 1

# initial conditions:

theta0 = np.deg2rad(170) # angle
theta_dot0 = 0 # angular velocity
mu = 0
F_0 = 0
omega = 3.1305

#defining the system of diffeq.
#y[0] is theta, y[1] is theta_dot

def pendulum_ODE(t, y):
    return (y[1], -mu*y[1] + (-g*np.sin(y[0])/ell) + F_0*np.sin(omega*t))

#solving ODE

time_range = 20
fps = 60
sol = solve_ivp(
    pendulum_ODE, [0, time_range], (theta0, theta_dot0),
    t_eval=np.linspace(0, time_range, fps*time_range),
    rtol=1e-10, atol=1e-12  # much tighter and capped step
)

#output:
theta = sol.y[0]
theta_dot = sol.y[1]
t = sol.t

#rad -> deg
theta_deg = np.rad2deg(theta)
theta_dot_deg = np.rad2deg(theta_dot)

# np.savetxt('pend.csv', np.transpose([theta_deg, theta_dot_deg]), delimiter=',')

#plotting angle and angular velocity vs time

PLOT_GRAPH = False
if PLOT_GRAPH:
    plt.figure(figsize=(8, 8))  # Make it square for better phase diagram visibility
    plt.plot(t, theta_deg, 'r', lw= 2, label=r'$\theta$')
    # plt.plot(t, theta_dot_deg, 'b', lw= 2, label=r'$\dot \theta$')
    plt.title('Simple Pendulum')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel(r'$\theta$ (deg) $\dot \theta$ (deg/s)')
    plt.grid()
    plt.show()

#animating angle and angular velocity vs time
PLAY_ANI_GRAPH = True
if PLAY_ANI_GRAPH:
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    # initialize empty line objects (animator will fill data)
    theta_curve, = ax1.plot([], [], 'r')
    theta_dot_curve, = ax1.plot([], [], 'b')

    ax1.set_title('Simpel Pendulum: Angular position, velocity vs time')
    ax1.set_xlim(0, time_range)

    # compute y-limits from the data with padding so both traces fit nicely
    y_min = min(np.min(theta_deg), np.min(theta_dot_deg))
    y_max = max(np.max(theta_deg), np.max(theta_dot_deg))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax1.set_ylim(y_min - pad, y_max + pad)

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(r'$\theta$ (deg) $\dot \theta$ (deg/s)')
    ax1.legend([r'$\theta$', r'$\dot \theta$'])
    ax1.grid()

    def init_graph():
        theta_curve.set_data([], [])
        theta_dot_curve.set_data([], [])
        return theta_curve, theta_dot_curve

    def animate_graph(i):
        a = 0
        TRAIL = False
        if TRAIL:
            trailing_frames = 60
            if i >= trailing_frames:
                a = i + 1 - trailing_frames
        theta_curve.set_data(t[a:i+1], theta_deg[a:i+1])  # trajectory up to current frame
        theta_dot_curve.set_data(t[a:i+1], theta_dot_deg[a:i+1])
        return theta_curve, theta_dot_curve


    # ani = animation.FuncAnimation(fig, animate, frames=len(t))
    # ffmpeg_writer = animation.FFMpegWriter(fps=30)
    # ani.save('time_domain.mp4', writer=ffmpeg_writer)

    ani1 = animation.FuncAnimation(
        fig1,
        animate_graph,
        frames=len(t),
        init_func=init_graph,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )
    plt.show()

#Phase diagram

PLOT_DIAGRAM = False
if PLOT_DIAGRAM:
    # Calculate good bounds with 5% padding
    theta_min, theta_max = np.min(theta_deg), np.max(theta_deg)
    theta_dot_min, theta_dot_max = np.min(theta_dot_deg), np.max(theta_dot_deg)
    
    # Add 5% padding to make the plot look better
    padding_x = 0.05 * (theta_max - theta_min)
    padding_y = 0.05 * (theta_dot_max - theta_dot_min)
    
    plt.figure(figsize=(8, 8))  # Make it square for better phase diagram visibility
    plt.plot(theta_deg, theta_dot_deg, 'b', label='Phase trajectory')
    plt.xlim(theta_min - padding_x, theta_max + padding_x)
    plt.ylim(theta_dot_min - padding_y, theta_dot_max + padding_y)
    plt.title('Simple Pendulum: Phase Diagram')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'$\dot \theta$ (deg/s)')
    plt.grid()
    plt.legend()
    plt.show()

#animating phase diagram
PLAY_ANI_DIAGRAM = True
if PLAY_ANI_DIAGRAM:
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    # initialize empty line and a marker for the current point
    phase_curve, = ax2.plot([], [], 'b')        # trajectory (line)
    phase_dot, = ax2.plot([], [], 'ro')        # current position (red dot)

    ax2.set_title('Simple Pendulum: Phase Diagram')

    # compute bounds from data with padding so everything is visible
    theta_min, theta_max = np.min(theta_deg), np.max(theta_deg)
    theta_dot_min, theta_dot_max = np.min(theta_dot_deg), np.max(theta_dot_deg)
    pad_x = 0.05 * (theta_max - theta_min) if theta_max > theta_min else 1.0
    pad_y = 0.05 * (theta_dot_max - theta_dot_min) if theta_dot_max > theta_dot_min else 1.0
    ax2.set_xlim(theta_min - pad_x, theta_max + pad_x)
    ax2.set_ylim(theta_dot_min - pad_y, theta_dot_max + pad_y)

    ax2.set_xlabel(r'$\theta$ (deg)')
    ax2.set_ylabel(r'$\dot \theta$ (deg/s)')
    ax2.grid()

    # init function: clear data (needed if using blit=True)
    def init_diagram():
        phase_curve.set_data([], [])
        phase_dot.set_data([], [])
        return phase_curve, phase_dot

    def animate_diagram(i):
        a = 0
        TRAIL = True
        if TRAIL:
            trailing_frames = 120
            if i >= trailing_frames:
                a = i + 1 - trailing_frames

        # set trajectory up to current frame and a marker at current point
        phase_curve.set_data(theta_deg[a:i+1], theta_dot_deg[a:i+1])
        phase_dot.set_data([theta_deg[i]], [theta_dot_deg[i]])
        return phase_curve, phase_dot

    ani2 = animation.FuncAnimation(
        fig2,
        animate_diagram,
        frames=len(t),
        init_func=init_diagram,
        interval=1000/fps,
        blit=True,      # set blit=False if your backend has issues
        repeat=False
    )

    plt.show()

#animating the pendulum
PLAY_ANI_PEND = True
if PLAY_ANI_PEND:
    def pend_pos(theta):
        return (ell*np.sin(theta), -ell*np.cos(theta))

    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(aspect='equal')
    ax3.set_xlim(-1.25, 1.25)
    ax3.set_ylim(-1.25, 1.25)
    ax3.grid()

    x0, y0 = pend_pos(theta0)
    line, = ax3.plot([0, x0], [0,y0], lw=2, c='k')
    circle = ax3.add_patch(plt.Circle(pend_pos(theta0), 0.05, fc='r', zorder=3))

    def animate_pend(i):
        x,y = pend_pos(theta[i])
        line.set_data([0,x], [0,y])
        circle.set_center((x,y))

    ani3 = animation.FuncAnimation(
        fig3,
        animate_pend,
        frames=len(t),
        interval=1000/fps,
        blit=False,      # set blit=False if your backend has issues
        repeat=False
    )

    plt.show()

ANIMATE_ALL = False
if ANIMATE_ALL:
    
    fig = plt.figure(figsize=(8, 8))
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

    theta_curve_1, = ax1.plot(t[0], theta_deg[0], 'b')
    theta_dot_curve_1, = ax1.plot(t[0], theta_dot_deg[0], 'r')

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

    phase_curve_2, = ax2.plot(theta_deg[0], theta_dot_deg[0], 'b')  
    phase_dot, = ax2.plot(theta_deg[0], theta_dot_deg[0], 'ro') 

    #pendulum
    def pend_pos(theta):
        return (ell*np.sin(theta), -ell*np.cos(theta))

    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(aspect='equal')
    ax3.set_xlim(-1.25, 1.25)
    ax3.set_ylim(-1.25, 1.25)
    ax3.grid()

    x0, y0 = pend_pos(theta0)
    line, = ax3.plot([0, x0], [0,y0], lw=2, c='k')
    circle = ax3.add_patch(plt.Circle(pend_pos(theta0), 0.05, fc='r', zorder=3))