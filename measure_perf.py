import time
import numpy as np
from scipy.integrate import solve_ivp

# copy constants from oscillator.py
g = 9.81
ell = 1
theta0 = np.deg2rad(170)
theta_dot0 = 0
mu = 0.2
F_0 = 2
omega = 3.1305

# ODE
def pendulum_ODE(t, y):
    return (y[1], -mu*y[1] + (-g*np.sin(y[0])/ell) + F_0*np.sin(omega*t))

# solve
time_range = 40
fps = 30
start = time.perf_counter()
sol = solve_ivp(
    pendulum_ODE, [0, time_range], (theta0, theta_dot0),
    t_eval=np.linspace(0, time_range, fps*time_range),
    rtol=1e-10, atol=1e-12
)
solve_time = time.perf_counter() - start

theta = sol.y[0]
N = len(theta)

# precompute trig
start = time.perf_counter()
x_positions = ell * np.sin(theta)
y_positions = -ell * np.cos(theta)
precomp_time = time.perf_counter() - start

# per-frame trig in Python loop
start = time.perf_counter()
x_loop = np.empty(N)
y_loop = np.empty(N)
for i in range(N):
    x_loop[i] = ell * np.sin(theta[i])
    y_loop[i] = -ell * np.cos(theta[i])
loop_time = time.perf_counter() - start

print(f"solve_ivp time: {solve_time:.4f} s")
print(f"precompute trig time (vectorized): {precomp_time:.6f} s")
print(f"per-frame trig time (Python loop): {loop_time:.6f} s")
print(f"vectorized is {loop_time/precomp_time:.2f}x faster than Python loop")
