# app.py
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Mass–Spring Animation", layout="wide")

# -------------------------
# Session state (for animation)
# -------------------------
if "play" not in st.session_state:
    st.session_state.play = False
if "t" not in st.session_state:
    st.session_state.t = 0.0
if "x" not in st.session_state:
    st.session_state.x = 1.0  # default initial displacement
if "v" not in st.session_state:
    st.session_state.v = 0.0

# -------------------------
# UI
# -------------------------
st.title("Mass–Spring–Damper (Animated)")

colA, colB, colC = st.columns(3)
with colA:
    m = st.slider("Mass m (kg)", 0.1, 10.0, 1.0, 0.1)
    k = st.slider("Spring k (N/m)", 0.1, 200.0, 20.0, 0.1)
    c = st.slider("Damping c (kg/s)", 0.0, 10.0, 0.5, 0.1)
with colB:
    x0 = st.slider("Initial displacement x₀ (m)", -3.0, 3.0, 1.0, 0.1)
    v0 = st.slider("Initial velocity v₀ (m/s)", -5.0, 5.0, 0.0, 0.1)
    dt = st.select_slider("Time step Δt (s)", [0.002, 0.005, 0.01, 0.02], value=0.01)
with colC:
    L0 = st.slider("Rest spring length L₀ (m)", 0.5, 3.0, 1.5, 0.1)
    n_coils = st.slider("Spring coils (visual)", 3, 12, 6, 1)
    T_vis = st.slider("Visible width (m)", 2.5, 8.0, 5.0, 0.5)

omega0 = np.sqrt(k/m)
zeta = c/(2*np.sqrt(k*m))
st.markdown(f"**Natural freq:** ω₀ = {omega0:.3f} rad/s &nbsp;&nbsp;|&nbsp;&nbsp; **Damping ratio:** ζ = {zeta:.3f}")

# Controls
btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
if btn_col1.button("▶️ Play / ⏸ Pause"):
    st.session_state.play = not st.session_state.play

def _reset():
    st.session_state.play = False
    st.session_state.t = 0.0
    st.session_state.x = x0
    st.session_state.v = v0

if btn_col2.button("⟲ Reset"):
    _reset()

# Make sure initial conditions match sliders on first load
if st.session_state.t == 0.0 and st.session_state.x == 1.0 and x0 != 1.0:
    _reset()

place_anim = st.empty()
place_readout = st.empty()

# -------------------------
# Physics (semi-implicit Euler)
#   m x'' + c x' + k x = 0
# -------------------------
def step():
    a = -(c/m)*st.session_state.v - (k/m)*st.session_state.x
    st.session_state.v += dt * a
    st.session_state.x += dt * st.session_state.v
    st.session_state.t += dt

# -------------------------
# Drawing
# -------------------------
def draw_scene(x):
    """
    Draw a wall at x=0, a spring to a block.
    The block center sits at X = L0 + x.
    """
    X_block_center = L0 + x
    block_w, block_h = 0.6, 0.6
    block_left = X_block_center - block_w/2
    block_right = X_block_center + block_w/2
    y0 = 0.0  # vertical center line

    # Build spring polyline from wall (x=0) to block_left
    Xs = [0.0]
    Ys = [y0]
    length = max(block_left, 0.05)  # ensure positive
    # straight lead-in
    lead = min(0.2, 0.3*length)
    Xs.append(lead); Ys.append(y0)

    # coils
    usable = max(length - lead - 0.1, 0.05)
    xs = np.linspace(lead, lead + usable, n_coils*2+1)
    for i, xi in enumerate(xs):
        # zigzag up/down around center line
        amp = 0.15
        yi = y0 + (amp if i % 2 else -amp)
        Xs.append(xi); Ys.append(yi)
    # straight into the block
    Xs.append(block_left); Ys.append(y0)

    fig, ax = plt.subplots(figsize=(8, 2.4))
    # wall
    ax.plot([0, 0], [-1, 1], linewidth=4)
    # spring
    ax.plot(Xs, Ys, linewidth=2)
    # block
    rect = plt.Rectangle((block_left, y0 - block_h/2), block_w, block_h, fill=True, alpha=0.6)
    ax.add_patch(rect)

    # bounds
    ax.set_xlim(-0.3, T_vis)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Mass–Spring–Damper Animation")
    fig.tight_layout()
    return fig

# -------------------------
# One-run render (and animate if playing)
# -------------------------
frames_per_run = 1200  # cap work per rerun so the UI stays responsive
if st.session_state.play:
    start = time.perf_counter()
    for _ in range(frames_per_run):
        step()
        fig = draw_scene(st.session_state.x)
        place_anim.pyplot(fig)
        place_readout.write(
            f"t = **{st.session_state.t:.2f} s**, x = **{st.session_state.x:.3f} m**, v = **{st.session_state.v:.3f} m/s**"
        )
        # ~60 FPS feeling without burning CPU
        time.sleep(1/60)
        # Safety timeout per rerun (prevents infinite block if system is slow)
        if time.perf_counter() - start > 3.0:
            break
else:
    # Static render when paused
    fig = draw_scene(st.session_state.x)
    place_anim.pyplot(fig)
    place_readout.write(
        f"t = **{st.session_state.t:.2f} s**, x = **{st.session_state.x:.3f} m**, v = **{st.session_state.v:.3f} m/s**"
    )
