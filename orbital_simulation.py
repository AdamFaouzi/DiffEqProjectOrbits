"""
Differential Equations in the Real World — Orbital Mechanics Simulation
MATH 101 Group Project | Dr Georgios Charalambous
Topic: Planetary Orbits & GPS/Satellite Trajectories

Physics:  Newton's Law of Gravitation drives a system of 1st-order ODEs.
Solver:   scipy.integrate.solve_ivp  (RK45 — Runge-Kutta 4/5)

State vector:  [x, y, vx, vy]
ODE:
    dx/dt  = vx
    dy/dt  = vy
    dvx/dt = -GM * x / r³
    dvy/dt = -GM * y / r³

Left  → Exaggerated ellipse e=0.5 (real Earth e=0.0167 ≈ circle)
Right → GPS satellite, real SI parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# ── Constants ────────────────────────────────────────────────
GM_SUN   = 4.0 * np.pi**2      # AU³/yr²
GM_EARTH = 3.986004418e14       # m³/s²
R_EARTH  = 6.371e6              # m
ALT_GPS  = 20_200e3             # m

# ── ODE (shared) ─────────────────────────────────────────────
def orbit_ode(t, s, GM):
    x, y, vx, vy = s
    r3 = (x*x + y*y) ** 1.5
    return [vx, vy, -GM*x/r3, -GM*y/r3]

# ── Solve elliptical orbit  e=0.50 ───────────────────────────
a_e, e_e = 1.0, 0.50
x0_e  = a_e * (1 - e_e)
vy0_e = np.sqrt(GM_SUN * (1 + e_e) / (a_e * (1 - e_e)))
T_e   = 1.0
t_e   = np.linspace(0, T_e, 2000)
sol_e = solve_ivp(orbit_ode, [0, T_e], [x0_e, 0, 0, vy0_e],
                  args=(GM_SUN,), method='RK45', t_eval=t_e,
                  rtol=1e-10, atol=1e-13, dense_output=True)
xe, ye = sol_e.y[0], sol_e.y[1]

# ── Solve GPS orbit ───────────────────────────────────────────
r_gps = R_EARTH + ALT_GPS
v_gps = np.sqrt(GM_EARTH / r_gps)
T_gps = 2 * np.pi * r_gps / v_gps
t_g   = np.linspace(0, T_gps, 2000)
sol_g = solve_ivp(orbit_ode, [0, T_gps], [r_gps, 0, 0, v_gps],
                  args=(GM_EARTH,), method='RK45', t_eval=t_g,
                  rtol=1e-10, atol=1e-13, dense_output=True)
xg_km = sol_g.y[0] / 1e3
yg_km = sol_g.y[1] / 1e3

# ── Pre-interpolate ───────────────────────────────────────────
FRAMES = 600
t_anim_e = np.linspace(0, T_e,   FRAMES)
t_anim_g = np.linspace(0, T_gps, FRAMES)
ed = sol_e.sol(t_anim_e)
gd = sol_g.sol(t_anim_g)

# ── Figure ────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), facecolor='#0a0a1a')
fig.suptitle('Differential Equations in Orbital Mechanics  ·  MATH 101',
             color='white', fontsize=13, fontweight='bold', y=0.97)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.05, right=0.97, top=0.93, bottom=0.07,
                       wspace=0.38, hspace=0.50)
ax_orb  = fig.add_subplot(gs[:, 0])
ax_gps  = fig.add_subplot(gs[0, 1])
ax_info = fig.add_subplot(gs[1, 1])

def style(ax, title):
    ax.set_facecolor('#060616')
    ax.set_title(title, color='#aaddff', fontsize=10, pad=7)
    ax.tick_params(colors='#8899aa', labelsize=7.5)
    for sp in ax.spines.values(): sp.set_color('#334455')

# ── Left: Elliptical orbit ────────────────────────────────────
style(ax_orb, "Elliptical Orbit  (e = 0.50)  —  Kepler's Laws")
ax_orb.set_aspect('equal', adjustable='box')
ax_orb.set_xlabel('x  (AU)', color='#8899aa', fontsize=9)
ax_orb.set_ylabel('y  (AU)', color='#8899aa', fontsize=9)

# Centre view on the orbit's own bounding box midpoint (NOT the Sun)
cx_mid   = (xe.max() + xe.min()) / 2
cy_mid   = (ye.max() + ye.min()) / 2
half     = max(xe.max()-xe.min(), ye.max()-ye.min()) / 2 * 1.30
ax_orb.set_xlim(cx_mid - half, cx_mid + half)
ax_orb.set_ylim(cy_mid - half, cy_mid + half)

ax_orb.plot(xe, ye, color='#1a3a5a', lw=1.0, ls='--', alpha=0.5)

# Sun at the FOCUS (x=0, y=0)
ax_orb.plot(0, 0, 'o', color='#ffe066', markersize=16, zorder=5)
ax_orb.add_patch(plt.Circle((0,0), half*0.04, color='#ffe066', alpha=0.2))
ax_orb.text(0.03, half*0.08, 'Sun\n(focus)', color='#ffe066', fontsize=7.5)

# Perihelion / aphelion labels
peri_x = xe.max()
aphe_x = xe.min()
ax_orb.annotate('Perihelion\n(fastest)', xy=(peri_x, 0),
                xytext=(peri_x - 0.05, half * 0.35),
                color='#ffaa66', fontsize=7.5, ha='right',
                arrowprops=dict(arrowstyle='->', color='#ffaa66', lw=0.9))
ax_orb.annotate('Aphelion\n(slowest)', xy=(aphe_x, 0),
                xytext=(aphe_x + 0.05, half * 0.35),
                color='#66aaff', fontsize=7.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='#66aaff', lw=0.9))

trail_e,  = ax_orb.plot([], [], '-',  color='#4499ff', lw=1.5, alpha=0.75)
dot_e,    = ax_orb.plot([], [], 'o',  color='#55bbff', markersize=10, zorder=6)
arr_v_e   = ax_orb.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color='#88ffcc', lw=1.8))
arr_r_e   = ax_orb.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color='#ffaa44', lw=1.4, linestyle='dashed'))

ax_orb.legend(handles=[
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ffe066', markersize=10, label='Sun (focus)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#55bbff', markersize=8,  label='Planet'),
    Line2D([0],[0], color='#88ffcc', lw=2, label='Velocity  v'),
    Line2D([0],[0], color='#ffaa44', lw=2, ls='--', label='Position  r'),
], loc='lower right', fontsize=7.5,
   facecolor='#111122', edgecolor='#334455', labelcolor='white', framealpha=0.85)

# ── Top-right: GPS ────────────────────────────────────────────
style(ax_gps, 'GPS Satellite  (MEO  ~20 200 km)')
ax_gps.set_aspect('equal', adjustable='box')
ax_gps.set_xlabel('x  (km)', color='#8899aa', fontsize=8)
ax_gps.set_ylabel('y  (km)', color='#8899aa', fontsize=8)
r_lim = r_gps / 1e3 * 1.20
ax_gps.set_xlim(-r_lim, r_lim)
ax_gps.set_ylim(-r_lim, r_lim)
ax_gps.plot(xg_km, yg_km, color='#1a3a5a', lw=0.7, ls='--', alpha=0.4)
ax_gps.add_patch(plt.Circle((0,0), R_EARTH/1e3, color='#2255bb', zorder=4))
ax_gps.add_patch(plt.Circle((0,0), R_EARTH/1e3*1.07, color='#2255bb', alpha=0.12))

trail_g,  = ax_gps.plot([], [], '-',  color='#ff9944', lw=1.2, alpha=0.75)
dot_g,    = ax_gps.plot([], [], 's',  color='#ffcc44', markersize=8, zorder=6)
arr_v_g   = ax_gps.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color='#88ffcc', lw=1.6))
arr_r_g   = ax_gps.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.3, linestyle='dashed'))

ax_gps.legend(handles=[
    mpatches.Patch(color='#2255bb', label='Earth'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#ffcc44', markersize=7, label='GPS Satellite'),
    Line2D([0],[0], color='#88ffcc', lw=2, label='Velocity  v'),
    Line2D([0],[0], color='#ff6666', lw=2, ls='--', label='Position  r'),
], loc='lower right', fontsize=7,
   facecolor='#111122', edgecolor='#334455', labelcolor='white', framealpha=0.85)

# ── Bottom-right: ODE readout ─────────────────────────────────
ax_info.set_facecolor('#08080f')
ax_info.axis('off')
ax_info.set_title('Live ODE Solver  (RK45)', color='#aaddff', fontsize=10, pad=6)

FMT = (
    "── ODE SYSTEM (both panels, different GM) ──\n"
    "  dx/dt  = vx\n"
    "  dy/dt  = vy\n"
    "  dvx/dt = −GM·x / r³\n"
    "  dvy/dt = −GM·y / r³\n"
    "────────────────────────────────────────────\n"
    "  ELLIPTICAL ORBIT  (e = 0.50)\n"
    "  t  = {te:.4f} yr  ({ted:.1f} days)\n"
    "  r  = {re:.4f} AU      |v| = {ve:.4f} AU/yr\n"
    "  x  = {xe:+.4f} AU     y  = {ye:+.4f} AU\n"
    "  vx = {vxe:+.5f}       vy = {vye:+.5f}\n"
    "────────────────────────────────────────────\n"
    "  GPS SATELLITE\n"
    "  t  = {tg:.0f} s  ({tgh:.2f} h)\n"
    "  r  = {rg:.1f} km      |v| = {vg:.3f} km/s\n"
    "  x  = {xg:+.1f} km    y  = {yg:+.1f} km\n"
    "  vx = {vxg:+.5f} km/s  vy = {vyg:+.5f} km/s\n"
    "────────────────────────────────────────────\n"
    "  scipy.solve_ivp  RK45  |  rtol = 1e-10"
)

info_txt = ax_info.text(
    0.03, 0.97, '', transform=ax_info.transAxes,
    va='top', ha='left', fontsize=8.0,
    fontfamily='monospace', color='#ccffdd', linespacing=1.65
)

# ── Animation ────────────────────────────────────────────────
TRAIL = 150

def init():
    for obj in [trail_e, dot_e, trail_g, dot_g]:
        obj.set_data([], [])
    info_txt.set_text('')
    return trail_e, dot_e, trail_g, dot_g, info_txt

def update(f):
    i0 = max(0, f - TRAIL)

    # Elliptical
    trail_e.set_data(ed[0, i0:f+1], ed[1, i0:f+1])
    cx, cy   = ed[0,f], ed[1,f]
    cvx, cvy = ed[2,f], ed[3,f]
    dot_e.set_data([cx], [cy])
    vs = half * 0.12
    arr_v_e.set_position((cx, cy));  arr_v_e.xy = (cx + cvx*vs/np.hypot(cvx,cvy+1e-30)*half*0.20,
                                                    cy + cvy*vs/np.hypot(cvx,cvy+1e-30)*half*0.20)
    arr_r_e.set_position((0, 0));    arr_r_e.xy  = (cx * 0.93, cy * 0.93)

    # GPS
    i0g = max(0, f - TRAIL)
    trail_g.set_data(gd[0,i0g:f+1]/1e3, gd[1,i0g:f+1]/1e3)
    gx, gy   = gd[0,f]/1e3, gd[1,f]/1e3
    gvx, gvy = gd[2,f]/1e3, gd[3,f]/1e3
    dot_g.set_data([gx], [gy])
    gvmag = np.hypot(gvx, gvy) + 1e-30
    gvs   = r_lim * 0.22
    arr_v_g.set_position((gx, gy));  arr_v_g.xy = (gx + gvx/gvmag*gvs, gy + gvy/gvmag*gvs)
    arr_r_g.set_position((0, 0));    arr_r_g.xy  = (gx * 0.90, gy * 0.90)

    # Readout
    te = t_anim_e[f]; re = np.hypot(cx,cy); ve = np.hypot(cvx,cvy)
    tg = t_anim_g[f]; rg = np.hypot(gx,gy); vg = np.hypot(gvx,gvy)
    info_txt.set_text(FMT.format(
        te=te, ted=te*365.25, re=re, ve=ve,
        xe=cx, ye=cy, vxe=cvx, vye=cvy,
        tg=tg, tgh=tg/3600,
        rg=rg, vg=vg,
        xg=gx, yg=gy, vxg=gvx, vyg=gvy,
    ))
    return trail_e, dot_e, trail_g, dot_g, info_txt

ani = FuncAnimation(fig, update, frames=FRAMES,
                    init_func=init, interval=28, blit=False)
plt.show()