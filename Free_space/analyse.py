import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
df = pd.read_csv("msd_data_D_1.0_dt_0.001_N_100000.csv")
time = df['time'].values
msdx = df['msd_x'].values
msdy = df['msd_y'].values
msd_total = msdx + msdy

# Fit MSD vs time 
fit_x = np.polyfit(time[-90000:], msdx[-90000:], 1)
fit_y = np.polyfit(time[-90000:], msdy[-90000:], 1)
fit_total = np.polyfit(time[-90000:], msd_total[-90000:], 1)

D_eff_x = fit_x[0] / 2
D_eff_y = fit_y[0] / 2
D_eff_total = fit_total[0] / 4  # 2D passive particle

# Print values
print(f"Effective diffusion constant Dx: {D_eff_x:.5f}")
print(f"Effective diffusion constant Dy: {D_eff_y:.5f}")
print(f"Effective diffusion constant Dtotal: {D_eff_total:.5f}")

# Fit lines
t_sample = np.linspace(time[-90000], time[-1], 50)
y_fit_x = fit_x[0] * t_sample + fit_x[1]
y_fit_y = fit_y[0] * t_sample + fit_y[1]
y_fit_total = fit_total[0] * t_sample + fit_total[1]

# Calculate the slope of msdx vs time
fit_msdx = np.polyfit(time[-90000:], msdx[-90000:], 1)
slope_msdx = fit_msdx[0]
print(f"Slope of MSD_x vs Time (Diffusion constant): {slope_msdx / 2:.5f}")

# Plotting setup
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])  # MSD_x
ax2 = fig.add_subplot(gs[0, 1])  # MSD_y
ax3 = fig.add_subplot(gs[1, :])  # MSD_total

# Plotting MSDx
ax1.loglog(time[1:], msdx[1:], linewidth=1, color='blue', label='MSD_x')
ax1.loglog(
    t_sample, y_fit_x, 
    marker='^',
    linestyle='None',
    markersize=4,
    color='navy',
    label='Fit',
    markerfacecolor='none',
    clip_on = False
)
ax1.set_title(f"MSD_x | D ≈ {D_eff_x:.3f}")
ax1.set_xlabel("Time")
ax1.set_ylabel("MSD_x")
ax1.legend()

# Plotting MSDy
ax2.loglog(time[1:], msdy[1:], linewidth=1, color='red', label='MSD_y')
ax2.loglog(t_sample, y_fit_y, 
           marker='^', 
           linestyle='None', 
           markersize=4, 
           color='purple', 
           label='Fit',
           markerfacecolor = 'none',
           clip_on = False
           )
ax2.set_title(f"MSD_y | D ≈ {D_eff_y:.3f}")
ax2.set_xlabel("Time")
ax2.set_ylabel("MSD_y")
ax2.legend()

# Plotting total MSD
ax3.loglog(time[1:], msd_total[1:], linewidth=1, color='green', label='MSD_total')
ax3.loglog(t_sample, y_fit_total, 
           marker='^', 
           linestyle='None', 
           markersize=4, 
           color='darkgreen', 
           label='Fit', 
           markerfacecolor = 'none',
           clip_on = False
           )
ax3.set_title(f"MSD_total = MSD_x + MSD_y | D ≈ {D_eff_total:.3f}")
ax3.set_xlabel("Time")
ax3.set_ylabel("Total MSD")
ax3.legend(frameon=False)
ax2.legend(frameon=False)
ax1.legend(frameon=False)

ax1.set_xlim(0.0001, 100)
ax2.set_xlim(0.0001, 100)
ax3.set_xlim(0.0001, 100)

ax1.set_box_aspect(0.75) 
ax2.set_box_aspect(0.75) 
ax3.set_box_aspect(0.75) 
plt.savefig(f"Dt_1.pdf")
plt.tight_layout()
plt.show()

# Plotting msdx vs time
plt.figure(figsize=(6, 5))
plt.plot(time, msdx, label='MSD_x', color='blue')
plt.xlabel('Time')
plt.ylabel('MSD_x')
plt.title('MSD_x vs Time')
plt.legend()
plt.grid(True)

# Fitting a line to msdx vs time
plt.plot(t_sample, y_fit_x, 'r--', label='Fit')

meanx = df['meanx'].values
meany = df['meany'].values

# Linear fitting to mean displacements
fit_meanx = np.polyfit(time[-900000:], meanx[-900000:], 1)
fit_meany = np.polyfit(time[-900000:], meany[-900000:], 1)

vx = fit_meanx[0]
vy = fit_meany[0]

print(f"Slope of mean_dx vs time (v_x): {vx:.5f}")
print(f"Slope of mean_dy vs time (v_y): {vy:.5f}")



plt.tight_layout()
plt.legend()

plt.show()




