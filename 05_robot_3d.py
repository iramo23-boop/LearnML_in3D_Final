import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def load_path(file="data_v8.npz"):
    data = np.load(file, allow_pickle=True)
    if "positions" in data:
        return data["positions"]
    if "x" in data and "y" in data:
        return np.column_stack([data["x"], data["y"]])
    arr = data[list(data.keys())[0]]
    return arr[:, :2]

path = load_path("data_v8.npz")
x = path[:, 0]
y = path[:, 1]
z = np.sin(np.linspace(0, 6, len(x))) * 0.2

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.set_title("LearnML_in3D - 3D Robot Navigation")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Robot height")

ax.plot(x, y, z, linewidth=2, label="AI driving path")

robot, = ax.plot([], [], [], marker="o", markersize=12, label="3D Robot")
direction, = ax.plot([], [], [], linewidth=3, label="Robot direction")

ax.legend()

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z) - 0.5, max(z) + 0.5)

def update(frame):
    robot.set_data([x[frame]], [y[frame]])
    robot.set_3d_properties([z[frame]])

    if frame < len(x) - 2:
        dx = x[frame + 1] - x[frame]
        dy = y[frame + 1] - y[frame]
    else:
        dx, dy = 0, 0

    direction.set_data([x[frame], x[frame] + dx * 5], [y[frame], y[frame] + dy * 5])
    direction.set_3d_properties([z[frame], z[frame]])

    return robot, direction

ani = FuncAnimation(fig, update, frames=range(0, len(x), max(1, len(x)//300)), interval=30)

plt.show()