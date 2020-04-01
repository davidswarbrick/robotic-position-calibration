from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# First test of Webcam vs Robot positioning:
# day = 2
# robotTime = [15, 37]
# webcamTime = [15, 38, 24, 598040]

# day = 9
# webcamTime = [16, 39, 52, 765186]


# Analysing specific case:
year = 2020
month = 3
day = 10
robotTime = [16, 59]
webcamTime = [16, 59, 57, 143959]

# Import robot position data
r = pd.read_csv(
    "Turtlebot_position_log-{}-{}-{}-{}:{}.csv".format(year, month, day, *robotTime),
    header=0,
    parse_dates=[0],
    index_col=0,
    infer_datetime_format=True,
)
# sort by timestamp
r = r.sort_values(by="Timestamp")
# remove unnecessary empty column (caused by trailing comma)
r = r.drop(columns="Unnamed: 4")

# Import webcam data
w = pd.read_csv(
    "Webcam_position_log-{}-{:02d}-{:02d}T{}:{}:{}.{}.csv".format(
        year, month, day, *webcamTime
    ),
    header=0,
    parse_dates=[0],
    index_col=0,
    infer_datetime_format=True,
)
# rename theta column so it matches robot data
w = w.rename(columns={"theta(rad)": "Theta(rad)"})
w = w.sort_values(by="Timestamp")
w = w.drop(columns="Unnamed: 5")
# select just tag 42, and discard the tag ID column from here onwards
w = w[w["TagID"] == 42].drop(columns="TagID")

df = pd.concat([w, r], keys=["webcam", "robot"], names=["Data Source", "Timestamp"],)
df = df.sort_values(by="Timestamp")

# # Plotting paths
# fig1, ax1 = plt.subplots()
#
# ax1.plot(
#     df.loc[("robot", "x(m)")],
#     df.loc[("robot", "y(m)")],
#     label="Robot",
#     color="tab:blue",
# )
# ax1.plot(
#     df.loc[("webcam", "x(m)")],
#     df.loc[("webcam", "y(m)")],
#     label="Webcam",
#     color="tab:orange",
# )
# ax1.legend()
# ax1.set_title("Path Taken by the Robot")
# ax1.set_xlabel("x(m)")
# ax1.set_ylabel("y(m)")
# ax1.set_aspect("equal", "box")


def arrow_returner(angle, length=0.1):
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    return dx, dy


def latest_position_at_millisecond(t):
    if t < 0:
        raise KeyError("Cannot get negative time from start of range.")
    # Relative to first timestamp, t*1e6[ns] = t [ms]
    time = df.index[0][1] + pd.Timedelta(t * 1e6)
    if time > df.index[-1][1]:
        raise KeyError("Requested time outside data range.")

    # Look for the timestamp (index) at "time", if not use last value ->"pad"
    r_i = df.loc["robot"].index.get_loc(time, method="pad")
    w_i = df.loc["webcam"].index.get_loc(time, method="pad")

    # Return the latest data points
    return df.loc["robot"].iloc[r_i], df.loc["webcam"].iloc[w_i]


fig = plt.figure(figsize=(5, 5), dpi=200, facecolor="w", edgecolor="k")
ax = fig.add_subplot()
#
# fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.set_title("Path Taken by the Robot")
ax.set_xlabel("x(m)")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_ylabel("y(m)")
ax.set_aspect("equal", "box")


def plot_pos_at_time(t_s):
    ax.clear()
    ax.set_title("Path Taken by the Robot")
    ax.set_xlabel("x(m)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_ylabel("y(m)")
    ax.set_aspect("equal", "box")

    ax.plot(
        df.loc[("robot", "x(m)")],
        df.loc[("robot", "y(m)")],
        label="Robot",
        color="tab:blue",
    )
    ax.plot(
        df.loc[("webcam", "x(m)")],
        df.loc[("webcam", "y(m)")],
        label="Webcam",
        color="tab:orange",
    )
    try:
        r_d, w_d = latest_position_at_millisecond(t_s * 1e3)
        ax.arrow(
            r_d["x(m)"],
            r_d["y(m)"],
            *arrow_returner(r_d["Theta(rad)"]),
            width=0.05,
            color="tab:blue"
        )
        ax.arrow(
            w_d["x(m)"],
            w_d["y(m)"],
            *arrow_returner(w_d["Theta(rad)"]),
            width=0.05,
            color="tab:orange"
        )
    except KeyError:
        print("Failed")
        pass
    # plt.show()


ax.margins(x=0)

# axcolor = "lightgoldenrodyellow"
axcolor = "white"

axamp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

# sfreq = Slider(axfreq, "Freq", 0.1, 30.0, valinit=f0, valstep=delta_f)
time = Slider(axamp, "Time", 30, 300, valinit=30, valstep=1)
time.on_changed(plot_pos_at_time)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")


def reset(event):
    time.reset()


button.on_clicked(reset)


def return_between_milliseconds(t1, t2):
    """A function to return a data slice between two times measured from the start of test."""
    if t1 > t2:
        raise KeyError("Cannot get negative time slice.")
    # Relative to first timestamp, t*1e6[ns] = t [ms]
    time1 = df.index[0][1] + pd.Timedelta(t1 * 1e6)
    time2 = df.index[0][1] + pd.Timedelta(t2 * 1e6)
    if time1 > df.index[-1][1] or time2 > df.index[-1][1]:
        raise KeyError("Requested time outside data range.")

    # Look for the timestamp (index) at "time", if not use last value ->"pad"
    r_i1 = df.loc["robot"].index.get_loc(time1, method="pad")
    w_i1 = df.loc["webcam"].index.get_loc(time1, method="pad")

    r_i2 = df.loc["robot"].index.get_loc(time2, method="pad")
    w_i2 = df.loc["webcam"].index.get_loc(time2, method="pad")

    # Return the latest data points
    return df.loc["robot"].iloc[r_i1:r_i2], df.loc["webcam"].iloc[w_i1:w_i2]


def plot_between_seconds(t1, t2):
    r_t_slice, w_t_slice = return_between_milliseconds(t1 * 1e3, t2 * 1e3)
    # print(r_t_slice["x(m)"])
    # print(w_t_slice)
    fig3, ax3 = plt.subplots()
    ax3.plot(r_t_slice["x(m)"], r_t_slice["y(m)"], label="Robot", color="tab:blue")
    ax3.plot(w_t_slice["x(m)"], w_t_slice["y(m)"], label="Webcam", color="tab:orange")
    ax3.legend()
    ax3.set_title("Subsection of Path Taken by the Robot")
    ax3.set_xlabel("x(m)")
    ax3.set_ylabel("y(m)")
    ax3.set_aspect("equal", "box")

    fig4, ax4 = plt.subplots()
    ax4.plot(r_t_slice.index, r_t_slice["Theta(rad)"], label="Robot", color="tab:blue")
    ax4.plot(
        w_t_slice.index, w_t_slice["Theta(rad)"], label="Webcam", color="tab:orange"
    )
    plt.show()


print(return_between_milliseconds(54e3, 84e3))
r_t_slice, w_t_slice = return_between_milliseconds(54 * 1e3, 68 * 1e3)


# Merge the two slices, recording the webcam measurement closest in time to each robot measurement.
m = pd.merge_asof(
    r_t_slice,
    w_t_slice,
    left_index=True,
    right_index=True,
    direction="nearest",
    suffixes=["_r", "_w"],
)
# m.plot()
# plt.title("Merged Data")
m["e_x"] = m["x(m)_r"] - m["x(m)_w"]
m["e_y"] = m["y(m)_r"] - m["y(m)_w"]
m["e_t"] = m["Theta(rad)_r"] - m["Theta(rad)_w"]

plt.plot(m["e_t"].index, np.mean(m["e_t"]) * np.ones(m["e_t"].shape), "-")


def rotate_frame(points, theta):
    """Rotate vector of points by Theta"""
    if points.shape[1] != 2:
        return ValueError("Need to supply a n x 2 matrix of points")
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(r, points.T).T


plt.show()
