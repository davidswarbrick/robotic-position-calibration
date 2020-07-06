from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set font to Helvetica
font = {
    "family": "Helvetica",
}
plt.rc("font", **font)

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


def time_sorted(time, direction="nearest"):
    mf = pd.merge_asof(
        df.loc[("robot")],
        df.loc[("webcam")],
        left_index=True,
        right_index=True,
        direction=direction,  # backward, forward,nearest
        suffixes=["_r", "_w"],
        tolerance=pd.Timedelta("{}ms".format(time)),
    )
    # Remove the NaN values through a sanity check on the webcam x value:
    mf = mf[mf["x(m)_w"] > -0.5]
    # Least Squares Matching

    q_i = mf[["x(m)_w", "y(m)_w"]].to_numpy()
    p_i = mf[["x(m)_r", "y(m)_r"]].to_numpy()

    # Convert to zero-centred
    x_i = p_i - np.mean(p_i, axis=0)
    y_i = q_i - np.mean(q_i, axis=0)

    # Calculate least-squares matrices
    S = np.matmul(x_i.T, y_i)
    u, s, vh = np.linalg.svd(S)
    vu_det = np.linalg.det(np.matmul(vh.T, u.T))
    diag = np.array([[1, 0], [0, vu_det]])

    # Use these to calculate rotation and translation
    R = np.matmul(vh.T, np.matmul(diag, u.T))
    t = np.mean(q_i, axis=0) - np.matmul(R, np.mean(p_i, axis=0))
    print(R)
    # print("Arccos", np.rad2deg(np.arccos(R[0][0])))
    # print("Arcsin", np.rad2deg(-np.arcsin(R[0][1])))
    rotation = 0.5 * (np.arccos(R[0][0]) + np.arcsin(R[1][0]))
    # print("Avg rotation: ", np.rad2deg(rotation))

    # Rotate & Translate the robot data accordingly
    out = np.matmul(R, p_i.T)

    out[0, :] = out[0, :] + t[0]
    out[1, :] = out[1, :] + t[1]

    mf["x(m)_r"] = out.T[:, 0]
    mf["y(m)_r"] = out.T[:, 1]

    mf["e_x"] = mf["x(m)_r"] - mf["x(m)_w"]
    mf["e_y"] = mf["y(m)_r"] - mf["y(m)_w"]
    mf["e_d"] = np.sqrt(mf["e_x"] ** 2 + mf["e_y"] ** 2)
    mf["e_t"] = mf["Theta(rad)_w"] - mf["Theta(rad)_r"]
    return mf, rotation


def accuracy_given_bound(time, bound, direction="nearest"):
    (mf,) = time_sorted(time, direction)
    low_err = mf[mf["e_d"] < bound]
    high_err = mf[mf["e_d"] >= bound]
    acc = (low_err.shape[0] / mf.shape[0]) * 100
    return acc, mf.shape[0]


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
# fig, (ax1, ax2) = plt.subplots(2, 1)
for a in [ax1, ax2, ax3]:
    a.set_xlabel("x(m)",)
    a.set_ylabel("y(m)",)
    a.set_aspect("equal", "box")
ax2.set_xlim([0.3, 1.6])
ax3.set_xlim([0.3, 1.6])

ax2.set_ylim([-1.9, -0.2])
ax3.set_ylim([-1.9, -0.2])

ax1.set_title("Raw Position Data",)
ax1.plot(
    df.loc[("robot", "x(m)")],
    df.loc[("robot", "y(m)")],
    label="Odometry",
    color="tab:blue",
    marker=".",
    ls="None",
    ms=0.5,
)
ax1.plot(
    df.loc[("webcam", "x(m)")],
    df.loc[("webcam", "y(m)")],
    label="Webcam",
    color="tab:orange",
    marker=".",
    ls="None",
    ms=0.5,
)
ax1.legend(loc="lower left", markerscale=40)

direction = "nearest"
time = 5
tr_r, rotation = time_sorted(time, direction)
bound = 0.05
ax2.set_title("Aligned within 5ms, LLS Transformed",)
ax2.plot(
    tr_r["x(m)_w"],
    tr_r["y(m)_w"],
    label="Webcam",
    color="tab:orange",
    marker=".",
    ls="None",
    ms=4,
)
ax2.plot(
    tr_r["x(m)_r"],
    tr_r["y(m)_r"],
    label="Odometry",
    color="tab:blue",
    marker=".",
    ls="None",
    ms=4,
)
ax2.legend(loc="upper right", markerscale=1)

ax3.set_title("Odometry Measurement Error ",)
low_err = tr_r[tr_r["e_d"] < bound]
high_err = tr_r[tr_r["e_d"] >= bound]
ax3.plot(
    low_err["x(m)_r"],
    low_err["y(m)_r"],
    label="Odometry - Usable",
    color="tab:blue",
    marker=".",
    ls="None",
    ms=4,
)

# ax2.plot(
#     low_err["x(m)_w"],
#     low_err["y(m)_w"],
#     label="Webcam - Usable",
#     color="tab:orange",
#     marker=".",
#     ls="None",
#     ms=4,
# )

ax3.plot(
    high_err["x(m)_r"],
    high_err["y(m)_r"],
    label="Odometry - High Error",
    color="tab:red",
    marker="x",
    ls="None",
    ms=4,
)
print(
    "{:.2f}% Usable Odometry Data using {}, N={}, t={}".format(
        (low_err.shape[0] / tr_r.shape[0]) * 100, direction, tr_r.shape[0], time
    )
)
print(df.loc["robot"].shape[0], " number of data points in total")

# for dir in ["nearest", "forward", "backward"]:
#     for t in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
#         if dir != "nearest":
#             t = 2 * t
#         a, n = accuracy_given_bound(t, 0.05, dir)
#         print(
#             "{}, {}ms, {:.2f}%, {}, {:.2f}%".format(
#                 dir, t, a, n, (n / df.loc["robot"].shape[0]) * 100
#             )
#         )

ax3.legend(loc="upper right", markerscale=1)


# rcParams["figure.figsize"] (default: [6.4, 4.8]) = [6.4, 4.8].
size = {
    "figsize": [3, 3],
}
plt.rc("figure", **size)

tr_r.plot(y="e_d", use_index=True, label=r"$e_d$")
plt.ylabel("Error (m)")

plt.title("Error vs Time")
# plt.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/positionerror.png",
#     dpi=300,
#     bbox_inches="tight",
# )

plt.figure(5)
plt.hist(tr_r["e_d"], 40)
plt.ylabel("Number of Data Points Per Bin")
plt.title("Histogram of Position Error")
plt.xlabel(r"$e_d$ (m)")
# plt.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/errhistogram.png",
#     dpi=300,
#     bbox_inches="tight",
# )
print(
    "Error d info: mean = {:.4f}, max = {:.4f}, std dev = {:.4f}".format(
        np.mean(tr_r["e_d"]), np.max(tr_r["e_d"]), np.std(tr_r["e_d"])
    )
)
# plt.figure(6)
# plt.hist(tr_r["e_x"], 40)
# plt.ylabel("Number of Data Points")
# plt.title("Histogram of Position Error")
# plt.xlabel("e_x (m)")
#
# plt.figure(7)
# plt.hist(tr_r["e_y"], 40)
# plt.ylabel("Number of Data Points")
# plt.title("Histogram of Position Error")
# plt.xlabel("e_y (m)")
# print(tr_r.index)
# fig1.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/rawpositiondata.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# fig2.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/lls5mspositiondata.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# fig3.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/errorboundpositiondata.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# fig3.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/errorboundpositiondataclear.svg",
#     # dpi=300,
#     bbox_inches="tight",
#     transparent=True,
# )

# plt.figure()

size = {
    "figsize": [4, 3.5],
}
plt.rc("figure", **size)

tr_r.plot(y="e_t", label=r"$e_\theta$")
plt.title("Orientation Measurement Error")
plt.ylabel("Radians")

mean = np.mean(tr_r["e_t"])
plt.plot(
    tr_r["e_t"].index,
    mean * np.ones(tr_r["e_t"].shape),
    "--",
    label="Measured Mean : {:.2f}".format(mean),
)
plt.plot(
    tr_r["e_t"].index,
    rotation * np.ones(tr_r["e_t"].shape),
    "--",
    label="LLS Mean : {:.2f}".format(rotation),
)
plt.legend(loc="upper left")
# plt.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/thetaerr.png",
#     dpi=300,
#     bbox_inches="tight",
# )


def dist_travelled(x, y):
    dx = x - x.shift(-1)
    dy = y - y.shift(-1)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return np.sum(dist)


x = tr_r["x(m)_w"]
y = tr_r["y(m)_w"]
d = dist_travelled(x, y)
print("Webcam Test Dist ", d, "m")
start = tr_r.iloc[0].name
stop = tr_r.iloc[-1].name
t = stop - start
avg_speed = d / t.seconds
print("speed: ", avg_speed, r"m/s")


x = tr_r["x(m)_r"]
y = tr_r["y(m)_r"]
d = dist_travelled(x, y)
print("Robot Test Dist ", d, "m")
start = tr_r.iloc[0].name
stop = tr_r.iloc[-1].name
t = stop - start
avg_speed = d / t.seconds
print("speed: ", avg_speed, r"m/s")

mf = pd.merge_asof(
    df.loc[("robot")],
    df.loc[("webcam")],
    left_index=True,
    right_index=True,
    direction=direction,  # backward, forward,nearest
    suffixes=["_r", "_w"],
    tolerance=pd.Timedelta("{}ms".format(5)),
)
# Remove the NaN values through a sanity check on the webcam x value:
mf = mf[mf["x(m)_w"] > -0.5]

q_i = mf[["x(m)_w", "y(m)_w"]].to_numpy()
p_i = mf[["x(m)_r", "y(m)_r"]].to_numpy()

# Convert to zero-centred
x_i = p_i - np.mean(p_i, axis=0)
y_i = q_i - np.mean(q_i, axis=0)

# Calculate least-squares matrices
S = np.matmul(x_i.T, y_i)
u, s, vh = np.linalg.svd(S)
vu_det = np.linalg.det(np.matmul(vh.T, u.T))
diag = np.array([[1, 0], [0, vu_det]])
# Use these to calculate rotation and translation
R = np.matmul(vh.T, np.matmul(diag, u.T))
translation = np.mean(q_i, axis=0) - np.matmul(R, np.mean(p_i, axis=0))


def shift_by_time(time=0, timedelta=5):
    rob = df.loc[("robot")]
    web = df.loc[("webcam")]
    rob.index = rob.index + pd.DateOffset(milliseconds=time)
    # web.index = web.index + pd.DateOffset(milliseconds=time)
    mf = pd.merge_asof(
        rob,
        web,
        left_index=True,
        right_index=True,
        direction=direction,  # backward, forward,nearest
        suffixes=["_r", "_w"],
        tolerance=pd.Timedelta("{}ms".format(timedelta)),
    )
    # Remove the NaN values through a sanity check on the webcam x value:
    mf = mf[mf["x(m)_w"] > -0.5]
    # Least Squares Matching

    # q_i = mf[["x(m)_w", "y(m)_w"]].to_numpy()
    p_i = mf[["x(m)_r", "y(m)_r"]].to_numpy()
    #
    # # Convert to zero-centred
    # x_i = p_i - np.mean(p_i, axis=0)
    # y_i = q_i - np.mean(q_i, axis=0)
    #
    # # Calculate least-squares matrices
    # S = np.matmul(x_i.T, y_i)
    # u, s, vh = np.linalg.svd(S)
    # vu_det = np.linalg.det(np.matmul(vh.T, u.T))
    # diag = np.array([[1, 0], [0, vu_det]])
    # # Use these to calculate rotation and translation
    # R = np.matmul(vh.T, np.matmul(diag, u.T))
    # t = np.mean(q_i, axis=0) - np.matmul(R, np.mean(p_i, axis=0))

    # Rotate & Translate the robot data accordingly
    # print(R, translation)
    out = np.matmul(R, p_i.T)
    out[0, :] = out[0, :] + translation[0]
    out[1, :] = out[1, :] + translation[1]
    mf["x(m)_r"] = out.T[:, 0]
    mf["y(m)_r"] = out.T[:, 1]
    mf["e_x"] = mf["x(m)_r"] - mf["x(m)_w"]
    mf["e_y"] = mf["y(m)_r"] - mf["y(m)_w"]
    mf["e_d"] = np.sqrt(mf["e_x"] ** 2 + mf["e_y"] ** 2)
    mf["e_t"] = mf["Theta(rad)_w"] - mf["Theta(rad)_r"]
    return mf


def accuracy_time_shift(time, timedelta=5, bound=0.05):
    md = shift_by_time(time, timedelta)
    low_err = md[md["e_d"] < bound]
    high_err = md[md["e_d"] >= bound]
    acc = (low_err.shape[0] / md.shape[0]) * 100
    return acc, md.shape[0]


plt.figure(figsize=[5, 3])

timewidth = 2000

for timedelta in [5, 10, 20]:
    print("Time accuracy number of correspondenes")
    total_acc = []
    time = np.linspace(-timewidth, timewidth, 200)
    num = 0
    for t in time:
        a, n = accuracy_time_shift(t, timedelta)
        total_acc.append(a)
        # print("{:+10.2f}ms: {:4.2f}% {:5d}".format(t, a, n))
        num += n

    plt.plot(time, total_acc, label="$\pm${}ms, $e_d$<0.05m".format(timedelta))
    max_acc = np.max(total_acc)
    max_acc_ind = np.argmax(total_acc)
    print("{} values, avg: {}".format(num, num / time.shape[0]))
plt.xlabel("Time (ms)")
plt.title("Accuracy with Time Offset")
plt.ylabel("Accuracy (%)")
plt.legend(loc="upper left")
# plt.savefig(
#     "/home/david/Documents/Cambridge/Master's Project/Final Report/img/timelag.png",
#     dpi=300,
#     bbox_inches="tight",
# )
print("Max acc:", max_acc, " Time: ", time[max_acc_ind])

print(tr_r.shape)
print(low_err.shape)
tr_r = shift_by_time(350, 5)
print(tr_r.shape)
plt.figure(figsize=[6.4, 4.8])
plt.title("Aligned within 5ms, LLS Transformed, Lag Compensated",)
ax = plt.gca()
ax.set_xlabel("x(m)",)
ax.set_ylabel("y(m)",)
ax.set_aspect("equal", "box")
# plt.plot(
#     tr_r["x(m)_w"],
#     tr_r["y(m)_w"],
#     label="Webcam",
#     color="tab:orange",
#     marker=".",
#     ls="None",
#     ms=4,
# )
low_err = tr_r[tr_r["e_d"] < 0.05]
print(low_err.shape)
high_err = tr_r[tr_r["e_d"] >= 0.05]
plt.plot(
    low_err["x(m)_r"],
    low_err["y(m)_r"],
    label="Odometry",
    color="tab:blue",
    marker=".",
    ls="None",
    ms=4,
)
plt.plot(
    high_err["x(m)_r"],
    high_err["y(m)_r"],
    label="Odometry High Error",
    color="tab:red",
    marker="x",
    ls="None",
    ms=4,
)
plt.legend(loc="upper right", markerscale=1)

plt.show()
