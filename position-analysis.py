from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

year = 2020
month = 3
day = 2
robotTime = [15, 37]
webcamTime = [15, 38, 24, 598040]
# time = [14, 29]

robotData = pd.read_csv(
    "Turtlebot_position_log-{}-{}-{}-{}:{}.csv".format(year, month, day, *robotTime),
    header=0,
    parse_dates=[0],
    infer_datetime_format=True,
    # skiprows=[6, 210],  # Need to handle rows from previous iterations that are too long
)

webcamData = pd.read_csv(
    "Webcam_position_log-{}-{:02d}-{:02d}T{}:{}:{}.{}.csv".format(
        year, month, day, *webcamTime
    ),
    header=0,
    parse_dates=[0],
    infer_datetime_format=True,
    # skiprows=[6, 210],  # Need to handle rows from previous iterations that are too long
)

print(robotData.head())
print(webcamData.head())
# self.fig_start_end = plt.figure(1)
# self.fig_start_end_ax = self.fig_start_end.add_subplot(1, 1, 1, aspect=1)
# self.fig_start_end_ax.set_title("Q1: Robot Positions")
plt.figure(1)
plt.title(
    "Path Measured By Robot During Test at {}:{} on {}-{}-{}".format(
        *robotTime, day, month, year
    )
)
plt.plot(robotData["x(m)"], robotData["y(m)"])
plt.ylabel("y(m)")
plt.xlabel("x(m)")

robotData["Timestep"] = robotData["Timestamp"] - robotData["Timestamp"].shift(-1)

plt.figure(2)
plt.title(
    "Path Measured By Webcam During Test at {}:{} on {}-{}-{}".format(
        *webcamTime[:2], day, month, year
    )
)
plt.plot(
    webcamData[webcamData["TagID"] == 42]["x(m)"],
    -webcamData[webcamData["TagID"] == 42]["y(m)"],
)

plt.show()

# class robotDataPoint:
#     def __init__(self, dt, x, y, theta):
#         self.dt = dt
#         self.x = x
#         self.y = y
#         self.theta = theta

#
# year = 2020
# month = 2
# day = 3
# time = [13, 46]
# f = open("Turtlebot_position_log-{}-{}-{}-{}:{}.csv".format(year, month, day, *time))
# headers = f.readline()
#
#
# index = 0
# target = (0, 0, 0)
#
# robotData_sorted_by_targets = [[]]
# targets = [target]
# for line in f:
#     hour, minute, second, microsecond, x, y, theta = line.split(",")
#     if hour == minute == second == microsecond == "00":
#         target = (x, y, theta)
#         targets.append(target)
#         index += 1
#         robotData_sorted_by_targets.append([])
#         print("Found new target")
#
#     else:
#         dt = datetime(
#             year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc
#         )
#         point = robotDataPoint(dt, x, y, theta)
#         robotData_sorted_by_targets[index].append(point)
