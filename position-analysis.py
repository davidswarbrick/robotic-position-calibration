from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

year = 2020
month = 3

# First test of Webcam vs Robot positioning:
day = 2
robotTime = [15, 37]
webcamTime = [15, 38, 24, 598040]


robotData = pd.read_csv(
    "Turtlebot_position_log-{}-{}-{}-{}:{}.csv".format(year, month, day, *robotTime),
    header=0,
    parse_dates=[0],
    infer_datetime_format=True,
)

webcamData = pd.read_csv(
    "Webcam_position_log-{}-{:02d}-{:02d}T{}:{}:{}.{}.csv".format(
        year, month, day, *webcamTime
    ),
    header=0,
    parse_dates=[0],
    infer_datetime_format=True,
)

print(robotData.head())

fig = plt.figure(1)
robotAxes = fig.add_subplot(1, 2, 1, aspect=1)
robotAxes.set_ylabel("y(m)")
robotAxes.set_xlabel("x(m)")
robotAxes.set_title(
    # "Path Measured By Robot During Test at {}:{} on {}-{}-{}".format(
    #     *robotTime, day, month, year
    # )
    "Robot"
)

robotAxes.plot(robotData["x(m)"], robotData["y(m)"])

webcamAxes = fig.add_subplot(1, 2, 2, aspect=1)

# robotData["Timestep"] = robotData["Timestamp"] - robotData["Timestamp"].shift(-1)
#
# plt.figure(2)
webcamAxes.set_ylabel("y(m)")
webcamAxes.set_xlabel("x(m)")
webcamAxes.set_title(
    # "Path Measured By Webcam During Test at {}:{} on {}-{}-{}".format(
    #     *webcamTime[:2], day, month, year
    # )
    "Webcam"
)
webcamAxes.plot(
    webcamData[webcamData["TagID"] == 42]["x(m)"],
    webcamData[webcamData["TagID"] == 42]["y(m)"],
)

plt.show()
