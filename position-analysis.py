from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

year = 2020
month = 3
day = 10
robotTime = [16, 59]
webcamTime = [16, 59, 57, 143959]
# First test of Webcam vs Robot positioning:
# day = 2
# robotTime = [15, 37]
# webcamTime = [15, 38, 24, 598040]

# day = 9
# webcamTime = [16, 39, 52, 765186]

fig = plt.figure(1)
robotAxes = fig.add_subplot(1, 2, 1, aspect=1)
robotAxes.set_ylabel("y(m)")
robotAxes.set_xlabel("x(m)")
robotAxes.set_title("Robot")
webcamAxes = fig.add_subplot(1, 2, 2, aspect=1)
webcamAxes.set_ylabel("y(m)")
webcamAxes.set_xlabel("x(m)")
webcamAxes.set_title("Webcam")


try:
    robotData = pd.read_csv(
        "Turtlebot_position_log-{}-{}-{}-{}:{}.csv".format(
            year, month, day, *robotTime
        ),
        header=0,
        parse_dates=[0],
        infer_datetime_format=True,
    )
except NameError:
    pass
try:
    webcamData = pd.read_csv(
        "Webcam_position_log-{}-{:02d}-{:02d}T{}:{}:{}.{}.csv".format(
            year, month, day, *webcamTime
        ),
        header=0,
        parse_dates=[0],
        infer_datetime_format=True,
    )
except NameError:
    pass


def arrow_returner(angle, length=0.05):
    x = length * np.cos(angle)
    y = length * np.sin(angle)
    return x, y

    # "Path Measured By Robot During Test at {}:{} on {}-{}-{}".format(
    #     *robotTime, day, month, year
    # "Path Measured By Webcam During Test at {}:{} on {}-{}-{}".format(
    #     *webcamTime[:2], day, month, year


robotAxes.plot(
    robotData["x(m)"], robotData["y(m)"],
)

#
# robotAxes.plot(robotData["x(m)"], robotData["y(m)"])
webcamAxes.plot(
    webcamData[webcamData["TagID"] == 42]["x(m)"],
    webcamData[webcamData["TagID"] == 42]["y(m)"],
)

# webcamAxes.arrow(
#     webcamData[webcamData["TagID"] == 42]["x(m)"],
#     webcamData[webcamData["TagID"] == 42]["y(m)"],
#     *arrow_returner(webcamData[webcamData["TagID"] == 42]["theta(rad)"]),
# )

plt.show()
