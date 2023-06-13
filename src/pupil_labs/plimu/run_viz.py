import click
import cv2
import numpy as np

from pupil_labs.realtime_api.simple import Device
from pupil_labs.plimu.visualizer import IMUVisualizer


def get_quat(imu_datum):
    q = np.asarray(
        [
            imu_datum.quaternion.x,
            imu_datum.quaternion.y,
            imu_datum.quaternion.z,
            imu_datum.quaternion.w,
        ]
    )
    if np.all(q == 0.0):
        q[0] = 1.0
    return q


@click.command()
@click.option(
    "--address",
    help="Address of the Neon Companion to connect to.",
)
@click.option("--port", default=8080, help="Port to connect to.")
@click.option("--show_stars", default=False, help="Flag for star visualization.")
def run_viz(address, port, show_stars=False):
    device = Device(address, port)

    imu_visualizer = IMUVisualizer(show_stars=show_stars)

    while True:
        frame = device.receive_scene_video_frame()
        imu_datum = device.receive_imu_datum()
        quat_imu = get_quat(imu_datum)

        frame = imu_visualizer.draw(frame.bgr_pixels, quat_imu, frame.datetime)

        cv2.imshow("PLIMU Visualizer 0.1 - Press ESC to quit", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
