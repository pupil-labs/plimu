import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pupil_labs.plimu.geometry import (
    get_star_polygon,
    get_unit_circle_polygon,
    angle_between,
)
from pupil_labs.plimu.utils import get_letter_from_svg, get_pl_logo_from_svg
from pupil_labs.plimu.rotations import euler_from_quaternion_yxz

# we use a single Neon calibration for now

cameraMatrix = np.asarray(
    [
        [887.68675656, 0.0, 800.62430596],
        [0.0, 887.66067058, 601.73557764],
        [0.0, 0.0, 1.0],
    ]
)

distCoeffs = np.asarray(
    [
        [
            -0.12984718,
            0.10905513,
            0.00036732,
            -0.00033288,
            0.00078861,
            0.17031277,
            0.05104419,
            0.02662509,
        ]
    ]
)

####################

poles = {
    "south": {"anchor": np.array([0.0, -1.0, 0.0])},
    "north": {"anchor": np.array([0.0, 1.0, 0.0])},
    "east": {"anchor": np.array([1.0, 0.0, 0.0])},
    "west": {"anchor": np.array([-1.0, 0.0, 0.0])},
    "up": {"anchor": np.array([0.0, 0.0, 1.0])},
    "down": {"anchor": np.array([0.0, 0.0, -1.0])},
}

horizons = {
    "horizon": {"cos_idx": 0, "sin_idx": 2},
    "north_south": {"cos_idx": 0, "sin_idx": 1},
    "east_west": {"cos_idx": 1, "sin_idx": 2},
}


def horizon_point(phi, cos_idx=0, sin_idx=2):
    point = np.zeros(3)
    point[cos_idx] = np.cos(phi)
    point[sin_idx] = np.sin(phi)
    return point


class SphericalPolygon:
    def __init__(
        self,
        points,
        anchor=np.array([0.0, 0.0, 1.0]),
        scale=1.0,
        color=(255, 255, 255),
        delta=None,
        R=np.eye(2),
    ):
        self.color = color
        self.scale = scale
        self.anchor = anchor / np.linalg.norm(anchor)
        self.points = np.asanyarray(points, dtype=np.float32)
        self.points = (R @ self.points.T).T
        e1, e2 = self.get_tangent_coordinate_system(delta=delta)
        self.points_in_tangent_plane = (
            self.anchor
            + self.scale * self.points[:, 0, None] * e1
            + self.scale * self.points[:, 1, None] * e2
        )

    def get_tangent_coordinate_system(self, delta=None):
        down = np.array([0.0, 0.0, -1.0])

        e1 = np.cross(down, self.anchor)
        e2 = np.cross(self.anchor, e1)

        if np.linalg.norm(e1) == 0.0:
            if delta is None:
                delta = np.random.uniform(-1e-4, 1e-4, 3)
            else:
                assert delta.shape == (3,), "delta must be a 3D vector"
            e1 = np.cross(down, self.anchor + delta)
            e2 = np.cross(self.anchor + delta, e1)

        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)

        return e1, e2

    def is_in_image(self, pose):
        optical_camera_axis = pose @ np.asarray([0.0, 0.0, 1.0])
        if np.arccos(np.dot(self.anchor, optical_camera_axis)) > np.pi / 2.8:
            return False
        else:
            return True


class HorizonSegment:
    def __init__(self, start, end, cos_idx=0, sin_idx=2):
        self.start = horizon_point(start, cos_idx, sin_idx)
        self.end = horizon_point(end, cos_idx, sin_idx)
        self.points = np.array([self.start, self.end])
        self.center = horizon_point((start + end) / 2, cos_idx, sin_idx)

    def is_in_image(self, pose):
        optical_camera_axis = pose @ np.asarray([0.0, 0.0, 1.0])
        if angle_between(self.center, optical_camera_axis, deg=True) > 65:
            return False
        else:
            return True


class IMUVisualizer:
    def __init__(
        self,
        imu_offset=12.0,
        show_stars=True,
        show_logo=True,
        show_time=True,
        show_roll_pitch_yaw=True,
    ):
        self.imu_offset = imu_offset

        self.show_stars = show_stars
        self.show_logo = show_logo
        self.show_time = show_time
        self.show_roll_pitch_yaw = show_roll_pitch_yaw

        points_on_unit_circle = get_unit_circle_polygon()
        self.poles = [
            SphericalPolygon(
                points_on_unit_circle, scale=0.03, color=(255, 255, 255), **value
            )
            for key, value in poles.items()
            if key != "up"  # on the upper pole we will draw the logo
        ]

        self.labels = [
            SphericalPolygon(
                get_letter_from_svg(key[0]), scale=0.015, color=(0, 0, 0), **value
            )
            for key, value in poles.items()
            if key != "up" and key != "down"  # upper and lower poles are not labeled
        ]

        ############################################################

        delta = np.radians(1)
        extent = np.radians(5)

        self.horizon_segments = [
            HorizonSegment(phi - delta, phi + delta, **value)
            for phi in np.arange(0, 2 * np.pi, extent + delta)
            for key, value in horizons.items()
        ]

        def angle_with_up(v):
            return np.arccos(np.dot(v, np.array([0, 0, 1])))

        self.horizon_segments = list(
            filter(
                lambda x: angle_with_up(x.center) > np.radians(3), self.horizon_segments
            )
        )

        ############################################################

        star_points = get_star_polygon()
        self.stars = []
        for p in np.random.normal(0, 0.9, (220, 2)):
            scale = np.random.uniform(0.003, 0.01)
            star = SphericalPolygon(
                star_points,
                anchor=np.array([p[0], p[1], 1.0]),
                scale=scale,
                color=(255, 255, 0),
            )
            self.stars.append(star)

        ############################################################

        delta = np.asarray([1e-5, 0.0, 0.0])
        R = np.asarray(
            [[-0.70710678, 0.70710678], [-0.70710678, -0.70710678]]
        )  # rotation by 225 deg

        self.logo = [
            SphericalPolygon(
                path,
                anchor=poles["up"]["anchor"],
                scale=0.03,
                color=(255, 255, 255),
                delta=delta,
                R=R,
            )
            for path in get_pl_logo_from_svg()
        ]

    def draw_element(self, frame, element, rot_world):
        if element.is_in_image(rot_world.T):
            polygon = cv2.projectPoints(
                element.points_in_tangent_plane,
                rot_world,
                np.asarray([0.0, 0.0, 0.0]),
                cameraMatrix,
                distCoeffs,
                None,
            )[0][:, 0, :]
            cv2.fillPoly(frame, np.int32([polygon]), element.color)

    def write_time(self, frame, time):
        frame_txt_font_name = cv2.FONT_HERSHEY_SIMPLEX
        frame_txt_font_scale = 1.0
        frame_txt_thickness = 3
        frame_txt = str(time)

        cv2.putText(
            frame,
            frame_txt,
            (20, 50),
            frame_txt_font_name,
            frame_txt_font_scale,
            (255, 255, 255),
            thickness=frame_txt_thickness,
            lineType=cv2.LINE_8,
        )

    def write_roll_pitch_yaw(self, frame, quat_imu):
        roll, pitch, yaw = euler_from_quaternion_yxz(*quat_imu)

        frame_txt_font_name = cv2.FONT_HERSHEY_SIMPLEX
        frame_txt_font_scale = 1.0
        frame_txt_thickness = 3
        frame_txt = "roll: {:.2f} pitch: {:.2f} yaw: {:.2f}".format(
            np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
        )

        cv2.putText(
            frame,
            frame_txt,
            (20, 100),
            frame_txt_font_name,
            frame_txt_font_scale,
            (255, 255, 255),
            thickness=frame_txt_thickness,
            lineType=cv2.LINE_8,
        )

    def draw(
        self,
        frame,  # as BGR
        quat_imu,
        datetime=None,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
    ):
        # convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_imu_in_world = R.from_quat(quat_imu).as_matrix()
        pose_scene_in_imu = cv2.Rodrigues(
            np.asarray([np.radians(-90 - self.imu_offset), 0.0, 0.0])
        )[0]
        pose_scene_in_world = pose_imu_in_world @ pose_scene_in_imu
        extrinsics_scene_in_world = pose_scene_in_world.T

        for segment in self.horizon_segments:
            if segment.is_in_image(pose_scene_in_world):
                line = cv2.projectPoints(
                    segment.points,
                    extrinsics_scene_in_world,
                    np.asarray([0.0, 0.0, 0.0]),
                    cameraMatrix,
                    distCoeffs,
                    None,
                )[0][:, 0, :]
                cv2.line(
                    frame,
                    (int(line[0, 0]), int(line[0, 1])),
                    (int(line[1, 0]), int(line[1, 1])),
                    (255, 255, 255),
                    4,
                )

        for element in self.poles + self.labels:
            self.draw_element(frame, element, extrinsics_scene_in_world)

        if self.show_logo:
            for element in self.logo:
                self.draw_element(frame, element, extrinsics_scene_in_world)

        if self.show_stars:
            for star in self.stars:
                self.draw_element(frame, star, extrinsics_scene_in_world)

        if self.show_time and datetime is not None:
            self.write_time(frame, datetime)

        if self.show_roll_pitch_yaw:
            self.write_roll_pitch_yaw(frame, quat_imu)

        # convert back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame


if __name__ == "__main__":
    imu_visualizer = IMUVisualizer()
