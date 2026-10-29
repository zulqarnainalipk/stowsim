"""Orthographic depth-map camera (observation) and RGB snapshot camera (visualisation)."""
import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient


class Camera:
    """
    Class that photographs the container state and produces a height map (height image)
    """
    def __init__(self, client: BulletClient, config: dict):
        """
        config:
            target_pos (list): centre point the camera looks at [x, y, z]
            distance (float): distance from the look-at point
            yaw, pitch, roll (float): camera rotation angles (degrees).
                                      pitch=-90 looks straight down (top-down).
                                      Set pitch=0 or similar to look from the side.
            img_width, img_height (int): resolution of the output image
            fov (float): field of view
            near_val, far_val (float): minimum and maximum distance of the imaging range
        """
        self.client = client
        self.width = config['img_width']
        self.height = config['img_height']
        self.near = config['near_val']
        self.far = config['far_val']
        self.fov = config['fov']


        # compute the camera pose (view matrix) and the projection settings (projection matrix)
        self.distance = config['distance']
        self.yaw = config['yaw']
        self.pitch = config['pitch']
        self.roll = config['roll']
        self.view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=config['target_pos'],
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            upAxisIndex=2  # Z axis is up
        )
        
        self.proj_matrix = self.client.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / self.height,
            nearVal=self.near,
            farVal=self.far
        )

        self.camera_y = config['target_pos'][1] + config['distance']


    def get_heightmap(self, center: tuple[float, float, float], lwh: tuple[float, float, float]):
        """
        Photograph the current container state and return the depth map
        
        Returns:
            np.array: 2-D array of shape (height, width). Values are heights in metres.
        """
        # compute the camera pose (view matrix) and the projection settings (projection matrix)
        # depth of the container
        c_depth = lwh[1]
        self.door_y = center[1] - (c_depth / 2.0)
        distance = 0.15*c_depth # keep some margin for the camera viewpoint

        # fix the camera position directly on the door side (negative Y)
        camera_eye = [center[0], self.door_y - distance, center[2]]
        self.view_matrix = self.client.computeViewMatrix(
            cameraEyePosition=camera_eye,
            cameraTargetPosition=center,
            cameraUpVector=[0, 0, 1] # Z axis is up
        )

        # --- padding logic that prevents distortion ---
        # 1. aspect ratio of the image, fixed by the shared memory
        target_aspect = self.width / self.height
        
        # 2. aspect ratio of the actual container
        container_aspect = lwh[0] / lwh[2]

        # 3. widen the physical frame (phys_w, phys_h) so the image is not distorted
        if container_aspect > target_aspect:
            # container is wider than the image: fit the width and show extra height (margins top and bottom)
            phys_w = lwh[0]
            phys_h = lwh[0] / target_aspect
        else:
            # container is taller than (or as tall as) the image: fit the height and show extra width (margins left and right)
            phys_h = lwh[2]
            phys_w = lwh[2] * target_aspect

        half_w = phys_w / 2.0
        half_h = phys_h / 2.0
        
        left = -half_w
        right = half_w
        bottom = -half_h
        top = half_h
        
        self.near = distance - 0.01 # start imaging slightly in front of the door
        self.far  = distance + c_depth + 0.1

        # orthographic projection matrix
        self.proj_matrix = [
            2.0 / (right - left), 0.0, 0.0, 0.0,
            0.0, 2.0 / (top - bottom), 0.0, 0.0,
            0.0, 0.0, -2.0 / (self.far - self.near), 0.0,
            -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(self.far + self.near) / (self.far - self.near), 1.0
        ]

        # get the image data from PyBullet
        # it is important to specify ER_TINY_RENDERER so that it also works in p.DIRECT mode
        _, _, _, depth_buffer, _ = self.client.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )

        # convert the depth buffer to a NumPy array, reshape: (H, W)
        depth_buffer = np.array(depth_buffer).reshape((self.height, self.width))
        depth_linear = self.near + (self.far - self.near) * depth_buffer

        available_depth = depth_linear - distance
        available_depth = np.where(available_depth > c_depth + 0.05, 0.0, available_depth)
        available_depth = np.maximum(available_depth, 0.0)


        return available_depth.astype(np.float32)


    def get_rgb_image(self):
        """Get an RGB image for debugging (meant for humans to look at)"""
        w, h, rgb, _, _ = self.client.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        # RGB array (H, W, 4) -> RGBA
        return np.array(rgb, dtype=np.uint8).reshape((self.height, self.width, 4))

