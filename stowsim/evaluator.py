"""Evaluation of a finished scene: fill score and fraction of bags placed."""
import pybullet as p
import numpy as np
import math
import os
from pybullet_utils.bullet_client import BulletClient
from .containers import Container
from .items import Item


class Evaluator:
    """
    Evaluation module that computes the loading state inside the containers
    """
    def __init__(self, client: BulletClient, config: dict):
        self.client = client
        self.config = config
        self.inclusion_margin: float = config.get('inclusion_margin', 0.01)


    def calculate_fill_rate(self, containers: list[Container]) -> tuple[float, list[Item]]:
        """
        Compute the fill rate. Only bags that lie completely inside the container are counted.
        """
        total_item_volume = 0
        total_container_volume = sum(c.volume for c in containers)
        out_items = []
        for container in containers:
            if not container.packed_items:
                continue

            for item in container.packed_items:
                pos, orn = item.get_pose(self.client)
                if pos is None or orn is None:
                    continue
                rot_mat = np.array(self.client.getMatrixFromQuaternion(orn)).reshape(3,3)
                hl, hw, hh = item.length/2, item.width/2, item.height/2
                local_corners = np.array([
                    [hl, hw, hh],
                    [hl, hw, -hh],
                    [hl, -hw, hh],
                    [hl, -hw, -hh],
                    [-hl, hw, hh],
                    [-hl, hw, -hh],
                    [-hl, -hw, hh],
                    [-hl, -hw, -hh]
                ])
                global_corners = np.dot(local_corners, rot_mat.T) + np.array(pos)
                is_inside = True
                n_vecs = np.array(container.n_vecs)
                points = np.array(container.points)
                for corner in global_corners:
                    # vector from the corner to a point on each plane
                    vecs = corner - points
                    # dot product with the normal vector
                    dots = np.sum(n_vecs * vecs, axis=1)
                    if np.any(dots > self.inclusion_margin):
                        is_inside = False
                        print(f'item {item.index} not inside (hit boundary plane), {dots}')
                        out_items.append(item)
                        break
                
                # add the volume of bags that satisfy every condition (completely inside)
                if is_inside:
                    total_item_volume += item.volume


        if total_item_volume==0:
            return 0, out_items

        fill_score = min(100*total_item_volume / total_container_volume, 100)

        return fill_score, out_items


    def evaluate(self, containers: list[Container], total_items: int) -> dict[str, float]:
        """Return the evaluation of the current containers as a dict"""
        num_packed_items = sum(len(c.packed_items) for c in containers)
        packed_items_percent = num_packed_items / total_items
        fill_score, out_items = self.calculate_fill_rate(containers)

        report = {
            "fill_score": fill_score,
            "num_placed_items": packed_items_percent
        }

        return report
