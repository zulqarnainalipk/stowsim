"""Placement validation: action format, container inclusion, insertion-path collision test and settling test."""
import pybullet as p
import numpy as np
import time
import math
from pybullet_utils.bullet_client import BulletClient

from .utils import ORNS, get_half_ext
from .items import Item
from .containers import Container


class BaseValidator:
    def __init__(self, client: BulletClient, config: dict, render_mode: str | None = None):
        self.client = client
        self.config = config
        self.render_mode = render_mode
        self.safety_margin: float = self.config['safety_margin']
        self.start_margin: float = self.config.get('start_margin', 0.01)
        self.inclusion_margin: float = self.config.get('inclusion_margin', 0.02)
        self.start_z: float = self.config['start_z']
        self.ceiling_margin: float = self.config.get('ceiling_margin', 0.01)


    def check_action(self, action, action_config: dict, num_containers: int, num_visible_items: int):
        valid = True
        truncated = False
        info = {}

        # check the keys
        if set(action.keys()) != set(action_config['keys'].keys()):
            valid = False
            truncated = True
            info = {'status': {'has_valid_action_keys': False}}

        # check the placement coordinates
        if len(action['place_pos'])==3:
            for v in action['place_pos']:
                if not action_config['pos_lim']['low'] <= v <= action_config['pos_lim']['high']:
                    valid = False
                    truncated = True
                    info = {'status': {f'has_valid_action_pos (all elements must be in the range of {action_config['pos_lim']['low']} to {action_config['pos_lim']['high']}': False}}
        else:
            valid = False
            truncated = True
            info = {'status': {f'has_valid_action_pos (length must be 3)': False}}

        # check the indices of the container, the bag and the orientation
        container_idx_max = num_containers - 1
        visible_pool_max = num_visible_items - 1
        orientation_max = len(action_config['orientations']) - 1
        if not (0 <= action['container_idx'] <= container_idx_max):
            valid = False
            truncated = True
            info = {'status': {f'has_valid_action_index (index in container_idx must be in the range of {0} to {container_idx_max})': False}}
        if not (0 <= action['item_idx'] <= visible_pool_max):
            valid = False
            truncated = True
            info = {'status': {f'has_valid_action_index (index in item_idx must be in the range of {0} to {visible_pool_max})': False}}
        if not (0 <= action['orientation'] <= orientation_max):
            valid = False
            truncated = True
            info = {'status': {f'has_valid_action_index (index in orientation must be in the range of {0} to {orientation_max})': False}}
        
        return valid, truncated, info


class PlacementValidator(BaseValidator):
    def check_inclusion(self, container: Container, item: Item | None, target_pos: tuple[float, float, float], target_orn_idx: int) -> bool:
        """
        Judge whether the bag lies inside the container at the target position
        """
        if item is None:
            return False
        original_lwh =[item.length, item.width, item.height]
        half_lwh = get_half_ext(original_lwh, target_orn_idx)

        dots = (np.array(container.n_vecs)*(target_pos - np.array(container.points))).sum(axis=1) + np.dot(np.abs(np.array(container.n_vecs)), half_lwh)
        isin_all = np.all(dots<=self.inclusion_margin)
        if not bool(isin_all):
            print(f'not included. {dots.tolist()}')

        return bool(isin_all)


    def check_transport_path(self, container: Container, item: Item | None, target_pos: tuple[float, float, float], target_orn_idx: int, step_len: float = 0.01) -> bool:
        """
        Judge whether there is a collision on the trajectory that carries the bag to the target position (target_pos: world frame)
        Coordinates are absolute
        """
        if item is None:
            return False
        step_len = float(self.config.get('path_step', step_len))   # discretisation of the insertion path (m); default 1 cm

        # save the current state before carrying the bag
        original_lwh =[item.length, item.width, item.height]
        half_lwh = get_half_ext(original_lwh=original_lwh, orn_idx=target_orn_idx)
        x_min = -container.length/2 + container.thickness + container.cut_x + half_lwh[0] + self.start_margin
        x_max = container.length/2 - container.thickness - half_lwh[0] - self.start_margin
        rel_target = container.global_to_local(target_pos)    # convert to relative coordinates
        rel_x = min(max(rel_target[0], x_min), x_max)

        # 1. define the reference surfaces
        # surfaces the bag can rest on (floor, top of the shelf)
        resting_surfaces = [
            container.thickness, 
            (container.height / 2.0) + container.thickness + container.buffer
        ]
        
        # ceiling surfaces (underside of the shelf, container ceiling)
        ceiling_surfaces = [
            (container.height / 2.0) + container.buffer,
            container.height + container.buffer - container.thickness
        ]

        # default float / drop amount
        effective_start_z = self.start_z

        # 2. check below (resting on a surface)
        bottom_z = target_pos[2] - half_lwh[2]
        for r_z in resting_surfaces:
            if 0 <= (bottom_z - r_z) <= 0.05:
                effective_start_z = 0.0
                break

        # 3. check above (avoid hitting the ceiling)
        top_z = target_pos[2] + half_lwh[2]
        if effective_start_z > 0.0: # only checked when the bag is not resting on a surface
            for c_z in ceiling_surfaces:
                clearance = c_z - top_z
                # when the gap to the ceiling is smaller than the planned float amount + margin
                if 0 <= clearance < (effective_start_z + self.ceiling_margin):
                    # clip to the highest height that still avoids a collision
                    effective_start_z = max(0.0, clearance - self.ceiling_margin - 0.0005)
                    break

        rel_z = min(container.height + container.buffer - container.thickness - half_lwh[2] - self.start_margin, rel_target[2] + effective_start_z) # drop it from slightly above
        rel_start = (rel_x, -container.width/2, rel_z)
        start_pos = container.local_to_global(rel_start)

        # create the bag
        target_orn = p.getQuaternionFromEuler(ORNS[target_orn_idx])
        item_id = item.spawn(self.client, initial_pos=start_pos, initial_orn=target_orn)
        state_id = self.client.saveState()

        print(f'transport item {item.index}')

        # move along the Y axis (the z coordinate is already matched to the target)
        target_0 = (start_pos[0], target_pos[1], start_pos[2])
        steps = max(math.ceil(abs(target_0[1]-start_pos[1])/step_len), 1)
        is_packable, current_pos = self._move_item(container=container, item=item, item_id=item_id, start_pos=start_pos,
                                                   target_pos=target_0, target_orn=target_orn, steps=steps)
        if not is_packable:
            print(f'item {item.index} not packable. removed.')
            self.client.restoreState(stateId=state_id)
            self.client.removeState(state_id)
            item.remove(client=self.client)
            self.client.stepSimulation()

            return is_packable

        # move along the X axis
        target_1 = (target_pos[0], target_pos[1], start_pos[2])
        steps = max(math.ceil(abs(target_1[0]-current_pos[0])/step_len), 1)
        is_packable, current_pos = self._move_item(container=container, item=item, item_id=item_id, start_pos=current_pos,
                                                   target_pos=target_1, target_orn=target_orn, steps=steps)


        # when the check is over, restore the state and free the memory
        self.client.restoreState(stateId=state_id)
        self.client.removeState(state_id)
        if not is_packable:
            print(f'item {item.index} not packable. removed.')
            item.remove(client=self.client)
            self.client.stepSimulation()

        return is_packable


    def place_item(self, item: Item | None, target_pos: tuple[float, float, float], target_orn_idx: int) -> bool:
        if item is None:
            return False
        target_orn = p.getQuaternionFromEuler(ORNS[target_orn_idx])
        displacement_threshold = self.config['displacement_threshold']
        angle_displacement_threshold = self.config.get('angle_displacement_threshold', 30)

        # 1. warp the bag straight to the target position whose interference check passed
        item.set_pose(self.client, target_pos, target_orn)

        # 2. short physics run for settling right after the warp
        for _ in range(self.config['settle_wait_step']):
            self.client.stepSimulation()
            if self.render_mode == "human": 
                time.sleep(1./240.)


        # 3. final check that the bag has not collapsed after settling (stability)
        final_pos, final_orn = item.get_pose(self.client)

        # if the final position is far from the target, judge that the bag collapsed after placement
        displacement = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
        dot_product = abs(sum(a*b for a, b in zip(target_orn, final_orn))) # type: ignore
        dot_product = min(1.0, dot_product)
        angle_diff = 2 * math.acos(dot_product)
        if displacement > displacement_threshold or angle_diff > math.radians(angle_displacement_threshold):
            item.remove(client=self.client)
            self.client.stepSimulation()
            print(f'Not safe place (disp: {displacement:.2f}m, angle: {math.degrees(angle_diff):.1f}deg). Item {item.index} removed.')
            return False
        print(f'place item {item.index}')
        return True


    def _move_item(self, container: Container, item: Item, item_id: int | None, start_pos: tuple[float,float, float],
                   target_pos: tuple[float, float, float], target_orn: tuple[float, float, float, float],
                   steps: int) -> tuple[bool, tuple[float, float, float]]:
        is_packable = True
        # split the way from the start to the target and check while moving a little at a time
        for i in range(steps + 1):
            # compute the current position on the path by linear interpolation
            fraction = i / steps
            current_pos = (
                start_pos[0] + (target_pos[0] - start_pos[0]) * fraction,
                start_pos[1] + (target_pos[1] - start_pos[1]) * fraction,
                start_pos[2] + (target_pos[2] - start_pos[2]) * fraction
            )
            # without advancing physics time, force the bag onto the path position
            item.set_pose(self.client, current_pos, target_orn)
            self.client.performCollisionDetection()
            
            # detect objects within the given distance (safety_margin) with getClosestPoints
            for other_item in container.packed_items:
                if other_item.pybullet_id is not None:
                    closest_points = self.client.getClosestPoints(
                        bodyA=item_id,
                        bodyB=other_item.pybullet_id,
                        distance=self.safety_margin
                    )

                    # a return value means something is within the margin
                    for pt in closest_points:
                        body_b = pt[2] # object ID of the contact partner
                        if body_b == other_item.pybullet_id:
                            print(f'collision: {item.index} and {other_item.index} at {current_pos}, distance {pt[8]}')
                            is_packable = False
                            break


            # collision check with the shelf, if there is one
            if container.require_shelf:
                closest_points = self.client.getClosestPoints(
                    bodyA=item_id,
                    bodyB=container.shelf_bullet_id,
                    distance=self.safety_margin
                )
        
                for pt in closest_points:
                    body_b = pt[2] # object ID of the contact partner
                    if body_b == container.shelf_bullet_id:
                        print(f'collision: {item.index} and {body_b}(shelf) at {current_pos}')
                        is_packable = False
                        break

            # collision check with the small shelf on the left
            closest_points = self.client.getClosestPoints(
                bodyA=item_id,
                bodyB=container.small_shelf_bullet_id,
                distance=self.safety_margin
            )

            for pt in closest_points:
                body_b = pt[2] # object ID of the contact partner
                if body_b == container.small_shelf_bullet_id:
                    print(f'collision: {item.index} and {body_b}(small shelf) at {current_pos}, distance {pt[8]}')
                    is_packable = False
                    break


            if not is_packable:
                break

        return is_packable, current_pos