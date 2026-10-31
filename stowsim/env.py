"""Gymnasium-style environment: builds the physics world, validates and executes actions, produces observations."""
import numpy as np
import pybullet_data
import pybullet as p
import gymnasium as gym
import time
import os
import random
import atexit
from datetime import datetime
from gymnasium import spaces
from pybullet_utils import bullet_client
from PIL import Image
from multiprocessing import shared_memory

from .containers import MultiContainerManager
from .items import ItemStreamManager
from .validator import PlacementValidator
from .camera import Camera
from .evaluator import Evaluator


class GroundHandlingEnv(gym.Env):
    """
    3-D packing environment based on physics simulation with PyBullet
    Provides push-in placement into side-opening containers and an interface for reinforcement learning
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, config: dict[str, dict], verbose: bool, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config
        self.optimize = config['agent']['optimize']
        # optional episode rule: a bag that cannot be placed (no placement found, blocked path, collapse) is discarded and loading goes on
        # with the next bag, instead of ending the episode (default: the episode ends)
        self.continue_after_reject = bool(config.get('episode', {}).get('continue_after_reject', False))
        self.n_discarded = 0
        self.verbose = verbose


        # 1. connect to PyBullet
        if render_mode == 'human':
            self.client = bullet_client.BulletClient(connection_mode=p.GUI)
            self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # hide the unneeded GUI menus
        else:
            self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
            
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 2. initialise each submodule
        self.container_manager = MultiContainerManager(client = self.client, config = self.config['containers'])
        self.stream_manager = ItemStreamManager(config = self.config['item_stream'])
        self.validator = PlacementValidator(client=self.client, config=self.config['validator'])
        self.camera = Camera(client = self.client, config = self.config['camera'])
        self.evaluator = Evaluator(client = self.client, config = {'inclusion_margin': self.config['validator']['inclusion_margin']})
        self.visualize = self.config['visualizer']['vis']
        if self.visualize:
            self.visualizer = Camera(client=self.client, config=self.config['visualizer']['camera'])
            self.image_dir = os.path.join('images', 'steps', datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.image_dir, exist_ok=True)


        # 3. define the action space
        self.num_containers = len(self.container_manager.config['container_list'])
        self.action_space = spaces.Dict({
            'item_idx': spaces.Discrete(self.stream_manager.lookahead_k), # which bag in the pool to choose
            'container_idx': spaces.Discrete(self.num_containers),
            'place_pos': spaces.Box( # only the local coordinates (x, y, z) are specified
                low=np.array([-0.2, -0.2, 0.4], dtype=np.float32),
                high=np.array([0.2, 0.2, 0.8], dtype=np.float32),
                dtype=np.float32
            ),
            'orientation': spaces.Discrete(len(self.config['action']['orientations'])) # orientation of the bag
        })
        self.pos_lim = config['action']['pos_lim']

        # 4. define the observation space (for testing)
        img_h, img_w = 64, 64
        self.observation_space = spaces.Dict({
            "depth_map": spaces.Box(low=0, high=3.0, shape=(img_h, img_w), dtype=np.float32),
            "item_features": spaces.Box(low=0, high=5.0, shape=(self.stream_manager.lookahead_k, 3), dtype=np.float32)
        })

        self.max_num_containers=config['camera']['num_containers']
        self.shm_shape = (self.max_num_containers, self.camera.height, self.camera.width)
        self.shm_dtype = np.float32
        shm_size = self.shm_shape[0] * self.shm_shape[1] * self.shm_shape[2] * np.dtype(self.shm_dtype).itemsize

        # create the shared-memory region
        self.shm = shared_memory.SharedMemory(create=True, size=shm_size)
        # create a NumPy array that references that buffer directly
        self.shm_depth_map = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
        atexit.register(self._cleanup_shared_memory_at_exit)

        self.num_step: int = 0
        self.num_total_items: int = 0


    def _cleanup_shared_memory_at_exit(self):
        """Method that reliably removes the shared memory from the OS when the process exits"""
        try:
            self.shm.close()
            self.shm.unlink()
            if self.verbose:
                print("[atexit] Shared memory released safely.")
        except FileNotFoundError:
            pass
        except Exception as e:
            pass



    def reset_settings(self):
        # reset PyBullet
        self.client.resetSimulation()
        self.client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        physics = self.config.get('physics', {})   # optional engine settings (defaults: PyBullet's own time step and solver iterations)
        if physics.get('time_step') is not None:
            self.client.setPhysicsEngineParameter(fixedTimeStep=float(physics['time_step']))
        if physics.get('solver_iterations') is not None:
            self.client.setPhysicsEngineParameter(numSolverIterations=int(physics['solver_iterations']))
        self.client.setGravity(0, 0, -9.8) # set gravity
        self.client.loadURDF('plane.urdf') # create the floor

        # rebuild the containers
        self.container_manager.clear()
        self.container_manager.build()

        self.num_total_items = self.stream_manager.total_items + sum(len(c.packed_items) for c in self.container_manager.containers)

        # adjust the camera parameters for debugging
        target_position = [0.0, 0.0, 0.0]
        distance = 0
        for c in self.container_manager.containers:
            target_position[0] += c.center[0]
            target_position[1] += c.center[1]
            target_position[2] += c.center[2]
            distance += c.width
        
        target_position[0] /= len(self.container_manager.containers)
        target_position[1] /= len(self.container_manager.containers)
        target_position[2] /= len(self.container_manager.containers)
        distance = 1.8*distance/len(self.container_manager.containers)

        self.client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=self.config['visualizer']['camera']['yaw'],
            cameraPitch=self.config['visualizer']['camera']['pitch'], # -90 gives a top-down view
            cameraTargetPosition=target_position
        )


    def reset_item_stream(self):
        # create the bag stream
        self.stream_manager.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        observation = self._get_obs()
        if self.verbose:
            print(f'env: {len(self.stream_manager.all_items)}, {len(self.stream_manager.visible_pool)}')
        info = {}

        # if the visualisation flag is true, save a snapshot
        if self.visualize:
            img = Image.fromarray(self.visualizer.get_rgb_image())
            img.save(os.path.join(self.image_dir, f"{self.num_step:04}.png"))
        self.num_step += 1

        return observation, info


    def get_init_states(self) -> dict[str, list | int | None]:

        return {
            'optimize': self.optimize,
            'lookahead_k': self.stream_manager.lookahead_k,
            'container_list': self.container_manager.get_item_info_in_containers()
        }


    def get_info_for_optimization(self) -> list[dict]:
        return self.stream_manager.get_all_items()


    def set_item_order(self, order) -> bool:
        if isinstance(order, list) and set(order) == self.stream_manager.all_indices:
            self.stream_manager.set_order(order)
            return True
        return False


    def step(self, action):
        """
        Run one step: parse the action -> check the target position -> physical placement -> update the observation
        """
        reward = 0
        start = time.time()
        # ------ 0. check the action format etc. ------
        num_visible_items = len(self.stream_manager.visible_pool)
        valid, truncated, info = self.validator.check_action(action, self.config['action'], self.num_containers, num_visible_items)
        if not valid: # when False, truncated is always True
            return {}, 0, True, truncated, info

        # ------ 1. parse the action ------
        local_xyz = action['place_pos'] # [x, y, z]
        item_idx = action['item_idx']
        container_idx = action['container_idx']
        target_orn_idx = action['orientation']

        ## get the data of the selected bag (not spawned in the physics world yet)
        target_item = self.stream_manager.get_item(item_idx)
        target_container = self.container_manager.get_container(container_idx)

        ## convert to world coordinates
        global_pos = target_container.local_to_global(local_xyz)

        # ------ 2. pre-checks (e.g. no penetration) ------
        terminated = False
        truncated = False
        is_placed_safe = False
        is_valid = False

        def rejected():
            """The action failed: end the episode, or discard the bag and go on when continue_after_reject is set."""
            if self.continue_after_reject:
                self.stream_manager.pop_and_refill(item_idx)
                self.n_discarded += 1
                return False
            return True

        is_included = self.validator.check_inclusion(target_container, target_item, global_pos, target_orn_idx)

        if not is_included:
            terminated = rejected()
        else:
            is_valid = self.validator.check_transport_path(target_container, target_item, global_pos, target_orn_idx)
            if is_valid:
                # ------ 3. spawn into the physics world and execute the placement ------
                is_placed_safe = self.validator.place_item(target_item, global_pos, target_orn_idx)
                if is_placed_safe: # success: consume from the stream and refill
                    self.container_manager.update_and_add_item_to_container(container_id=container_idx, item=target_item)
                    self.stream_manager.pop_and_refill(item_idx)

                else:
                    ## failure: the episode ends
                    terminated = rejected()
            else:
                terminated = rejected()

        ## finish when all bags have been loaded
        if self.stream_manager.is_empty():
            terminated = True

        # -------- 4. output the observation ---------
        observation = self._get_obs()
        if self.verbose:
            print(f'env : {len(self.stream_manager.all_items)}, {len(self.stream_manager.visible_pool)}(elapsed time {round(time.time()-start,3)} sec)')
        if self.render_mode=='human':
            time.sleep(1.0)

        # if the visualisation flag is true, save a snapshot
        if self.visualize:
            img = Image.fromarray(self.visualizer.get_rgb_image())
            img.save(os.path.join(self.image_dir, f"{self.num_step:04}.png"))
        
        self.num_step += 1

        return observation, reward, terminated, truncated, {'status': {'is_included': is_included, 'is_valid': is_valid, 'is_placed_safe': is_placed_safe}}


    def evaluate(self):
        evaluation_report = self.evaluator.evaluate(self.container_manager.containers, self.num_total_items)

        return evaluation_report


    def _get_obs(self):
        """Get the current observation (bag information) and return it as a dict"""

        for container in self.container_manager.containers:
            # move the camera in front of the target container
            lwh = container.length, container.width, container.height
            
            # take the image and write it directly over channel i of the shared memory
            depth_map = self.camera.get_heightmap(container.center, lwh)
            np.copyto(self.shm_depth_map[container.index], depth_map)

        return {
            'shm_name': self.shm.name,
            'shm_shape': self.shm_shape,
            'shm_dtype': 'float32',
            'optimize': self.optimize,
            'lookahead_k': self.stream_manager.lookahead_k,
            'container_list': self.container_manager.get_item_info_in_containers(),
            'pool_list': self.stream_manager.get_items_of_visible_pool()
        }


    def close(self):
        print('close env.')
        self.client.disconnect()
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink()
