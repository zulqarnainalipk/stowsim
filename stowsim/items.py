"""Bag (cuboid) physics properties and the bag stream with its visible pool."""
import pybullet as p
from dataclasses import dataclass
from pybullet_utils.bullet_client import BulletClient


@dataclass
class Item:
    """Class that manages the physical properties of a single bag (cuboid) and its body in PyBullet"""
    index: int
    length: float
    width: float
    height: float
    mass: float = 1.0
    is_prioritized: bool = False
    is_soft: bool = False
    belongs_to: int | None = None          # container it is placed in

    pos: tuple[float, float, float] | None = None           # position in the world frame
    orn: tuple[float, float, float, float] | None = None    # quaternion describing the orientation

    lateralFriction: float = 0.8      # lateral friction: prevents sliding (close to the value of cardboard)
    rollingFriction: float = 0.01     # rolling friction: prevents unnatural rolling about a corner
    spinningFriction: float = 0.01    # spinning friction: prevents rotational drift at the contact surface
    restitution: float = 0.0          # restitution: no bouncing on landing, the energy is absorbed
    angularDamping: float = 0.8

    # only effective when is_soft=True
    contactStiffness: float = 5000    # contact stiffness (the lower it is, the more the bag sinks in like a cushion)
    contactDamping: float = 500       # contact damping (suppresses the wobble after sinking in)
    linearDamping: float = 0.8        # damping of the translational motion (the higher it is, the sooner it stops sliding)


    def __post_init__(self):
        self.color: list[float] = [0.8, 0.6, 0.4, 1.0] if not self.is_prioritized else [0.0, 1.0, 0.0, 1.0]
        if self.is_soft:
            self.color = [0, 0, 1, 1]

        self.volume: float = self.length * self.width * self.height
        self.pybullet_id: int | None = None


    def get_info(self) -> dict[str, int | float | tuple ]:
        """Get the latest position and orientation in the physics world"""
        info = {
            'index': self.index,
            'length': self.length,
            'width': self.width,
            'height': self.height,
            'mass': self.mass,
            'is_prioritized': self.is_prioritized,
            'is_soft': self.is_soft,
            'belongs_to': self.belongs_to,
            'pos': self.pos,
            'orn': self.orn,
            'lateralFriction': self.lateralFriction,
            'rollingFriction': self.rollingFriction,
            'spinningFriction': self.spinningFriction,
            'restitution': self.restitution,
            'angularDamping': self.angularDamping
        }
        if self.is_soft:
            info.update({
                'contactStiffness': self.contactStiffness,
                'contactDamping': self.contactDamping,
                'linearDamping': self.linearDamping
            })

        return info


    def spawn(self, client: BulletClient, initial_pos: tuple[float, float, float] = (0.0, 0.0, 5.0), initial_orn: tuple[float, float, float, float] = (0,0,0,1)) -> int | None:
        """Spawn the bag in the PyBullet physics world. A body ID in the physics world is assigned."""
        
        # PyBullet box creation takes the half size from the centre, not the full size (length)
        half_extents = [self.length / 2, self.width / 2, self.height / 2]
        
        # 1. create the shape used for collision detection (collision shape)
        col_id = client.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        
        # 2. create the shape used for rendering (visual shape)
        vis_id = client.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=self.color)
        
        # 3. combine mass and shapes and create the body in the physics world
        self.pybullet_id = client.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=initial_pos,
            baseOrientation=initial_orn
        )
        
        # 4. tune the physical parameters for stable loading
        dynamics_params = {
            'lateralFriction': self.lateralFriction,
            'rollingFriction': self.rollingFriction,
            'spinningFriction': self.spinningFriction,
            'restitution': self.restitution,
            'angularDamping': self.angularDamping
        }
        if self.is_soft:
            dynamics_params.update(
                {
                    'contactStiffness': self.contactStiffness,
                    'contactDamping': self.contactDamping,
                    'linearDamping': self.linearDamping  # overwrite the base value
                }
            )

        client.changeDynamics(
            self.pybullet_id, 
            -1, # -1 refers to the base link (the bag itself)
            **dynamics_params
        )

        text_label = f"Item {self.index}"
        
        # specify, in local coordinates, a position shifted slightly upwards (Z) from the bag centre
        # shown at half the bag height + 5 cm above the centre
        local_pos = [0, 0, (self.height / 2) + 0.05]
        
        self.debug_text_id = client.addUserDebugText(
            text=text_label,
            textPosition=local_pos,
            textColorRGB=[1.0, 1.0, 1.0],  # white text
            textSize=1.0,                  # text size
            parentObjectUniqueId=self.pybullet_id, # follows the bag
            parentLinkIndex=-1             # attached to the base link
        )

        return self.pybullet_id


    def remove(self, client: BulletClient):
        """Remove the spawned bag from the physics world"""
        if self.pybullet_id is not None:
            client.removeBody(self.pybullet_id)
            self.pybullet_id = None
            self.pos = None
            self.orn = None


    def set_pose(self, client: BulletClient, pos: tuple[float, float, float], orn: tuple[float, float, float ,float]):
        """Instantly warp the bag to the given pose without advancing the simulation time"""
        if self.pybullet_id is not None:
            client.resetBasePositionAndOrientation(self.pybullet_id, pos, orn)


    def get_pose(self, client: BulletClient) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | tuple[None, None]:
        """Get the current position (pos) and orientation (orn: quaternion)"""
        if self.pybullet_id is not None:
            pos, orn = client.getBasePositionAndOrientation(self.pybullet_id)
            return pos, orn
        return None, None


    def register_pos_orn(self, container_index: int, pos: tuple[float, float, float], orn: tuple[float, float, float, float]):
        self.belongs_to = container_index
        self.pos = pos
        self.orn = orn # quaternion


class ItemStreamManager:
    """Class that holds the information of all bags in advance and manages and supplies k bags on the conveyor as a pool"""
    
    def __init__(self, config: dict):
        """
        config:
            - item_list: list[Item], list of all bags
            - lookahead: int, maximum number of visible bags
            - visible_pool: list[Item], list of the bags that are pooled and visible
        """
        self.all_items: list[Item] = [Item(**item_config) for item_config in config['item_list']]
        self.total_items: int = len(self.all_items)
        self.all_indices: set = {item.index for item in self.all_items}
        self.lookahead_k: int = config['look_ahead']
        self.max_space: int = config['max_space'] # number of remaining bags that decides when to refill
        assert self.lookahead_k >= self.max_space

        self.visible_pool: list[Item] = [Item(**item_config) for item_config in config['visible_pool']]
        self.current_index: int = 0 # index in all_items of the next bag to add to the pool


    def set_order(self, order: list[int]) -> None:
        order_map = {v: i for i, v in enumerate(order)}
        self.all_items.sort(key=lambda x: order_map[x.index])


    def get_items_of_visible_pool(self) -> list[dict]:
        info_list = [
            item.get_info() for item in self.visible_pool
        ]
        return info_list


    def get_all_items(self) -> list[dict]:
        info_list = [
            item.get_info() for item in self.all_items
        ]
        return info_list


    def reset(self):
        """Reset the stream at the start of an episode and put the first k bags into the pool"""
        self.current_index = 0
        self.visible_pool = []
        for _ in range(self.lookahead_k):
            self._refill_item()


    def _refill_item(self):
        """Refill the pool with a new bag from the list of all bags"""
        if self.current_index < self.total_items:
            item = self.all_items[self.current_index]
            self.visible_pool.append(item)
            self.current_index += 1


    def get_item(self, pool_index: int) -> Item | None:
        """Get the item object at the given index in the pool"""
        if 0 <= pool_index < len(self.visible_pool):
            return self.visible_pool[pool_index]
        return None
        

    def pop_and_refill(self, pool_index: int) -> None:
        """
        Update the state of the pool (when a bag is taken or when refilling)
        After a bag is selected, remove it from the pool and refill with a new bag from the back
        """
        if 0 <= pool_index < len(self.visible_pool):
            self.visible_pool.pop(pool_index)
            # one bag was removed, so add a new bag to the pool
            num_space = self.lookahead_k - len(self.visible_pool)
            if num_space >= self.max_space:
                for _ in range(num_space):
                    self._refill_item()


    def is_empty(self) -> bool:
        """Whether all bags in the pool have been processed and the pool is empty"""
        return len(self.visible_pool) == 0
