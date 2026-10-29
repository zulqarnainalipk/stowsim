import numpy as np

class Agent():
    def __init__(self, module_path: str):
        # initialise any parameters you need
        pass

    def get_init_states(self, init_states: dict):
        self.num_containers = len(init_states.get('container_list', []))
        return True

    def optimize(self, item_list: list):
        # example: sort by volume (largest first) and return the indices
        sorted_items = sorted(item_list, key=lambda x: x.get('volume', x['length']*x['width']*x['height']), reverse=True)
        return [item['index'] for item in sorted_items]

    def policy(self, observation: dict):
        # infer a placement from the observation
        depth_maps = observation.get('depth_map')
        pool_list = observation.get('pool_list', [])
        
        # ... decision logic ...

        action = {
            'item_idx': 0,
            'container_idx': 0,
            'place_pos': np.array([0.1, 0.1, 0.3], dtype=np.float32),
            'orientation': 0
        }
        return action
