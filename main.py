def main():
    print("Hello from chip-router-mk2!")


if __name__ == "__main__":
    main()


import numpy as np

class ChipPlacementEnv:
    def __init__(self, width, height, netlist, component_metadata):
        # 1. Physical Grid Setup
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width))
        
        # 2. Data Structures
        self.netlist = netlist  # Dictionary: {node_id: [neighbors]}
        self.comp_info = component_metadata # {node_id: {'w': w, 'h': h}}
        
        # 3. State Tracking
        self.placed_nodes = {} # {node_id: (x, y)}
        self.queue = list(netlist.keys())
        self.current_step = 0

    def _get_manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_reward(self, node_id, x, y):
        total_wirelength = 0
        neighbors = self.netlist.get(node_id, [])
        
        for neighbor in neighbors:
            if neighbor in self.placed_nodes:
                neighbor_pos = self.placed_nodes[neighbor]
                total_wirelength += self._get_manhattan_dist((x, y), neighbor_pos)
        
        # We return negative wirelength as the reward
        return -total_wirelength

    def step(self, action):
        """
        action: (x, y) coordinate for the top-left anchor
        """
        node_id = self.queue[self.current_step]
        dims = self.comp_info[node_id]
        x, y = action
        
        # --- HARD CONSTRAINT CHECK ---
        # 1. Boundary check 2. Overlap check
        if (x + dims['w'] > self.width or y + dims['h'] > self.height or 
            np.any(self.canvas[y : y+dims['h'], x : x+dims['w']] == 1)):
            
            return None, -100, True # Penalty and end game
        
        # --- SUCCESSFUL PLACEMENT ---
        self.canvas[y : y+dims['h'], x : x+dims['w']] = 1
        self.placed_nodes[node_id] = (x, y)
        
        reward = self._calculate_reward(node_id, x, y)
        self.current_step += 1
        done = self.current_step >= len(self.queue)
        
        return self.get_observation(), reward, done

    def get_observation(self):
        # This is where we will combine the CNN (canvas) 
        # and the GNN (node features) in the next step.
        return {"canvas": self.canvas, "current_node": self.queue[self.current_step] if self.current_step < len(self.queue) else None}

# Example Usage:
# netlist = {0: [1, 2], 1: [0], 2: [0]}
# metadata = {0: {'w': 2, 'h': 2}, 1: {'w': 1, 'h': 1}, 2: {'w': 1, 'h': 1}}
# env = ChipPlacementEnv(10, 10, netlist, metadata)