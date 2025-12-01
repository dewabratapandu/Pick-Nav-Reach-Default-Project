import numpy as np
import heapq
import math
import pybullet as p


class PathPlanner:
    def __init__(self, maze_info, n_rows, n_cols, cell_side_size = 1, robot_buffer_radius = 0, table_aabb = ()):
        self.cell_side_size = cell_side_size
        self.n_rows = np.floor(n_rows/cell_side_size).astype(int)
        self.n_cols = np.floor(n_cols/cell_side_size).astype(int)
        self.min_x = 0
        self.min_y = 0
        self.maze_info = maze_info
        self.table_aabb = table_aabb
        self.robot_buffer_cells = np.ceil(robot_buffer_radius/cell_side_size).astype(int)
        self.occupancy_map = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)

    def generate_map(self):
        if self.maze_info is None or len(self.maze_info) == 0:
            return None

        #finding min x & y
        min_x = 0
        min_y = 0
        for block in self.maze_info:
            x = block[0]
            if x < min_x:
                min_x = x
            y = block[1]
            if y < min_y:
                min_y = y

        self.min_x = np.floor(min_x)
        self.min_y = np.floor(min_y)

        #covering all block with respect to cell_side_size
        for block in self.maze_info:
            x = (np.floor(block[0]) - self.min_x).astype(int)
            y = (np.floor(block[1]) - self.min_y).astype(int)
            for i in range(np.floor(x/self.cell_side_size).astype(int) - self.robot_buffer_cells, np.floor((x+1)/self.cell_side_size).astype(int) + self.robot_buffer_cells):
                for j in range(np.floor(y / self.cell_side_size).astype(int) - self.robot_buffer_cells,
                               np.floor((y + 1) / self.cell_side_size).astype(int) + self.robot_buffer_cells):
                    if not self.is_in_bounds(i, j):
                        continue
                    self.occupancy_map[i][j] = 1

        #add table info in map
        if len(self.table_aabb) == 2:
            #rebase with min x & y values
            t_min_x = np.floor((self.table_aabb[0][0] - self.min_x)/self.cell_side_size).astype(int)
            t_max_x = np.floor((self.table_aabb[1][0] - self.min_x)/self.cell_side_size).astype(int)

            t_min_y = np.floor((self.table_aabb[0][1] - self.min_y)/self.cell_side_size).astype(int)
            t_max_y = np.floor((self.table_aabb[1][1] - self.min_y)/self.cell_side_size).astype(int)
            print(f"Env min_x: {self.min_x} | Env min_y: {self.min_y}")
            print(f"Table AABB: {self.table_aabb} | t_min_x: {t_min_x} | t_max_x: {t_max_x} | t_min_y: {t_min_y} | t_max_y: {t_max_y}")

            for i in range(t_min_x - self.robot_buffer_cells, t_max_x + self.robot_buffer_cells + 1):
                for j in range(t_min_y - self.robot_buffer_cells, t_max_y + self.robot_buffer_cells + 1):
                    if not self.is_in_bounds(i, j):
                        continue
                    self.occupancy_map[i][j] = 1
        print(f"occupancy map: {self.occupancy_map.tolist()}")

    def dijkstra_2d(self, start_rc, goal_rc, allow_diagonal=False):
        """
        Compute shortest path on a 2D occupancy grid using Dijkstra's algorithm.

        Parameters
        ----------
        occupancy_grid : 2D list or np.ndarray
            0 = free cell, 1 = obstacle
        start_rc : tuple (row, col)
            Robot start cell in grid coordinates.
        goal_rc : tuple (row, col)
            Goal cell in grid coordinates.
        allow_diagonal : bool
            If True, allow 8-connected neighbors; otherwise 4-connected.

        Returns
        -------
        path : list of (row, col)
            Shortest path from start to goal, including both.
            Empty list if no path exists.
        """
        occupancy_grid = self.occupancy_map
        rows, cols = occupancy_grid.shape

        def in_bounds(r, c):
            return 0 <= r < rows and 0 <= c < cols

        def is_free(r, c):
            return occupancy_grid[r][c] == 0

        # 4-connected (up, down, left, right)
        neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 8-connected (includes diagonals)
        neighbors_8 = neighbors_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        neighbors = neighbors_8 if allow_diagonal else neighbors_4

        #normalize co-ordinates
        #TODO: if need to figure out this
        start = self._normalize_xy(start_rc[0], start_rc[1])
        goal = self._normalize_xy(goal_rc[0], goal_rc[1])

        start_occupancy = occupancy_grid[start[0], start[1]]
        goal_occupancy = occupancy_grid[goal[0], goal[1]]

        # Distance map: initialize with +∞
        dist = [[math.inf] * cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0.0

        # Predecessor map for path reconstruction
        prev = [[None] * cols for _ in range(rows)]

        # Min-heap priority queue: (distance, (row, col))
        heap = [(0.0, start)]

        while heap:
            current_dist, (r, c) = heapq.heappop(heap)

            # Early exit if we reached the goal
            if (r, c) == goal:
                break

            # If this is an outdated entry, skip
            if current_dist > dist[r][c]:
                continue

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not is_free(nr, nc):
                    continue

                # Cost: 1 for orthogonal moves, sqrt(2) for diagonals (if enabled)
                step_cost = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                new_dist = current_dist + step_cost

                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    prev[nr][nc] = (r, c)
                    heapq.heappush(heap, (new_dist, (nr, nc)))

        # Reconstruct path
        path = []
        if dist[goal[0]][goal[1]] == math.inf:
            # No path found
            return path

        cur = goal
        while cur is not None:
            r, c = cur
            # cast to plain Python ints to avoid np.int64 in output
            path.append((int(r), int(c)))
            cur = prev[int(r)][int(c)]

        path.reverse()  # from start → goal

        dummy_path = path.copy()
        dummy_path = self.grid_to_world(dummy_path)
        #base shift path with respect to the robot init position
        #TODO: clean up
        path = self._base_shift_path(path, start)
        return dummy_path

    def follow_path(self, path, curr_idx, curr_robot_pos, starting_offset, waypoint_tol=0.02):

        #find joint values
        #check if current position is within waypoint_tol of current index
        #update to next
        path_completed = False
        if curr_idx < len(path):
            x = curr_robot_pos[0]
            y = curr_robot_pos[1]
            z_yaw = curr_robot_pos[2]

            #target x,y
            (wx, wy) = path[curr_idx]
            #offset to make it relative to starting position
            wx = wx - starting_offset[0]
            wy = wy - starting_offset[1]

            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)

            if dist <= waypoint_tol:
                #move to next point
                curr_idx = curr_idx + 1
                if curr_idx < len(path):
                    (wx, wy) = path[curr_idx]
                    #offset to make it relative to starting position
                    wx = wx - starting_offset[0]
                    wy = wy - starting_offset[1]
                else:
                    path_completed = True
                    return curr_robot_pos, curr_idx, path_completed

            new_robot_pos = curr_robot_pos.copy()
            #TODO: maybe need to slow down the robot here
            new_robot_pos[0] = wx
            new_robot_pos[1] = wy
            #keep the robot pointed in the movement direction
            target_yaw = math.atan2(wy - y, wx - x)
            new_robot_pos[2] = target_yaw
            return new_robot_pos, curr_idx, path_completed
        else:
            return curr_robot_pos, curr_idx, True

    def _normalize_xy(self, x , y, recenter=False):
        x = np.floor((x - self.min_x)/self.cell_side_size).astype(int)
        y = np.floor((y - self.min_y)/self.cell_side_size).astype(int)
        if recenter:
            return self._recenter_xy(x, y)
        return (x,y)

    def _recenter_xy(self, x, y):
        no_child_cells = 1/self.cell_side_size
        offset = np.floor(no_child_cells/2).astype(int)
        x = x + offset
        y = y + offset
        return (x,y)

    # base shift using start position of the base

    def _base_shift_path(self, path, start_pos):
        for i in range(len(path)):
            x = path[i][0] - start_pos[0]
            y = path[i][1] - start_pos[1]
            path[i] = (x,y)
        return path

    def grid_to_world(self, path):
        for i in range(len(path)):
            offset = self.cell_side_size/2
            x = self.min_x + (path[i][0]) * self.cell_side_size + offset
            y = self.min_y + (path[i][1]) * self.cell_side_size + offset
            path[i] = (round(x,2), round(y,2))
        return path

    #TODO: replace the other inline func
    def is_in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols



#move to utils
def get_full_robot_aabb(robot_id):
    num_joints = p.getNumJoints(robot_id)

    # Start with AABB of link 0 (base)
    aabb_min, aabb_max = p.getAABB(robot_id)

    min_x, min_y, min_z = aabb_min
    max_x, max_y, max_z = aabb_max

    # Loop all links
    for i in range(num_joints):
        aabb_min_i, aabb_max_i = p.getAABB(robot_id, i)
        min_x = min(min_x, aabb_min_i[0])
        min_y = min(min_y, aabb_min_i[1])
        min_z = min(min_z, aabb_min_i[2])
        max_x = max(max_x, aabb_max_i[0])
        max_y = max(max_y, aabb_max_i[1])
        max_z = max(max_z, aabb_max_i[2])

    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def get_robot_footprint(robot_id):
    aabb_min, aabb_max = get_full_robot_aabb(robot_id)
    width = aabb_max[0] - aabb_min[0]
    depth = aabb_max[1] - aabb_min[1]
    height = aabb_max[2] - aabb_min[2]
    return width, depth, height








