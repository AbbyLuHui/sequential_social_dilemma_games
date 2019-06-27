"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
import utility_funcs as util
import random


# basic moves every agent should do
BASE_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY',  # don't move
                5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                6: 'TURN_COUNTERCLOCKWISE'}  # Rotate clockwise


class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, norm, row_size, col_size):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        grid: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        # TODO(ev) change grid to env, this name is not very informative
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size
        self.reward_this_turn = 0
        self.norm = norm


    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_state(self):
        return util.return_view(self.grid, self.get_pos(),
                                self.row_size, self.col_size)

    def compute_reward(self):
        reward = self.reward_this_turn
        self.reward_this_turn = 0
        return reward

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.get_pos()
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid

    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        return temp_pos

    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions

        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.get_pos()
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        self.set_pos(temp_pos)
        # TODO(ev) list array consistency
        return self.get_pos(), np.array(old_pos)

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError

    def set_norm(self, social_norm, allowed):
        self.norm[social_norm] = allowed


HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: 'FIRE'})  # Fire a penalty beam

HARVEST_VIEW_SIZE = 7


class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=HARVEST_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(8)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char


        


CLEANUP_ACTIONS = BASE_ACTIONS.copy()

CLEANUP_ACTIONS.update({7: 'FIRE',  # Fire a penalty beam
                        8: 'CLEAN'})  # Fire a cleaning beam

CLEANUP_VIEW_SIZE = 7


class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, norm, view_len=CLEANUP_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, norm, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        elif char == 'D':
            return ' '
        else:
            return char

    def dist_to_apples(self, x, y, goalx, goaly):
        return abs(x-goalx) + abs(y-goaly)
    
    
    def find_goal(self, obs, x, y):
        min_coordinate=[float('inf'), float('inf')]    #find the level 0 goal of the current agent
        for item in obs:
            a = abs(x-item[0]) + abs(y-item[1])
            b = abs(x-min_coordinate[0]) + abs(y-min_coordinate[1])

            if a<b:
                min_coordinate[0]=item[0]
                min_coordinate[1]=item[1]
        return min_coordinate

    def find_final_goal(self, obs, x, y, agent_locs, depth):
        if depth == 0 :
            min_coordinate = self.find_goal(obs, x, y)
            return min_coordinate

        else:
            # Initialize with my distance to all possible goals
            goal_dist = {goal : self.dist_to_apples(x, y, goal[0], goal[1]) for goal in obs}
            for item in agent_locs:
                other_x = item[0]
                other_y = item[1]
                #print(agent_locs)
                #print(other_x, "  ", other_y)
                other_goal = self.find_final_goal(obs, other_x, other_y, agent_locs, depth-1)
                #print("other goal: ", other_goal, "other position:  ", (other_x, other_y))
                if other_goal == None:
                    other_dist = float('inf')
                else:
                    other_dist = self.dist_to_apples(other_x, other_y, other_goal[0], other_goal[1])
                    #print("other distance: ", other_dist)
                #print("agent id:  ", self.agent_id)
                #print("self.pos: ", self.pos)
                try:
                    # If other is closer to goal than I am
                    if other_dist < goal_dist[(other_goal[0], other_goal[1])]:
                        # Then goal is unreachable by me
                        goal_dist[(other_goal[0], other_goal[1])] = float('inf')
                except:
                    pass
            try:
                return min(goal_dist,key=goal_dist.get)
            except:
                return None




    def determine_action(self, x, y):
        allowed_norms = []
        #get a list of allowed norms
        for item in self.norm:
            if self.norm[item]:
                allowed_norms.append(item)
        #maze marking with 1 and 0: 1-norms not allowed, 0-allowed norms
        maze = []
        for row_elem in range(self.grid.shape[0]):
            row = []
            for column_elem in range(self.grid.shape[1]):
                if self.grid[row_elem][column_elem] == ' ' or self.grid[row_elem][column_elem] in allowed_norms:
                    row.append(0)
                else:
                    row.append(1)
            maze.append(row)
        path = astar(maze, (self.pos[0], self.pos[1]), (x, y))
        try:
            (x_coor, y_coor) = path[1]
        except:
            return 4

        if (y_coor!= self.pos[1]):
            action = 2 if self.pos[1]>y_coor else 3
        else:
            action = 0 if self.pos[0]>x_coor else 1
        return action
        # 3- go right; 2 - go left; 1 - go down; 0 - go up;


    
    def policy(self, depth):
        #get norms
        allowed_norms = []
        for elem in self.norm:
            if self.norm[elem]:
                allowed_norms.append(elem)

        #obtain apple and agent locations
        apple_locs = []
        agent_locs = []
        for row_elem in range(self.grid.shape[0]):
            for column_elem in range(self.grid.shape[1]):
                if self.grid[row_elem][column_elem] in allowed_norms:
                    apple_locs.append((row_elem, column_elem))
                elif self.grid[row_elem][column_elem] in '123456789P':
                    agent_locs.append((row_elem, column_elem))

        min_coordinate=self.find_final_goal(apple_locs, self.pos[0], self.pos[1], agent_locs,depth)
        if min_coordinate == None:
            return 4
        else:
            return self.determine_action(min_coordinate[0], min_coordinate[1])




#astar search algorithm
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)


    loop_cycle = 0
    # Loop until you find the end
    while len(open_list) > 0 and loop_cycle < 300:
        loop_cycle+=1
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path
        if loop_cycle == 299:
            return None

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = (abs(child.position[0] - end_node.position[0])) + (abs(child.position[1] - end_node.position[1]))
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

