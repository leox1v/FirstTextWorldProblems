import IPython
from recordclass import recordclass
import os

if 'FirstTextWorldProblems/ftwp/' in os.path.realpath(__file__):
    # package imports
    from ftwp.utils import flist
else:
    from utils import flist


GraphNode = recordclass('GraphNode', 'location unvisited_directions relative_path came_from closed_doors')

class Navigator:
    """
    The navigator stores all rooms as nodes in a graph and keeps track of the walked paths. It has a neural navigation model
    that takes raw description strings as input and outputs the possible directions to go next and (if existing) all
    closed doors.
    If the Navigators 'explore' command is invoked it determines a direction to go to based on DFS from the current location.
    """
    def __init__(self, navigation_model):
        self.graph = {}
        self.last_movement = None
        self.navigation_model = navigation_model
        self.initialized_graph = False

        self.discovered_locations = []

    def initialize_graph(self, description):
        """
        Initialized the graph based on the description of the first room.
        """
        self.update_graph(location=Navigator.extract_location(description),
                          description=description)
        self.initialized_graph = True

    def update_graph(self, location, description):
        """
        Updates the graph based on the description of the first room.
        """
        if location not in self.graph:
            came_from = self._get_inverse_move(self.last_movement) if self.last_movement is not None else 'start'
            self._add_graph_node(location, came_from, description)

    def get_navigational_commands(self, description):
        cmds = []
        if Navigator.extract_location(description) not in self.discovered_locations:
            self.discovered_locations.append(Navigator.extract_location(description))
        doors_to_open, directions = self.navigation_model(description)
        for door in doors_to_open:
            cmds.append('open {}'.format(door))
        for _dir in directions:
            cmds.append('go {}'.format(_dir))
        return cmds

    def explore(self, description):
        """
        Determines the low level action that needs to be performed to execute one more step of a DFS from the
        current location. Additionally the neural model determines all closed doors in the current location and adds the
        respective open commands.
        """
        location = Navigator.extract_location(description)
        commands = flist()
        self.update_graph(location=location,
                          description=description)

        if len(self.graph[location].closed_doors) > 0:
            commands.append(self.open_doors(location))

        # explore the first unvisited direction
        if len(self.graph[location].unvisited_directions) > 0:
            direction = self.graph[location].unvisited_directions.pop(0)
        else:
            # if no unvisited direction available anymore -> go back to where we came from
            if self.graph[location].came_from == 'start':
                direction = None
            else:
                direction = self.graph[location].came_from

        commands.append(self.do_move(direction))

        if len(commands) == 0:
            # fallback
            commands.append('look')

        return commands

    def do_move(self, direction):
        """
        Keeps track of all made moves. Updates the relative paths of the rooms in the graph.
        """
        if direction is None:
            return []
        self.last_movement = direction
        self._update_relative_path(direction)
        return ['go {}'.format(direction)]

    def open_doors(self, location):
        """
        Adds 'open' commands for all closed doors in the current location.
        """
        cmds = []
        for door in self.graph[location].closed_doors:
            self.graph[location].closed_doors.remove(door)
            cmds.append('open {}'.format(door))
        return cmds


    def _add_graph_node(self, location, came_from, description):
        """
        When a new location is added as a node to the graph, the neural model determines all possible directions to go
        from there as well as all closed doors in the way.
        """
        doors_to_open, directions = self.navigation_model(description)
        if came_from in directions:
            directions.remove(came_from)
        self.graph[location] = GraphNode(location=location,
                                         unvisited_directions=directions,
                                         relative_path=[],
                                         came_from=came_from,
                                         closed_doors=doors_to_open)


    def _update_relative_path(self, command):
        """
        Updates the relative path of each location after a movement command.
        """
        for location, node in self.graph.items():
            relative_path = self.graph[location].relative_path
            relative_path.insert(0, self._get_inverse_move(command))
            relative_path = self._remove_redundant_paths(relative_path)
            self.graph[location].relative_path = relative_path

    def _remove_redundant_paths(self, rp):
        """
        Removes redundancies in the relative paths. E.g. [north, west, east] -> [north]
        """
        def remove(rp):
            if len(rp) >= 2:
                for i in range(len(rp)-1):
                    if rp[i] == self._get_inverse_move(rp[i+1]):
                        return rp[:i] + rp[i+2:], False
            return rp, True
        while True:
            rp, finished = remove(rp)
            if finished:
                break
        return rp

    @staticmethod
    def extract_location(description):
        """
        Exctracts the location from a description string.
        """
        return description.split('-=')[1].split('=-')[0].strip()

    def _get_inverse_move(self, move):
        if move == 'north':
            return 'south'
        elif move == 'east':
            return 'west'
        elif move == 'south':
            return 'north'
        elif move == 'west':
            return 'east'

    def go_to(self, place):
        """
        Returns the relative path from the current location to some other previously visited location.
        """
        if place in self.graph:
            commands = self.graph[place].relative_path.copy()
            return [self.do_move(direction)[0] for direction in commands]
        else:
            print("Didn't find place '{}'".format(place))
            return []
