"""

Grid based Dijkstra planning

author: Atsushi Sakai(@Atsushi_twi)

modified by Burla Nur Korkmaz, Aras Umut Erarslan, Ertuğrul Bayraktar, Numan Çelebi

"""

import matplotlib.pyplot as plt
import math

show_animation = True

test_array_x = [] #ekledim
test_array_y = [] #ekledim

class Dijkstra:

    def __init__(self, ox, oy, resolution, robot_radius):
        """
        Initialize map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent = parent  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        iteration_count = 0

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while 1:
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                # plt.plot(self.calc_position(current.x, self.min_x),
                #          self.calc_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                
                test_array_x.append(self.calc_position(current.x, self.min_x)) #ekledim
                test_array_y.append(self.calc_position(current.y, self.min_y)) #ekledim
                
                plt.plot(test_array_x, #ekledim
                          test_array_y, "xc") #ekledim
                
                iteration_count = iteration_count + 1 #ekledim
                print(iteration_count) #ekledim
                
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent = current.parent
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                #ekledim
                cost = 0
                if(current.x + move_x >= 40 and (current.x + move_x) <= 120 and current.y + move_y >= 0 and current.y + move_y <= 60):
                    cost = 12
                if(current.x + move_x >= 100 and (current.x + move_x) <= 170 and current.y + move_y >= 100 and current.y + move_y <= 125):
                    cost = 6
                #ekledim
                
                if(current.x + move_x >= 0 and (current.x + move_x) <= 250 and current.y + move_x >= 120 and current.y + move_x <= 130):
                    cost = 11.573
                if(current.x + move_x >= 0 and (current.x + move_x) <= 10 and current.y + move_x >= 120 and current.y +move_x <= 225):
                    cost = 11.573
                if(current.x + move_x >= 105 and (current.x + move_x) <= 130 and current.y + move_x >= 215 and current.y + move_x <= 225):
                    cost = 11.573
                
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost + cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent = goal_node.parent
        while parent != -1:
            n = closed_set[parent]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent = n.parent

        return rx, ry

    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 1.0  # [m]
    sy = 125.0  # [m]
    gx = 222.0  # [m]
    gy = 125.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []   
    ox, oy = [], []
    for i in range(0, 250):
        ox.append(i)
        oy.append(0.0)
        
        test_array_x.append(i)
        test_array_y.append(0)
    for i in range(0, 250):
        ox.append(250.0)
        oy.append(i)
        
        test_array_x.append(250)
        test_array_y.append(i)
    for i in range(0, 251):
        ox.append(i)
        oy.append(250.0)
        
        test_array_x.append(i)
        test_array_y.append(250.0)
    for i in range(0, 251):
        ox.append(0.0)
        oy.append(i)
        
        test_array_x.append(0)
        test_array_y.append(i)
        
        
        
        
        
        
    for i in range(10, 117):
        ox.append(i)
        oy.append(130.0)
        
        test_array_x.append(i)
        test_array_y.append(130.0)    
        
    for i in range(127, 215):
        ox.append(i)
        oy.append(130.0)
        
        test_array_x.append(i)
        test_array_y.append(130.0)
        
    for i in range(10, 117):
        ox.append(i)
        oy.append(120.0)
        
        test_array_x.append(i)
        test_array_y.append(120.0)    
        
    for i in range(127, 225):
        ox.append(i)
        oy.append(120.0)
        
        test_array_x.append(i)
        test_array_y.append(120.0)  
        
        
        
        
    for i in range(70, 120):
        ox.append(10.0)
        oy.append(i)
        
        test_array_x.append(10)
        test_array_y.append(i)
        
        
    for i in range(10, 117):
        ox.append(i)
        oy.append(70.0)
        
        test_array_x.append(i)
        test_array_y.append(70)
        
    for i in range(127, 225):
        ox.append(i)
        oy.append(70.0)
        
        test_array_x.append(i)
        test_array_y.append(70)
        
        
    for i in range(0, 235):
        ox.append(i)
        oy.append(60.0)
        
        test_array_x.append(i)
        test_array_y.append(60)
        
    
        
    
    for i in range(130, 210):
        ox.append(10.0)
        oy.append(i)
        
        test_array_x.append(10)
        test_array_y.append(i)
    
    for i in range(10, 117):
        ox.append(i)
        oy.append(210.0)
        
        test_array_x.append(i)
        test_array_y.append(210)
        
    for i in range(127, 215):
        ox.append(i)
        oy.append(210.0)
        
        test_array_x.append(i)
        test_array_y.append(210)
        
    for i in range(0, 225):
        ox.append(i)
        oy.append(220.0)
        
        test_array_x.append(i)
        test_array_y.append(220)
        
        
        
        
        
    for i in range(70, 120):
        ox.append(117.0)
        oy.append(i)
        
        test_array_x.append(117)
        test_array_y.append(i)
    
    for i in range(130, 210):
        ox.append(117.0)
        oy.append(i)
        
        test_array_x.append(117)
        test_array_y.append(i)
        
    for i in range(70, 120):
        ox.append(127.0)
        oy.append(i)
        
        test_array_x.append(127)
        test_array_y.append(i)
    
    for i in range(130, 210):
        ox.append(127.0)
        oy.append(i)
        
        test_array_x.append(127)
        test_array_y.append(i)
    
    for i in range(70, 120):
        ox.append(225.0)
        oy.append(i)
        
        test_array_x.append(225)
        test_array_y.append(i)
        
    for i in range(60, 120):
        ox.append(235.0)
        oy.append(i)
        
        test_array_x.append(235)
        test_array_y.append(i)
    
    for i in range(130, 210):
        ox.append(215.0)
        oy.append(i)
        
        test_array_x.append(215)
        test_array_y.append(i)
        
    for i in range(130, 220):
        ox.append(225.0)
        oy.append(i)
        
        test_array_x.append(225)
        test_array_y.append(i)
        
    for i in range(225, 250):
        ox.append(i)
        oy.append(130.0)
        
        test_array_x.append(i)
        test_array_y.append(130)
        
    for i in range(235, 250):
        ox.append(i)
        oy.append(120.0)
        
        test_array_x.append(i)
        test_array_y.append(120)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()
