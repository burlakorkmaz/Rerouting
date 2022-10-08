"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

modified by Burla Nur Korkmaz, Aras Umut Erarslan, Ertuğrul Bayraktar, Numan Çelebi

"""

import math

import matplotlib.pyplot as plt

show_animation = True

test_array_x = [] #ekledim
test_array_y = [] #ekledim


class AStarPlanner:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        iteration_count = 0 #ekledim
        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart
        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                # plt.plot([self.calc_grid_position(current.x, self.minx)], #ekledim
                #          self.calc_grid_position(current.y, self.miny), "xc") #ekledim
                
                # test_array_x.append(self.calc_grid_position(current.x, self.minx)) #ekledim
                # test_array_y.append(self.calc_grid_position(current.y, self.miny)) #ekledim
                
                # plt.plot(test_array_x, #ekledim
                #          test_array_y, "xc") #ekledim
                iteration_count=iteration_count + 1 #ekledim
                print(iteration_count) #ekledim
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                #ekledim
                cost = 0
                if(current.x + self.motion[i][0] >= 40 and (current.x + self.motion[i][0]) <= 120 and current.y + self.motion[i][1] >= 0 and current.y + self.motion[i][1] <= 60):
                    cost = 12
                if(current.x + self.motion[i][0] >= 100 and (current.x + self.motion[i][0]) <= 170 and current.y + self.motion[i][1] >= 100 and current.y + self.motion[i][1] <= 125):
                    cost = 6
                #ekledim
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2] + cost, c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closed_set)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_grid_position(ngoal.x, self.minx)], [
            self.calc_grid_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, minp):
        """
        calc grid position

        :param index:
        :param minp:
        :return:
        """
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        # collision check
        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx) / self.reso)
        self.ywidth = round((self.maxy - self.miny) / self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
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
    sx = 5.0  # [m]
    sy = 190.0  # [m]
    gx = 195.0  # [m]
    gy = 5.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []   
    for i in range(0, 200):
        ox.append(i)
        oy.append(0.0)
        test_array_x.append(i)
        test_array_y.append(0)
    for i in range(0, 200):
        ox.append(200.0)
        oy.append(i)
        test_array_x.append(200)
        test_array_y.append(i)
    for i in range(0, 201):
        ox.append(i)
        oy.append(200.0)
        test_array_x.append(i)
        test_array_y.append(200)
    for i in range(0, 201):
        ox.append(0.0)
        oy.append(i)
        test_array_x.append(0)
        test_array_y.append(i)
    for i in range(10,60):
        ox.append(i)
        test_array_x.append(i)
    for i in range(198,148,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(60,112):
        ox.append(i)
        test_array_x.append(i)
    for i in range(148,200):
        oy.append(i)
        test_array_y.append(i)
    for i in range(175,115,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(0,60):
        ox.append(i)
        test_array_x.append(i)
    for i in range(60,0,-1):
        ox.append(i)
        test_array_x.append(i)
    for i in range(115,55,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(40,0,-1):
        ox.append(i)
        test_array_x.append(i)
    for i in range(65,25,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(40,105):
        ox.append(i)
        test_array_x.append(i)
    for i in range(65,0,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(100,75,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(100,75,-1):
        ox.append(i)
        test_array_x.append(i)
    for i in range(75,120):
        ox.append(i)
        test_array_x.append(i)
    for i in range(75,30,-1):
        oy.append(i)
        test_array_y.append(i)
    for i in range(120,170):
        ox.append(i)
        oy.append(30)
        test_array_x.append(i)
        test_array_y.append(30)
    for i in range(100,170):
        ox.append(i)
        oy.append(100)
        test_array_x.append(i)
        test_array_y.append(100)
    for i in range(100,30,-1):
        oy.append(i)
        ox.append(170)
        test_array_x.append(170)
        test_array_y.append(i)
    for i in range(100,170):
        ox.append(i)
        oy.append(125)
        test_array_x.append(i)
        test_array_y.append(125)
    for i in range(100,150):
        ox.append(i)
        test_array_x.append(i)
    for i in range(125,175):
        oy.append(i)
        test_array_y.append(i)
    for i in range(150,170):
        ox.append(i)
        oy.append(175)
        test_array_x.append(i)
        test_array_y.append(175)
    for i in range(175,125,-1):
        oy.append(i)
        ox.append(170)
        test_array_x.append(170)
        test_array_y.append(i)
        
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
