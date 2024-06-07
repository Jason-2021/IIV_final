import numpy as np
import utils
from fibonacci_heap_python.fib_heap import FibonacciHeap
import argparse
import os
import copy
import time
from scipy.interpolate import BSpline

'''
map status:
    0: unseen
    1: obstacle
    2: visited(end)
    3: seen, but not end
'''


class Dijkstra:
    def __init__(self, map, start, end, draw_mode=0) -> None:
        self.original_map = copy.deepcopy(map)
        self.map = copy.deepcopy(map)
        self.map_size = len(map)
        self.node_map = [[None for j in range(len(map))] for i in range(len(map))]
        self.start = start
        self.end = end
        self.pi = [[None for j in range(len(map))] for i in range(len(map))]
        self.distance = [[None for j in range(len(map))] for i in range(len(map))]
        self.fheap = FibonacciHeap()

        self.route = []
        self.smooth_route = None
        self.exp_time = float()
        self.draw_mode = draw_mode
        
        self.node_map[start[0]][start[1]] = self.fheap.insert(key=0, value=start)
        self.distance[start[0]][start[1]] = 0
    
    
    def add_node_to_fheap(self, now_node, i, j):
        now_x, now_y = now_node.value

        # neibor node
        candidate_x = int()
        candidate_y = int()
        candidate_x = now_x + i
        candidate_y = now_y + j

        # check if candidate is end
        if candidate_x == self.end[0] and candidate_y == self.end[1]:
            self.map[candidate_x][candidate_y] = 2
            self.pi[candidate_x][candidate_y] = now_node.value
            self.distance[candidate_x][candidate_y] = self.distance[now_node.value[0]][now_node.value[1]] + 1
            return None
        
        # check boundary
        if candidate_x < 0 or candidate_x >= self.map_size or candidate_y < 0 or candidate_y >= self.map_size:
            return None
        
        candidate_status = self.map[candidate_x][candidate_y]
        
        # check status
        if candidate_status != 1 and candidate_status != 2:
            if candidate_status == 0:
                self.map[candidate_x][candidate_y] = 3
                self.node_map[candidate_x][candidate_y] = self.fheap.insert(key=self.cost(now_node, candidate_x, candidate_y), value=(candidate_x, candidate_y))
                self.pi[candidate_x][candidate_y] = now_node.value
                self.distance[candidate_x][candidate_y] = self.distance[now_node.value[0]][now_node.value[1]] + 1
                
            elif candidate_status == 3:
                if self.node_map[candidate_x][candidate_y].key > self.cost(now_node, candidate_x, candidate_y):
                    self.fheap.decrease_key(self.node_map[candidate_x][candidate_y], self.cost(now_node, candidate_x, candidate_y))
                    self.pi[candidate_x][candidate_y] = now_node.value
                    self.distance[candidate_x][candidate_y] = self.distance[now_node.value[0]][now_node.value[1]] + 1
            else:
                raise ValueError("Wrong status")
            

    def cost(self, now_node, candidate_x=None, candidate_y=None):
        return now_node.key + 1
    
    
    def find_shortest_path(self):
        start_time = time.time()
        while len(self.fheap) > 0 and self.map[self.end[0]][self.end[1]] != 2:
            now_node = self.fheap.extract_min()
            
            self.map[now_node.value[0]][now_node.value[1]] = 2
            if now_node.value == self.end:
                break

            
            # neibor
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    else:
                        self.add_node_to_fheap(now_node, i, j)
            
        self.exp_time = time.time() - start_time
        
    
    def answer(self):
        return self.distance[self.end[0]][self.end[1]]
    

    def print_node_map(self):
        for r in self.node_map:
            for c in r:
                print(c.key)


    def stastistic(self):
        l = [0,0,0,0]
        for r in self.map:
            for c in r:
                l[c] += 1
        print(f"0: {l[0]}\n1: {l[1]}\n2: {l[2]}\n3: {l[3]}")


    def path_smoothing(self):
        path_points = np.array(self.route)
        # Extract x and y coordinates
        x = path_points[:, 0]
        y = path_points[:, 1]

        # Degree of the spline (quartic)
        degree = 4

        # Number of control points
        num_points = len(x)

        # Knot vector (open uniform)
        knot_vector = np.concatenate(([0] * degree, np.linspace(0, 1, num_points - degree + 1), [1] * degree))

        # Create the B-spline objects for x and y
        spl_x = BSpline(knot_vector, x, degree)
        spl_y = BSpline(knot_vector, y, degree)

        # Generate new points
        t_new = np.linspace(0, 1, 100)
        x_new = spl_x(t_new)
        y_new = spl_y(t_new)
        # return x_new, y_new
        
        self.smooth_route = np.vstack((x_new, y_new)).T
        

    def trace_route(self, filename=None):
        route = [self.end]
        now = self.end
        while now != self.start:
            now = self.pi[now[0]][now[1]]
            route.append(np.array(now))
        
        route.reverse()
        self.route = np.array(route)
        self.path_smoothing()
        
        

    def cal_dis_and_angle(self):
        original_distance = 0
        original_angle = 0
        smooth_distance = 0
        smooth_angle = 0
        
        for i in range(1, len(self.route)):
            original_distance += np.linalg.norm(self.route[i] - self.route[i-1])
            smooth_distance += np.linalg.norm(self.smooth_route[i] - self.smooth_route[i-1])
            if i >=2:
                original_angle += self.angle_between(self.route[i] - self.route[i-1],
                                            self.route[i-1] - self.route[i-2])
                smooth_angle += self.angle_between(self.smooth_route[i] - self.smooth_route[i-1],
                                            self.smooth_route[i-1] - self.smooth_route[i-2])
        
        print(f"Original distance: {original_distance}")
        print(f"Original angle:    {np.rad2deg(original_angle)}")
        print(f"Smooth distance: {smooth_distance}")
        print(f"Smooth angle:    {np.rad2deg(smooth_angle)}")
    

    def angle_between(self, a, b):
        u_a = a / np.linalg.norm(a)
        u_b = b / np.linalg.norm(b)

        return np.arccos(np.clip(np.dot(u_a, u_b), -1.0, 1.0))


class A_star(Dijkstra):
    def __init__(self, map, start, end) -> None:
        super().__init__(map, start, end)
    def cost(self, now_node, candidate_x, candidate_y):
        now_x, now_y = now_node.value
        now_distance = self.distance[now_x][now_y]
        # diagonal heuristic
        dx = abs(self.end[0] - candidate_x)
        dy = abs(self.end[1] - candidate_y)
        h_n = dx + dy + (2**0.5 - 2) * min(dx, dy)

        return (now_distance + 1) + h_n

    def debug(self):
        for i in self.distance:
            print(i)
        print(self.node_map[1][1].key)
        print(self.node_map[1][0].key)
        print(self.node_map[0][1].key)


class Improved_A_star(Dijkstra):
    def __init__(self, map, start, end, alpha=-0.04, beta=0.5, d0=15) -> None:
        super().__init__(map, start, end)
        self.obstacles = self.find_obstacle()
        self.e_x = np.array([1,0])
        self.e_y = np.array([0,1])
        self.alpha = alpha
        self.beta = beta
        self.d0 = d0


    def find_obstacle(self):
        obstacles = []
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 1:
                    obstacles.append(np.array([i,j]))
        return obstacles


    def e_vector(self, a:np.ndarray, b: np.ndarray):
        if(np.linalg.norm(a-b) == 0):
            raise ValueError(f"{a}, {b}")
        
        return ((a[0] - b[0])*self.e_x + (a[1] - b[1])*self.e_y) / np.linalg.norm(a-b)


    def F_r(self, k: np.ndarray, i: np.ndarray, d0=10):
        
        rho = np.linalg.norm(k - i)
        if rho > d0:
            return np.zeros((2,))
        else:
            return (1/rho - 1/d0)**2 * self.e_vector(k, i)
    

    def F_a(self, k: np.ndarray):
        d = np.array(self.end)
        
        return (np.linalg.norm(k-d)) * self.e_vector(d, k)
    

    def v_cost(self, now_pos: np.ndarray, candidate_x, candidate_y):
        neibor = np.array([candidate_x, candidate_y])
        # calculate F(k)
        # alpha = -0.04 
        # beta = 0.5 
        f_a = self.F_a(now_pos)
        f_b = np.zeros((2,))
        
        for obstacle in self.obstacles:
            f_b += self.F_r(now_pos, obstacle, d0=self.d0)

        F_k = self.alpha * f_a + self.beta * f_b
        theta_n = self.angle_between(self.e_vector(neibor, now_pos), F_k)

        return np.linalg.norm(F_k) * np.cos(theta_n)
        
    
    def cost(self, now_node, candidate_x, candidate_y):
        now_pos = np.array(now_node.value)
        now_distance = self.distance[now_pos[0]][now_pos[1]]
        # diagonal heuristic
        dx = np.abs(self.end[0] - candidate_x)
        dy = np.abs(self.end[1] - candidate_y)
        h_n = dx + dy + (2**0.5 - 2) * min(dx, dy)

        return (now_distance + 1) + h_n + self.v_cost(now_pos, candidate_x, candidate_y)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map',
                        type=str,
                        default=r'.\map\map2.txt')
    parser.add_argument('--start',
                        type=int,
                        
                        nargs=2,
                        default=[0,0])
    parser.add_argument('--end',
                        type=int,
                        
                        nargs=2,
                        default=[15,18])
    args = parser.parse_args()

    my_map = utils.read_map(args.map)

    dj = Dijkstra(map=my_map, start=args.start, end=args.end)
    dj.find_shortest_path()
    dj.trace_route()
    print(dj.start)
    utils.draw(dj.map, dj.route, None, args.map.split('\\')[-1].split('.')[0] + '_dj.png', 1)
    

    As = A_star(map=my_map, start=args.start, end=args.end)
    As.find_shortest_path()
    As.trace_route()
    utils.draw(As.map, As.route, None, args.map.split('\\')[-1].split('.')[0] + '_as.png', 1)
    As.stastistic()
    
    # As.cal_dis_and_angle()
    # print(As.exp_time)
    
    

    # i_As = Improved_A_star(map=my_map, start=args.start, end=args.end)
    # i_As.find_shortest_path()
    # i_As.trace_route(args.map.split('\\')[-1].split('.')[0] + '_i.png')
    # i_As.cal_dis_and_angle()
    # print(i_As.exp_time)
    # dj.stastistic()
    # As.stastistic()
    # print("----------")
    # i_As.stastistic()
    