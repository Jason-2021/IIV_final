from shortest_path import Dijkstra, A_star, Improved_A_star
from utils import draw, read_map
import copy
import os


def experiment(map, map_name='map1', start=(0,0), end=(0,1), alpha=-0.04, beta=0.5, d0=15):
    dj = Dijkstra(copy.deepcopy(map), start, end, 0)
    As = A_star(copy.deepcopy(map), start, end)
    improved = Improved_A_star(copy.deepcopy(map), start, end, alpha, beta, d0)

    dj.find_shortest_path()
    As.find_shortest_path()
    improved.find_shortest_path()

    dj.trace_route()
    As.trace_route()
    improved.trace_route()

    # print stastistics
    print(map_name)
    print(f"Dijakstra: \nTime: {dj.exp_time} sec\nShortest path: {dj.answer()}")
    dj.cal_dis_and_angle()
    dj.stastistic()
    print("----------------------------------")

    print(f"A star: \nTime: {As.exp_time} sec\nShortest path: {As.answer()}")
    As.cal_dis_and_angle()
    As.stastistic()
    print("----------------------------------")

    print(f"Ours: \nTime: {improved.exp_time} sec\nShortest path: {improved.answer()}")
    improved.cal_dis_and_angle()
    improved.stastistic()
    print("----------------------------------")

    

    # draw 4 color map
    draw(dj.map, dj.route, mode=1, filename=map_name+'_4c_dj.png')
    draw(As.map, As.route, mode=1, filename=map_name+"_4c_as.png")

    # 2 color map
    draw(map, dj.route, None, map_name + "_dj.png")
    
    # original
    draw(map, As.route, improved.route, filename=map_name+"_original_compare.png")
    # smooth
    draw(map, As.smooth_route, improved.smooth_route, filename=map_name+"_SMOOTH_compare.png", mode=0, smooth=True)

    # ori vs smooth
    # draw(map, As.route, None, filename='non_smooth.png')
    # draw(map, As.smooth_route, None, filename='smooth.png')

    


if __name__ == '__main__':
    map1 = read_map(os.path.join('.', 'map', 'map1.txt'))
    map2 = read_map(os.path.join('.', 'map', 'map2.txt'))
    map3 = read_map(os.path.join('.', 'map', 'map3.txt'))

    _ = experiment(map1, 'map1', start=(0,0), end=(34, 23))
    _ = experiment(map2, 'map2', start=(0,0), end=(15, 18), alpha=-0.0, beta=1.6, d0=10)
    _ = experiment(map3, 'map3', start=(35, 23), end=(34, 4))