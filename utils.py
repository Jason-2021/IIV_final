import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
from pprint import pprint
import os

def read_map(map_name):

    with open(map_name, 'r') as f:
        lines = f.readlines()
        map = []
        for line in lines:
            map.append([int(i) for i in line.strip()])
        return map

def draw(map, route_1=None, route_2=None, filename='test.png', mode=0, smooth=False):
    '''
    mode 0: 2 color map
    mode 1: 4 color map
    '''
    map_size = len(map)
    if mode == 1:
        cmap = colors.ListedColormap(['white', 'black', 'gray', 'lightgreen'])
    else:
        cmap = colors.ListedColormap(['white', 'black'])
    plt.figure(figsize=(map_size, map_size))
    plt.pcolor(map, cmap=cmap, edgecolors='k', linewidth=5)

    plt.gca().set_aspect(1)
    plt.gca().invert_yaxis()
    # plt.ylim(map_size, 0)
    plt.yticks(np.arange(0, map_size)[::5], np.arange(1, map_size+1)[::5], fontsize=24)
    plt.xticks(np.arange(0, map_size)[::5], np.arange(1, map_size+1)[::5], fontsize=24)

    
    if route_1 is not None:
        route_1 = route_1 + 0.5
        plt.plot(route_1[:, 1], route_1[:, 0], color='orange', linewidth=10)
        
        if route_2 is not None:
            route_2 = route_2 + 0.5
            plt.plot(route_2[:, 1], route_2[:, 0], color='blue', linewidth=10)
            plt.legend(['Ours', 'A*'], loc='upper right', framealpha=1, fontsize=40, borderpad=1)
            if smooth:
                plt.title('Smooth', fontsize=80)
            else:
                plt.title('Non-smooth', fontsize=80)
        plt.scatter(route_1[0,1], route_1[0,0], color='red', marker='o',zorder=2, s=500)
        plt.scatter(route_1[-1,1], route_1[-1,0], color='red', marker='o', zorder=2, s=500)
        plt.text(route_1[0,1]+1, route_1[0,0], "start", color='red', fontsize=40, va='center') 
        plt.text(route_1[-1,1]+1, route_1[-1,0], "end", color='red', fontsize=40, va='center')
        
    
    plt.savefig(os.path.join('.', 'image', filename))
    




if __name__ == '__main__':
    map = read_map(r'.\map\map1.txt')
    draw(map, mode=0, filename='map1.png')
        
