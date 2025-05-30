from enum import Enum
import globalVar
import random

BLOCK_NUM= globalVar.get_value('BLOCK_NUM')

class Color(Enum):
    ''' 颜色 '''
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    @staticmethod
    def random_color():
        '''设置随机颜色'''
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        return (r, g, b)

class Map(object):
    def __init__(self, mapsize):
        self.mapsize = mapsize

    def generate_cell(self, cell_width, cell_height):
        '''
        定义一个生成器，用来生成地图中的所有节点坐标
        :param cell_width: 节点宽度
        :param cell_height: 节点长度
        :return: 返回地图中的节点
        '''
        x_cell = -cell_width
        for num_x in range(self.mapsize[0] // cell_width):
            y_cell = -cell_height
            x_cell += cell_width
            for num_y in range(self.mapsize[1] // cell_height):
                y_cell += cell_height
                yield (x_cell, y_cell)

# 随机生成70个障碍物
def generate_random_obstacles(mapsize, num_obstacles, start, end):
    """
    生成随机障碍物
    
    参数:
        mapsize: 地图大小
        num_obstacles: 障碍物数量
        start: 起点坐标
        end: 终点坐标
    """
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = random.randint(0, mapsize[0]-1)
        y = random.randint(0, mapsize[1]-1)
        pos = (x, y)
        # 确保障碍物不会出现在起点和终点
        if pos != start and pos != end:
            obstacles.add(pos)
    return list(obstacles)