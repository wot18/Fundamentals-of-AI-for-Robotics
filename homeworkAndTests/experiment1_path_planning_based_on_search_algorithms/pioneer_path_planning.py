from imaplib import ParseFlags
import sim
import math
import time
import sys
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, g_cost=float('inf'), h_cost=0):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = None
        
    def f_cost(self):
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

def connect_simulator():
    sim.simxFinish(-1)
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to CoppeliaSim')
        return clientID
    else:
        print('Failed to connect to CoppeliaSim')
        sys.exit()

def get_object_position(clientID, object_handle):
    ret, position = sim.simxGetObjectPosition(clientID, object_handle, -1, sim.simx_opmode_oneshot_wait)
    # 将位置列表转换为元组
    return tuple(position) if ret == sim.simx_return_ok else None

def get_neighbors(current_pos, grid_size=0.5):
    # 8个方向的邻居节点
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
    neighbors = []
    for dx, dy in directions:
        new_pos = (current_pos[0] + dx * grid_size, current_pos[1] + dy * grid_size, current_pos[2])
        neighbors.append(new_pos)
    return neighbors

def get_obstacles_positions(clientID):
    obstacles = []
    # 获取场景中所有立方体障碍物的位置
    for i in range(6):  # 假设有6个障碍物
        ret, obstacle = sim.simxGetObjectHandle(clientID, f'Cuboid{i}', sim.simx_opmode_oneshot_wait)
        if ret == sim.simx_return_ok:
            pos = get_object_position(clientID, obstacle)
            if pos:
                obstacles.append(pos)
    return obstacles

def default_heuristic(start, goal):
    return math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)

def get_neighbors(current_pos, goal_pos, grid_size=0.5, heuristic_func=None):

    if heuristic_func is None:
        heuristic_func = default_heuristic
        pass
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
    neighbors = []
    current_distance = heuristic_func(current_pos, goal_pos)
    
    for dx, dy in directions:
        new_pos = (current_pos[0] + dx * grid_size, current_pos[1] + dy * grid_size, current_pos[2])
        # 检查新位置是否更接近目标且远离边界
        new_distance = heuristic_func(new_pos, goal_pos)
        if new_distance <= current_distance:
            neighbors.append(new_pos)
    return neighbors

def default_search(start_pos, goal_pos, obstacles, grid_size=0.5):

    start_node = Node(start_pos, 0)
    start_node.h_cost = heuristic(start_pos, goal_pos)
    
    open_list = []
    closed_set = set()
    heappush(open_list, start_node)
    nodes = {start_pos: start_node}
    
    while open_list:
        current = heappop(open_list)
        
        if heuristic(current.position, goal_pos) < grid_size:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
            
        closed_set.add(current.position)
        
        # 修改这里，传入goal_pos参数
        for neighbor_pos in get_neighbors(current.position, goal_pos, grid_size, heuristic_func=heuristic):
            if any(heuristic(neighbor_pos, obs) < grid_size for obs in obstacles):
                continue
                
            g_cost = current.g_cost + heuristic(current.position, neighbor_pos)
            
            if neighbor_pos not in nodes:
                neighbor = Node(neighbor_pos)
                nodes[neighbor_pos] = neighbor
            else:
                neighbor = nodes[neighbor_pos]
                
            if neighbor.position in closed_set:
                continue
                
            if g_cost < neighbor.g_cost:
                neighbor.g_cost = g_cost
                neighbor.h_cost = heuristic(neighbor_pos, goal_pos)
                neighbor.parent = current
                heappush(open_list, neighbor)
    
    return None

def move_to_target(clientID, robot, leftMotor, rightMotor, current_pos, target_pos, speed=0.2):
    # 计算朝向目标点需要的角度
    angle = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
    
    # 获取机器人当前朝向
    ret, euler_angles = sim.simxGetObjectOrientation(clientID, robot, -1, sim.simx_opmode_oneshot_wait)
    current_angle = euler_angles[2]
    
    # 计算需要转动的角度
    angle_diff = angle - current_angle
    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < -math.pi:
        angle_diff += 2 * math.pi
        
    # 转向（修正旋转方向）
    # 转向（使用差速驱动运动学公式）
    #wheel_radius = 0.0975  # Pioneer 3-DX轮子半径（米）
    #wheel_distance = 0.381  # Pioneer 3-DX两轮之间的距离（米）
    
    while abs(angle_diff) >= 0.1:  # 角度阈值保持0.05
       omega = 0.05 if angle_diff > 0 else -0.05  # 转向速度保持0.3
       # 修改左右轮速度的设置，使旋转方向与目标方向一致
       sim.simxSetJointTargetVelocity(clientID, leftMotor, -omega, sim.simx_opmode_streaming)
       sim.simxSetJointTargetVelocity(clientID, rightMotor, omega, sim.simx_opmode_streaming)
        
       # 等待一小段时间
       time.sleep(0.05)
        
       # 更新角度差
       ret, euler_angles = sim.simxGetObjectOrientation(clientID, robot, -1, sim.simx_opmode_oneshot_wait)
       current_angle = euler_angles[2]
       angle_diff = angle - current_angle
       if angle_diff > math.pi:
           angle_diff -= 2 * math.pi
       elif angle_diff < -math.pi:
           angle_diff += 2 * math.pi
    
    # 停止转向
    sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
    
    # 直线运动
    distance = math.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)
    sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_streaming)
    time.sleep(distance / speed)
    
    # 到达目标点后停止
    sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)

def draw_path(clientID, path):
    # 创建一个空的集合来存储路径点
    path_points = []
    for point in path:
        # 在场景中创建一个小立方体来表示路径点
        ret, point_dummy = sim.simxCreateDummy(clientID, 0.1, None, sim.simx_opmode_oneshot_wait)
        if ret == sim.simx_return_ok:
            # 设置立方体位置
            sim.simxSetObjectPosition(clientID, point_dummy, -1, point, sim.simx_opmode_oneshot)
            path_points.append(point_dummy)
    return path_points

def set_random_obstacles(clientID, num_obstacles=6):
    import random
    
    obstacles_positions = []
    min_distance = 0.5  # 障碍物之间的最小距离
    
    for i in range(num_obstacles):
        while True:
            # 随机生成位置
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            z = 0.15  # 障碍物高度的一半
            
            # 检查与其他障碍物的距离
            valid_position = True
            for pos in obstacles_positions:
                if math.sqrt((x - pos[0])**2 + (y - pos[1])**2) < min_distance:
                    valid_position = False
                    break
            
            # 如果位置有效，设置障碍物位置
            if valid_position:
                ret, obstacle = sim.simxGetObjectHandle(clientID, f'Cuboid{i}', sim.simx_opmode_oneshot_wait)
                if ret == sim.simx_return_ok:
                    sim.simxSetObjectPosition(clientID, obstacle, -1, (x, y, z), sim.simx_opmode_oneshot)
                    obstacles_positions.append((x, y, z))
                break
    
    return obstacles_positions

def draw_map(planned_path, actual_path, obstacles, grid_size=0.5):
    # 创建图形
    plt.figure(figsize=(10, 10))
    
    # 设置坐标轴范围
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
    # 绘制网格
    plt.grid(True)
    
    # 绘制障碍物（红色方块）
    # 先单独绘制一个带标签的障碍物
    if obstacles:
        plt.plot(obstacles[0][0], obstacles[0][1], 's', color='red', markersize=35, label='Obstacles')
        # 绘制其余障碍物，不带标签
        for obs in obstacles[1:]:
            plt.plot(obs[0], obs[1], 's', color='red', markersize=35)
    
    # 绘制规划路径（蓝色线）
    planned_x = [pos[0] for pos in planned_path]
    planned_y = [pos[1] for pos in planned_path]
    plt.plot(planned_x, planned_y, 'b-', label='Planned Path')
    
    # 绘制实际路径（绿色线）
    actual_x = [pos[0] for pos in actual_path]
    actual_y = [pos[1] for pos in actual_path]
    plt.plot(actual_x, actual_y, 'g-', label='Actual Path')
    
    # 标记起点和终点
    plt.plot(planned_x[0], planned_y[0], 'ko', label='Start')
    plt.plot(planned_x[-1], planned_y[-1], 'k*', label='Goal')
    
    # 添加图例
    plt.legend()
    
    # 添加标题和轴标签
    plt.title('Path Planning and Robot Movement')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 显示图形
    plt.show()

def runSimulation(search_func=None):
    
    if search_func is None:
        search_func = default_search
        pass

    clientID = connect_simulator()
    
    # 随机设置障碍物位置
    obstacles = set_random_obstacles(clientID)
    
    # 获取机器人和电机句柄
    ret, robot = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx', sim.simx_opmode_oneshot_wait)
    ret, leftMotor = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor', sim.simx_opmode_oneshot_wait)
    ret, rightMotor = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_rightMotor', sim.simx_opmode_oneshot_wait)
    
    # 获取起始位置和目标位置
    start_pos = get_object_position(clientID, robot)
    goal_pos = (1.5, -1.5, start_pos[2])
    
    # 用于记录实际路径的列表
    actual_path = [start_pos]
    
    try:
        # 使用A*算法规划路径
        planned_path = search_func(start_pos, goal_pos, obstacles)
        
        if planned_path:
            # 可视化路径
            path_points = draw_path(clientID, planned_path)
            
            # 沿着规划的路径移动
            for target in planned_path[1:]:  # 跳过起始点
                current_pos = get_object_position(clientID, robot)
                move_to_target(clientID, robot, leftMotor, rightMotor, current_pos, target)
                # 记录实际位置
                actual_pos = get_object_position(clientID, robot)
                actual_path.append(actual_pos)
                
            # 到达最终目标点后，确保完全停止
            sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
            sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
            print("到达目标点，机器人已停止")
            
            # 绘制路径图
            obstacles= get_obstacles_positions(clientID)
            draw_map(planned_path, actual_path, obstacles)
            
            # 显示提示框并停止仿真
            sim.simxDisplayDialog(clientID, "任务完成", "机器人已到达目标位置！", sim.sim_dlgstyle_ok, "", None, None, sim.simx_opmode_oneshot_wait)
            sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
        else:
            print("未找到可行路径")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        # 停止机器人
        sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
        sim.simxFinish(clientID)

if __name__ == '__main__':
    main()