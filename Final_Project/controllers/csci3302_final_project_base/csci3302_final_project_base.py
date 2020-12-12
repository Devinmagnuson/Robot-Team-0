import math
import time
import copy
import matplotlib.pyplot as plt
from controller import Robot, Motor, DistanceSensor, Camera
import csci3302_final_project_supervisor
import numpy as np
import heapq


state = "search"

LIDAR_SENSOR_MAX_RANGE = 3. # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

path_changed = False
search_found = False
object_found = False

box_location = [-1, -1]

# create the Robot instance.
csci3302_final_project_supervisor.init_supervisor()
robot = csci3302_final_project_supervisor.supervisor

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# get and enable lidar 
lidar = robot.getLidar("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

#Initialize lidar motors
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)


lidar_data = []
lidar_offsets = [0] * LIDAR_ANGLE_BINS

for i in range(LIDAR_ANGLE_BINS):
    lidar_offsets[i] = LIDAR_ANGLE_RANGE/2 - (LIDAR_ANGLE_RANGE * i / (LIDAR_ANGLE_BINS - 1))


# Map Variables
MAP_BOUNDS = [2.,2.] 
CELL_RESOLUTIONS = np.array([0.1, 0.1]) # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

object_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

search_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

"""
def populate_map(m):
    obs_list = csci3302_final_project_supervisor.supervisor_get_obstacle_positions()
    obs_size = 0.06 # 6cm boxes
    for obs in obs_list:
        obs_coords_lower = obs - obs_size/2.
        obs_coords_upper = obs + obs_size/2.
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1
        obs_coords_lower = [obs[0] - obs_size/2, obs[1] + obs_size/2.]
        obs_coords_upper = [obs[0] + obs_size/2., obs[1] - obs_size/2.]
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1
"""

# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

# Constants to help with the Odometry update
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# GAIN Values
theta_gain = 1.0
distance_gain = 0.3


EPUCK_MAX_WHEEL_SPEED = 0.12880519 # m/s
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_WHEEL_RADIUS = 0.0205 # ePuck's wheels are 0.041m in diameter.


# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)


camera = robot.getCamera('camera1')
camera.enable(SIM_TIMESTEP)
camera.recognitionEnable(SIM_TIMESTEP)


MAX_VEL_REDUCTION = 0.2


def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    '''
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    '''
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER;
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.;
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.;
    pose_theta = get_bounded_theta(pose_theta)

def get_bounded_theta(theta):
    '''
    Returns theta bounded in [-PI, PI]
    '''
    while theta > math.pi: theta -= 2.*math.pi
    while theta < -math.pi: theta += 2.*math.pi
    return theta

def get_wheel_speeds(target_pose):
    """   
    @param target_pose: Array of (x,y,theta) for the destination robot pose
    @return motor speed as percentage of maximum for left and right wheel motors
    """
    
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction
    
    pose_x, pose_y, pose_theta = csci3302_final_project_supervisor.supervisor_get_robot_pose()
    
    bearing_error = get_bounded_theta(math.atan2( (target_pose[1] - pose_y), (target_pose[0] - pose_x) ) - pose_theta)
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x,pose_y]))
    heading_error = target_pose[2] -  pose_theta
    
    BEAR_THRESHOLD = 0.1
    DIST_THRESHOLD = 0.03
    
    dT_gain = theta_gain
    dX_gain = distance_gain
    
    if distance_error > DIST_THRESHOLD:
    
        dTheta = bearing_error
        
        if abs(bearing_error) > BEAR_THRESHOLD:
            
            dX_gain = 0
                
    else:
    
        dTheta = heading_error
        dX_gain = 0
        
    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)
    
    phi_l = (dX - (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS
    
    left_speed_pct = 0
    right_speed_pct = 0
    
    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer
    
    if distance_error < 0.05 and abs(heading_error) < 0.05:
        left_speed_pct = 0
        right_speed_pct = 0
        
    
    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity()
    
    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * rightMotor.getMaxVelocity()
    
    return phi_l_pct, phi_r_pct


def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    """
    @param lidar_bin: The beam index that provided this measurement
    @param lidar_distance: The distance measurement from the sensor for that beam
    @return world_point: List containing the corresponding (x,y) point in the world frame of reference
    """
    
    global pose_x, pose_y, pose_theta, lidar_offsets
    adjusted_pose_y = pose_y
    adjusted_pose_theta = pose_theta
    
    robot_point = [0,0]
    
    robot_point[0] = lidar_distance * math.cos(lidar_offsets[lidar_bin])
    robot_point[1] = lidar_distance * math.sin(lidar_offsets[lidar_bin])
    
    x = pose_x + robot_point[0] * math.cos(adjusted_pose_theta) - robot_point[1] * math.sin(adjusted_pose_theta)
    y = adjusted_pose_y + robot_point[0] * math.sin(adjusted_pose_theta) + robot_point[1] * math.cos(adjusted_pose_theta)
    
    world_point = np.array([x, y])
    
    return world_point


def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))


def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None
    
    return np.array([(col+0.5)*CELL_RESOLUTIONS[1], (row+0.5)*CELL_RESOLUTIONS[0]])


def update_map(lidar_readings_array):
    """
    @param lidar_readings_array
    """
    global world_map, object_map, search_map, LIDAR_SENSOR_MAX_RANGE, path_changed, search_found, object_found
    global pose_x, pose_y, pose_theta
    global box_location
    
    map_coord = transform_world_coord_to_map_coord((pose_x, pose_y))
    
    if(map_coord != None):
    
        if(search_map[map_coord[0], map_coord[1]] <= 1):
                        
            search_map[map_coord[0], map_coord[1]] = 2
    
    max_distance = min(LIDAR_SENSOR_MAX_RANGE, math.sqrt((MAP_BOUNDS[0]*MAP_BOUNDS[0]) + (MAP_BOUNDS[1]*MAP_BOUNDS[1])))
    
    for i in range(len(lidar_readings_array)):
        
        distance = lidar_readings_array[i]
        
        if(distance < max_distance):
            
            world_coord = convert_lidar_reading_to_world_coord(i, distance)
            map_coord = transform_world_coord_to_map_coord(world_coord)
            
            if(map_coord != None):
            
                if(object_map[map_coord[0], map_coord[1]] < 100):
                
                    object_map[map_coord[0], map_coord[1]] += 10
                
                
                distance -= CELL_RESOLUTIONS[0]
            
            if(i >= 8 and i <= 12):
                
                while(distance > 0):
                
                    world_coord = convert_lidar_reading_to_world_coord(i, distance)
                    map_coord = transform_world_coord_to_map_coord(world_coord)
                        
                    if(map_coord != None):
                            
                        if(search_map[map_coord[0], map_coord[1]] <= 0):
                            
                            if(search_map[map_coord[0], map_coord[1]] == -1):
                                
                                search_found = True
                            
                            search_map[map_coord[0], map_coord[1]] = 1
                            
                    distance -= CELL_RESOLUTIONS[0]
                
        else:
        
            distance = max_distance
            
            while(distance > 0):
            
                world_coord = convert_lidar_reading_to_world_coord(i, distance)
                map_coord = transform_world_coord_to_map_coord(world_coord)
                    
                if(map_coord != None):
                        
                    if(object_map[map_coord[0], map_coord[1]] > -10):
                    
                        object_map[map_coord[0], map_coord[1]] -= 1
                        
                    if(i >= 8 and i <= 12):
                       
                        if(search_map[map_coord[0], map_coord[1]] <= 0):
                            
                            if(search_map[map_coord[0], map_coord[1]] == -1):
                                
                                search_found = True
                            
                            search_map[map_coord[0], map_coord[1]] = 1
                        
                distance -= CELL_RESOLUTIONS[0]
                
    for row in range(world_map.shape[0]-1,-1,-1):
        for col in range(world_map.shape[1]):
            if object_map[row,col] > 0:
            
                if not world_map[row,col] == 3:
                
                    if world_map[row,col] == 2:
                        path_changed = True
                        
                    world_map[row,col] = 1
                    
                    if search_map[row, col] == 1:
                        
                        search_map[row, col] = 3
                    
                    elif search_map[row, col] == 2:
                    
                        search_map[row, col] = 4
                        
                    
            else:
            
                if world_map[row,col] == 1:
                    world_map[row,col] = 0
                    
                if search_map[row, col] == 3:
                    
                    search_map[row, col] = 1
                
                elif search_map[row, col] == 4:
                
                    search_map[row, col] = 2
                
    if path_changed:
        for row in range(world_map.shape[0]-1,-1,-1):
            for col in range(world_map.shape[1]):
            
                if world_map[row,col] == 2 or world_map[row,col] == 3 or world_map[row,col] == 4:
                    world_map[row,col] = 0
                    
                if search_map[row,col] == -1:
                    search_map[row,col] = 0
                    
    if search_found:
        for row in range(world_map.shape[0]-1,-1,-1):
            for col in range(world_map.shape[1]):
            
                if world_map[row,col] == 2 or world_map[row,col] == 3 or world_map[row,col] == 4:
                    world_map[row,col] = 0
                    
                if search_map[row,col] == -1:
                    search_map[row,col] = 1
                    
                    
    if camera.getRecognitionNumberOfObjects() > 0:
    
        object = camera.getRecognitionObjects()[0]
        relative_loc = object.get_position()
        
        
        x = pose_x - relative_loc[2] * math.cos(pose_theta) + relative_loc[0] * math.sin(pose_theta)
        y = pose_y - relative_loc[2] * math.sin(pose_theta) - relative_loc[0] * math.cos(pose_theta)
        
        box_location = [x, y]
        
        object_found = True

                
def display_map(m):
    """
    @param m: The world map matrix to visualize
    """
    global object_map, search_map, box_location, object_found
    
    m2 = copy.copy(m)
    robot_pos = transform_world_coord_to_map_coord([pose_x,pose_y])
    if(object_found):
        box_pos = transform_world_coord_to_map_coord(box_location)
        m2[box_pos] = 9
    m2[robot_pos] = 8
    map_str = ""
    for row in range(m.shape[0]-1,-1,-1):
        for col in range(m.shape[1]):
            if m2[row,col] == 0: map_str += '[ ]'
            elif m2[row,col] == 1: map_str += '[X]'
            elif m2[row,col] == 2: map_str += '[+]'
            elif m2[row,col] == 3: map_str += '[G]'
            elif m2[row,col] == 4: map_str += '[S]'
            elif m2[row,col] == 8: map_str += '[r]'
            elif m2[row,col] == 9: map_str += '[B]'
            else: map_str += '[E]'

        map_str += '\n'

    print(map_str)
    print(' ')
    
    
    #To print the search_map
    
    """
    map_str = ""
    for row in range(m.shape[0]-1,-1,-1):
        for col in range(m.shape[1]):
            if search_map[row,col] == -1: map_str += '[T]'
            elif search_map[row,col] == 0: map_str += '[ ]'
            elif search_map[row,col] == 1: map_str += '[s]'
            elif search_map[row,col] == 2: map_str += '[S]'
            elif search_map[row,col] == 3: map_str += '[f]'
            elif search_map[row,col] == 4: map_str += '[F]'
            else: map_str += '[E]'
            
        map_str += '\n'
            
            
    print(map_str)
    print(' ')
    print(' ')
    """
    
    """
    #To save an image of the object_map
    
    img_scale = 20
    
    img_map = np.zeros([NUM_Y_CELLS * img_scale, NUM_X_CELLS * img_scale])
    
    for row in range(m.shape[0]-1,-1,-1):
        for col in range(m.shape[1]):
        
            for x in range(img_scale):
                for y in range(img_scale):
        
                    img_map[img_scale*row + x, img_scale*col + y] = object_map[m.shape[0] - row - 1, col]
    
    plot = np.array(img_map)
    plt.imshow(plot)
    
    cmap = plt.cm.jet
    plt.imsave('map_image.png', plot, cmap=cmap)
    """





def get_travel_cost(source_vertex, dest_vertex):
    """
    @param source_vertex: world_map coordinates for the starting vertex
    @param dest_vertex: world_map coordinates for the destination vertex
    @return cost: Cost to travel from source to dest vertex.
    """
    global world_map
    
    cost = 1e5 # TODO: Replace with your code
    
    source_x, source_y = source_vertex
    dest_x, dest_y = dest_vertex
    
    if (world_map[dest_x, dest_y] != 1):
    
        x_diff = abs(source_x - dest_x)
        y_diff = abs(source_y - dest_y)
        
        if ((x_diff + y_diff) == 1):
            cost = 1
        elif ((x_diff + y_diff) == 0):
            cost = 0
        
    
    return cost


def dijkstra(source_vertex):
    """
    @param source_vertex: Starting vertex for the search algorithm.
    @return prev: Data structure that maps every vertex to the coordinates of the previous vertex (along the shortest path back to source)
    """
    global world_map

    # TODO: Initialize these variables
    dist = copy.copy(world_map)
    q_cost = []
    prev = np.full([NUM_Y_CELLS,NUM_X_CELLS], None)

    # TODO: Your code here
    for row in range(world_map.shape[0]):
        for col in range(world_map.shape[1]):
        
            vertex = (row,col)
        
            if(vertex != source_vertex):
                dist[vertex] = 1e5
            else:
                dist[vertex] = 0
                
            prev[vertex] = None
            
            heapq.heappush(q_cost, (dist[vertex], vertex))
            
    while(len(q_cost) != 0):
    
        heapq.heapify(q_cost)
        
        pop = heapq.heappop(q_cost)
        vertex = pop[1]
        distance = pop[0]
        
        for row in range(world_map.shape[0]):
            for col in range(world_map.shape[1]):
            
                neighbor = (row,col)
            
                if(get_travel_cost(vertex, neighbor) == 1):
                    
                    temp = distance + 1
                    if(temp < dist[neighbor]):
                        q_cost.remove((dist[neighbor], neighbor))
                        
                        dist[neighbor] = temp
                        prev[neighbor] = vertex
                        
                        heapq.heappush(q_cost, (temp, neighbor))
        
    
    return prev


def reconstruct_path(prev, goal_vertex):
    """
    @param prev: Data structure mapping each vertex to the next vertex along the path back to "source" (from Dijkstra)
    @param goal_vertex: Map coordinates of the goal_vertex
    @return path: List of vertices where path[0] = source_vertex_ij_coords and path[-1] = goal_vertex_ij_coords
    """
    
    path = [goal_vertex]
    current = goal_vertex
    
    while(current != None):
    
        current = prev[current]
        
        if(current != None):
            
            path.append(current)
            
    path.reverse()
    
    return path


def visualize_path(path):
    """
    @param path: List of graph vertices along the robot's desired path    
    """
    global world_map
    
    for vertex in path:
        world_map[vertex] = 2
        
    world_map[path[0]] = 4
    world_map[path[-1]] = 3
    
    # TODO: Set a value for each vertex along path in the world_map for rendering: 2 = Path, 3 = Goal, 4 = Start
    
    return

def main():
    global robot, state, sub_state, world_map, search_map, box_location
    global lidar, path_changed, search_found, object_found
    global leftMotor, rightMotor, SIM_TIMESTEP, WHEEL_FORWARD, WHEEL_STOPPED, WHEEL_BACKWARDS
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    last_odometry_update_time = None

    # Keep track of which direction each wheel is turning
    left_wheel_direction = WHEEL_STOPPED
    right_wheel_direction = WHEEL_STOPPED

    # Important IK Variable storing final desired pose
    target_pose = None # Populated by the supervisor, only when the target is moved.
    object_first_found = True
    push_target = None

    # Sensor burn-in period
    for i in range(10): robot.step(SIM_TIMESTEP)

    start_pose = csci3302_final_project_supervisor.supervisor_get_robot_pose()
    pose_x, pose_y, pose_theta = start_pose

    
    # Main Control Loop:
    while robot.step(SIM_TIMESTEP) != -1:
    
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()
        time_elapsed = robot.getTime() - last_odometry_update_time
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        last_odometry_update_time = robot.getTime()
        
        
        if target_pose is None:
            target_pose = csci3302_final_project_supervisor.supervisor_get_target_pose()
            #world_map[transform_world_coord_to_map_coord(target_pose[:2])] = 3 # Goal vertex!
            print("New IK Goal Received! Target: %s" % str(target_pose))
            print("Current pose: [%5f, %5f, %5f]\t\t Target pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta, target_pose[0], target_pose[1], target_pose[2]))
            #populate_map(world_map)
            display_map(world_map)

        
        if path_changed:
            if(state == 'get_path' or state == 'get_waypoint' or state == 'move_to_waypoint'):
                
                state = 'get_path'
                
            if(state == 'search' or state == 'search_get_waypoint' or state == 'search_move'):
                
                state = 'search'
                
            if(state == 'push_object_path' or state == 'push_object_waypoint' or state == 'push_move'):
            
                state = 'push_object_path'
                
            path_changed = False
            
        if search_found:
            state = 'search'
            search_found = False
            
        if object_found and object_first_found:
            state = 'push_object_path'
            object_first_found = False
            


        if state == 'get_path': 
            # Compute a path from start to target_pose
            current_pose = (pose_x,pose_y)
            prev = dijkstra(transform_world_coord_to_map_coord(current_pose))
            
            if(object_found):
            
                path = reconstruct_path(prev, push_target)  
                visualize_path(path)
                
                if(transform_world_coord_to_map_coord(current_pose) == push_target):
                    state = 'push_object_path'
                elif(path[0] == transform_world_coord_to_map_coord(current_pose) and path[-1] == push_target):
                    waypoint_index = 0
                    
                    state = 'get_waypoint'
                pass 
            
            else:
                path = reconstruct_path(prev, transform_world_coord_to_map_coord(target_pose[:2]))  
                visualize_path(path)
            
                if(path[0] == transform_world_coord_to_map_coord(current_pose) and path[-1] == transform_world_coord_to_map_coord(target_pose[:2])):
                    waypoint_index = 0
                    state = 'get_waypoint'
                elif(transform_world_coord_to_map_coord(current_pose) == transform_world_coord_to_map_coord(target_pose[:2])):
                    state = 'finished'
                pass
            
        elif state == 'get_waypoint':
               
            # Get the next waypoint from the path
            waypoint_index += 1
            
            if(waypoint_index >= len(path)):
                state = 'get_path'
            
            else:
                waypoint = transform_map_coord_world_coord(path[waypoint_index])
                wp_pose_x, wp_pose_y = waypoint
                
                if(waypoint_index + 1 >= len(path)):
                
                    wp_pose_theta = target_pose[2]
                    
                else:
                    
                    next_waypoint = transform_map_coord_world_coord(path[waypoint_index + 1])
                    next_x, next_y = next_waypoint
                    
                    wp_pose_theta = math.atan2(next_y - wp_pose_y, next_x - wp_pose_x)
                
                if(wp_pose_theta > math.pi):
                    
                    wp_pose_theta -= 2*math.pi
                    
                elif(wp_pose_theta <= -1*math.pi):
                    
                    wp_pose_theta += 2*math.pi
                
                target_wp = (wp_pose_x, wp_pose_y, wp_pose_theta)
                
                state = 'move_to_waypoint'            
            pass
        elif state == 'move_to_waypoint':
            
            lspeed, rspeed = get_wheel_speeds(target_wp)
            leftMotor.setVelocity(lspeed)
            rightMotor.setVelocity(rspeed)
            
            if((lspeed == 0 and rspeed == 0)):
                state = 'get_waypoint'
            
        elif state == 'spin':
        
            left_wheel_direction, right_wheel_direction = -1, 1
            leftMotor.setVelocity(-1 * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity())
            rightMotor.setVelocity(1 * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity())
            
        elif state == 'search':
        
            if(object_found):
                state = 'push_object_path'
        
            # Compute a path from start to target_pose
            current_pose = (pose_x,pose_y)
            
            map_pose = transform_world_coord_to_map_coord(current_pose)
            
            search_map[map_pose[0], map_pose[1]] = 2
            closest = [0, 0]
            closest_len = 1e5
            
            for row in range(search_map.shape[0]-1,-1,-1):
                for col in range(search_map.shape[1]):
                
                    if(search_map[col,row] == 0):
                    
                        map_distance = math.sqrt((row - map_pose[1])**2 + (col - map_pose[0])**2)
                        
                        if(map_distance < closest_len):
                        
                            closest = [row,col]
                            closest_len = map_distance
                            
            
            if(closest_len == 1e5):
            
                state = 'finished'
            
            else:
                
                prev = dijkstra(transform_world_coord_to_map_coord(current_pose))
                path = reconstruct_path(prev, (closest[1],closest[0]))  
                visualize_path(path)
                
                search_map[closest[1],closest[0]] = -1
                
                if(path[0] == map_pose and path[-1] == (closest[1],closest[0])):
                    waypoint_index = 0
                    state = 'search_get_waypoint'
                    
                    if(object_found):
                        state = 'push_object_path'
        
        elif state == 'search_get_waypoint':
        
            waypoint_index += 1
            
            if(waypoint_index >= len(path)):
                state = 'get_path'
            
            else:
                waypoint = transform_map_coord_world_coord(path[waypoint_index])
                wp_pose_x, wp_pose_y = waypoint
                
                if(waypoint_index + 1 >= len(path)):
                
                    wp_pose_theta = target_pose[2]
                    
                else:
                    
                    next_waypoint = transform_map_coord_world_coord(path[waypoint_index + 1])
                    next_x, next_y = next_waypoint
                    
                    wp_pose_theta = math.atan2(next_y - wp_pose_y, next_x - wp_pose_x)
                
                if(wp_pose_theta > math.pi):
                    
                    wp_pose_theta -= 2*math.pi
                    
                elif(wp_pose_theta <= -1*math.pi):
                    
                    wp_pose_theta += 2*math.pi
                
                target_wp = (wp_pose_x, wp_pose_y, wp_pose_theta)
                
                state = 'search_move'            
            pass
        
        elif state == 'search_move':
        
            lspeed, rspeed = get_wheel_speeds(target_wp)
            leftMotor.setVelocity(lspeed)
            rightMotor.setVelocity(rspeed)
            
            if((lspeed == 0 and rspeed == 0)):
                if(object_found):
                    state = 'push_object_path'
                else:
                    state = 'search_get_waypoint'
        
        elif state == 'push_object_path':
        
            # Compute a path from start to target_pose
            current_pose = box_location
            
            map_pose = transform_world_coord_to_map_coord(current_pose)
            
            
            prev = dijkstra(transform_world_coord_to_map_coord(current_pose))
            path = reconstruct_path(prev, transform_world_coord_to_map_coord(target_pose[:2]))  
            visualize_path(path)
                
            if(map_pose == transform_world_coord_to_map_coord(target_pose[:2])):
                state = 'finished'
            elif(path[0] == map_pose and path[-1] ==  transform_world_coord_to_map_coord(target_pose[:2])):
                waypoint_index = 0
                state = 'push_object_waypoint'
            
        
        elif state == 'push_object_waypoint':
        
            waypoint_index += 1
            
            if(waypoint_index >= len(path)):
            
                state = 'push_object_path'
            
            else:
                box_position = transform_world_coord_to_map_coord(box_location)
                robot_position = transform_world_coord_to_map_coord((pose_x,pose_y))
                
                waypoint_map = path[waypoint_index]
                
                waypoint = transform_map_coord_world_coord(path[waypoint_index])
                
                if(waypoint_map[0] == box_position[0] + 1 and waypoint_map[1] == box_position[1]):
                    push_target = (box_position[0] - 1, box_position[1])
                elif(waypoint_map[0] == box_position[0] - 1 and waypoint_map[1] == box_position[1]):
                    push_target = (box_position[0] + 1, box_position[1])
                elif(waypoint_map[0] == box_position[0] and waypoint_map[1] == box_position[1] + 1):
                    push_target = (box_position[0], box_position[1] - 1)
                elif(waypoint_map[0] == box_position[0] and waypoint_map[1] == box_position[1] - 1):
                    push_target = (box_position[0], box_position[1] + 1)
                ##else:
                ##    print(waypoint_map)
                ##    print(box_position)
                ##    state = 'push_object_path'
                        
                if not((robot_position == push_target) or (robot_position == box_position)):
                    
                    state = 'get_path'
                    
                elif state == 'push_object_waypoint':
                
                    wp_pose_x, wp_pose_y = transform_map_coord_world_coord(path[waypoint_index])
                    
                    if(waypoint_index + 1 >= len(path)):
                    
                        wp_pose_theta = target_pose[2]
                        
                    else:
                        
                        next_waypoint = transform_map_coord_world_coord(path[waypoint_index + 1])
                        next_x, next_y = next_waypoint
                        
                        wp_pose_theta = math.atan2(next_y - wp_pose_y, next_x - wp_pose_x)
                    
                    if(wp_pose_theta > math.pi):
                        
                        wp_pose_theta -= 2*math.pi
                        
                    elif(wp_pose_theta <= -1*math.pi):
                        
                        wp_pose_theta += 2*math.pi
                    
                    target_wp = (wp_pose_x, wp_pose_y, wp_pose_theta)
                    
                    state = 'push_move'            
            pass
        
        elif state == 'push_move':
        
            lspeed, rspeed = get_wheel_speeds(target_wp)
            leftMotor.setVelocity(lspeed)
            rightMotor.setVelocity(rspeed)
            
            if((lspeed == 0 and rspeed == 0)):
                state = 'push_object_waypoint'
        
        else:
        
            # Stop
            left_wheel_direction, right_wheel_direction = 0, 0
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(0)    

            
        lidar_data = lidar.getRangeImage()
        update_map(lidar_data)
        
        display_map(world_map)
        print(state)
        
    
    
if __name__ == "__main__":
    main()
