# -*- coding: utf-8 -*-

from task import Task
from buchi_parse import Buchi
from workspace import Workspace
from geodesic_path import Geodesic
import datetime
from collections import OrderedDict
import numpy as np
from biased_tree import BiasedTree
from construct_biased_tree import construction_biased_tree#, path_via_visibility
from draw_picture import path_plot, path_print
from text_editor import export_to_txt, export_cov_to_txt, export_disc_to_txt
import matplotlib.pyplot as plt
import pyvisgraph as vg
from termcolor import colored
import networkx as nx



if __name__ == "__main__":
    # task
    identity='run1'
    number_of_trials = 1
    save_waypoints = True
    save_covariances = True
    drone_height = 16.0     # altitude of drones
    waypoint_folder_location = "/home/samarth/catkin_ws/src/rotors_simulator/rotors_gazebo/resource"
    launch_folder_location = "/home/samarth/catkin_ws/src/rotors_simulator/rotors_gazebo/launch"
    
    
    time_array=[]   #stores time for each trial run
    cost_array=[]   #stores cost of each trial run
    
    
    for round_num in range(number_of_trials):
        print('Trial {}'.format(round_num+1))
        start = datetime.datetime.now()
        task = Task()
        buchi = Buchi(task)
        buchi.construct_buchi_graph()
        buchi.get_minimal_length()
        buchi.get_feasible_accepting_state()
        buchi_graph = buchi.buchi_graph
        NBA_time = (datetime.datetime.now() - start).total_seconds()
        print('Time for constructing the NBA: {0:.4f} s'.format(NBA_time))
        
        # workspace
        workspace = Workspace()
        geodesic = Geodesic(workspace, task.threshold)
        # parameters
        n_max = 100000
        para = dict()
        # lite version, excluding extending and rewiring
        para['is_lite'] = True
        # step_size used in function near
        para['step_size'] = 0.25 * buchi.number_of_robots
        # probability of choosing node q_p_closest
        para['p_closest'] = 0.9
        # probability used when deciding the target point
        para['y_rand'] = 0.99
        # probability used when deciding the target point when inside sensing range
        # more random actions to increase visibility of     
        # para['y_rand'] = 0.99
        # minimum distance between any pair of robots
        para['threshold'] = task.threshold
        # Updates landmark covariance when inside sensor range
        para['update_covariance'] = True
        # sensor range in meters
        para['sensor_range'] = 10
        # sensor measurement noise
        para['sensor_R'] = 0.5
        
        
        
        for b_init in buchi_graph.graph['init']:
            # initialization
            opt_cost = np.inf
            opt_path_pre = []
            opt_path_suf = []
        
            # ----------------------------------------------------------------#
            #                            Prefix Part                          #
            # ----------------------------------------------------------------#
        
            start = datetime.datetime.now()
            init_state = (task.init, b_init)
            # init_state = (((28.13007841678492, 96.5686624041831),
            #           (11.655391499113755, 77.18077120816896),
            #           (39.53486085989735, 30.22294758970175),
            #           (58.96178129887859, 76.04140804427708),
            #           (98.3515083106104, 51.015593324837354)),
            #           'T2_S2')
            init_label = task.init_label
            init_angle = task.init_angle
            tree_pre = BiasedTree(workspace, geodesic, buchi, task, init_state, init_label, init_angle, 'prefix', para)
            
            # print('------------------------------ prefix path --------------------------------')
            # construct the tree for the prefix part
            cost_path_pre, nodes, lm_cov, targets = construction_biased_tree(tree_pre, n_max)
            if len(tree_pre.goals):
                pre_time = ((datetime.datetime.now() - start).total_seconds())/60
                print('Time for the prefix path: {0:.4f} min'.format(pre_time))
                print('{0} accepting goals found'.format(len(tree_pre.goals)))
            else:
                print('Couldn\'t find the path within predetermined number of iteration')
                break
            
                    
            for i in range(len(tree_pre.goals)):
                if cost_path_pre[i][0] < opt_cost:
                    opt_path_suf=[]
                    opt_path_pre = cost_path_pre[i][1]
                    opt_cost = cost_path_pre[i][0]
               
            # path_print((opt_path_pre, opt_path_suf), workspace, buchi.number_of_robots)
            path_plot((opt_path_pre, opt_path_suf), workspace, tree_pre.biased_tree.nodes[cost_path_pre[0][1][-1]]['lm'], buchi.number_of_robots, round_num, identity)
            plt.show()
            
            print('Time for the prefix path: {0:.4f} min'.format(pre_time))
            print('Cost of path: {}'.format(cost_path_pre[0][0]))
            print(' ')
            print(' ')
            print(len(cost_path_pre[0][1]))
            time_array.append(pre_time)
            cost_array.append(cost_path_pre[0][0])
    print(time_array)
    print(cost_array)
    if number_of_trials == 1 and  save_waypoints:
        robot_waypoints, robot_wp_satsify_AP = export_disc_to_txt(cost_path_pre, targets, buchi.number_of_robots, drone_height, waypoint_folder_location, launch_folder_location, 10)
    if number_of_trials == 1 and  save_covariances:
        export_cov_to_txt(lm_cov, waypoint_folder_location, launch_folder_location)
    
    # tree_pre.biased_tree.nodes[cost_path_pre[0][1][-1]]['lm'].landmark['l11'][0] = [102,60]
    # tree_pre.biased_tree.nodes[cost_path_pre[0][1][-1]]['lm'].generate_samples_for_lm('l11')
    # rob_waypoint = robot_waypoints[0][95]
    # next_rob_waypoint = []#robot_waypoints[:][91:95]
    # for i in range(buchi.number_of_robots):
    #     next_rob_waypoint.append(robot_waypoints[i][96:110])
    # replaning_bool = task.Replanning_check(rob_waypoint, next_rob_waypoint, tree_pre.biased_tree.nodes[cost_path_pre[0][1][-1]]['lm'], robot_wp_satsify_AP, 0, buchi_graph)

    # x = ((98, 62), (10, 8), (15, 8), (20, 8), (25, 8))        
    # task.get_label_landmark(x,tree_pre.biased_tree.nodes[cost_path_pre[0][1][-1]]['lm'])
    # workspace.landmark={'l1': [[9.6, 132], [[3, 0], [0, 3]]], 
    #  'l2': [[25, 67], [[3, 0], [0, 4]]], 
    #  'l3': [[24, 59], [[3, 0], [0, 3]]], 
    #  'l4': [[15.985918660481234, 42.26619141644566], [[0.0625    , 0.0], [0.0, 0.06578947]]], 
    #  'l5': [[42.961187589052315, 21.973073644147597], [[0.00289687, 0.0 ], [0.0, 0.00289645]]], 
    #  'l6': [[97, 9], [[4, 0], [0, 2]]], 
    #  'l7': [[75, 47], [[3, 0], [0, 4]]], 
    #  'l8': [[100, 91], [[2, 0], [0, 2]]], 
    #  'l9': [[104, 136], [[1, 0], [0, 1]]], 
    #  'l10': [[128, 10], [[2, 0], [0, 2]]], 
    #  'l11': [[100, 55], [[2, 0], [0, 2]]], 
    #  'l12': [[135, 95], [[2, 0], [0, 2]]], 
    #  'l13': [[129, 48], [[2, 0], [0, 2]]]}
    # robot_wp_satsify_AP = [[[5.153910142707331, 8.475722259277205, 'T1_init', 'T0_S2'], [15.509597976223613, 41.326154298121445, 'T0_S2', 'accept_S3'], [23.902591083293558, 58.967541295641105, 'accept_S3', 'accept_S3']], [[10.1231993895458, 8.484584265546811, 'T1_init', 'T0_S2'], [24.441305754229795, 66.82018106004666, 'T0_S2', 'accept_S3'], [42.03044360503986, 22.15818674387984, 'accept_S3', 'accept_S3']], [[15.447213595499958, 8.223606797749978, 'T1_init', 'T0_S2'], [43.38941708675561, 20.96441783211289, 'T0_S2', 'accept_S3'], [96.26095897052463, 8.979984968314591, 'accept_S3', 'accept_S3']], [[20.433780334614408, 8.248665681793474, 'T1_init', 'T0_S2'], [74.58016138480517, 47.22131035408012, 'T0_S2', 'accept_S3'], [100.56156176580477, 54.71167675633696, 'accept_S3', 'accept_S3']], [[25.499951781667278, 8.006943774745379, 'T1_init', 'T0_S2'], [97.24934096146332, 8.344589918881281, 'T0_S2', 'accept_S3'], [128.0757374753, 11.00550620716955, 'accept_S3', 'accept_S3']]]
    # rob_waypoint = [14.834858118998467, 38.39865236781343, 'T1_init', 'T0_S2']
    # replanning_bool = task.Replanning_check(rob_waypoint, workspace, robot_wp_satsify_AP, 1, buchi_graph)
        
        
def generate_NBA():
    start = datetime.datetime.now()
    task = Task()
    buchi = Buchi(task)
    buchi.construct_buchi_graph()
    buchi.get_minimal_length()
    buchi.get_feasible_accepting_state()
    buchi_graph = buchi.buchi_graph
    NBA_time = (datetime.datetime.now() - start).total_seconds()
    print('Time for constructing the NBA: {0:.4f} s'.format(NBA_time))
    return buchi, buchi_graph

def generate_path(buchi, buchi_graph, workspace, init_state, save_waypoints = True, edit_launch_file=True, save_covariances = False):
    identity='run1'
    round_num = 1
    number_of_trials = 1
    drone_height = 16.0     # altitude of drones
    waypoint_folder_location = "/home/samarth/catkin_ws/src/rotors_simulator/rotors_gazebo/resource"
    launch_folder_location = "/home/samarth/catkin_ws/src/rotors_simulator/rotors_gazebo/launch"
    
    time_array=[]   #stores time for each trial run
    cost_array=[]   #stores cost of each trial run
    
    task = Task()
    
    geodesic = Geodesic(workspace, task.threshold)
    # parameters
    n_max = 100000
    para = dict()
    # lite version, excluding extending and rewiring
    para['is_lite'] = True
    # step_size used in function near
    para['step_size'] = 0.25 * buchi.number_of_robots
    # probability of choosing node q_p_closest
    para['p_closest'] = 0.9
    # probability used when deciding the target point
    para['y_rand'] = 0.99
    # probability used when deciding the target point when inside sensing range
    # more random actions to increase visibility of     
    # para['y_rand'] = 0.99
    # minimum distance between any pair of robots
    para['threshold'] = task.threshold
    # Updates landmark covariance when inside sensor range
    para['update_covariance'] = True
    # sensor range in meters
    para['sensor_range'] = 10
    # sensor measurement noise
    para['sensor_R'] = 0.5
    
    
    
    # initialization
    opt_cost = np.inf
    opt_path_pre = []
    opt_path_suf = []

    # ----------------------------------------------------------------#
    #                            Prefix Part                          #
    # ----------------------------------------------------------------#

    start = datetime.datetime.now()
    # init_state = (task.init, b_init)
    # init_state = (((28.13007841678492, 96.5686624041831),
    #           (11.655391499113755, 77.18077120816896),
    #           (39.53486085989735, 30.22294758970175),
    #           (58.96178129887859, 76.04140804427708),
    #           (98.3515083106104, 51.015593324837354)),
    #           'T2_S2')
    init_label = task.init_label
    init_angle = task.init_angle
    tree_pre = BiasedTree(workspace, geodesic, buchi, task, init_state, init_label, init_angle, 'prefix', para)
    
    # print('------------------------------ prefix path --------------------------------')
    # construct the tree for the prefix part
    cost_path_pre, nodes, lm_cov, targets = construction_biased_tree(tree_pre, n_max)
    if len(tree_pre.goals):
        pre_time = ((datetime.datetime.now() - start).total_seconds())/60
        print('Time for the prefix path: {0:.4f} min'.format(pre_time))
        print('{0} accepting goals found'.format(len(tree_pre.goals)))
        for i in range(len(tree_pre.goals)):
            if cost_path_pre[i][0] < opt_cost:
                opt_path_suf=[]
                opt_path_pre = cost_path_pre[i][1]
                opt_cost = cost_path_pre[i][0]
           
        # path_print((opt_path_pre, opt_path_suf), workspace, buchi.number_of_robots)
        path_plot((opt_path_pre, opt_path_suf), workspace, tree_pre.biased_tree.nodes[cost_path_pre[0][1][-1]]['lm'], buchi.number_of_robots, round_num, identity)
        plt.show()
        
        print('Time for the prefix path: {0:.4f} min'.format(pre_time))
        print('Cost of path: {}'.format(cost_path_pre[0][0]))
        print(' ')
        print(' ')
        print(len(cost_path_pre[0][1]))
        time_array.append(pre_time)
        cost_array.append(cost_path_pre[0][0])
        print(time_array)
        print(cost_array)
        if number_of_trials == 1 and  save_waypoints:
            robot_waypoints, robot_wp_satsify_AP = export_disc_to_txt(cost_path_pre, targets, buchi.number_of_robots, drone_height, waypoint_folder_location, launch_folder_location, 10, edit_launch_file)
        if number_of_trials == 1 and  save_covariances:
            export_cov_to_txt(lm_cov, waypoint_folder_location, launch_folder_location)
    else:
        print('Couldn\'t find the path within predetermined number of iteration')
        
    return robot_waypoints, robot_wp_satsify_AP
            


