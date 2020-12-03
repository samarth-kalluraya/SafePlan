#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA

from kf import kf_update

got_image_1 = False
got_image_2 = False
got_image_3 = False
got_image_4 = False
got_image_5 = False

marker_1 = Marker()
marker_2 = Marker()
marker_3 = Marker()
marker_4 = Marker()
marker_5 = Marker()


center_est = [(14, 137), (21, 62), (21, 55), (11, 48), (48, 26), (92, 14), (71, 42), 
                  (104, 86), (109, 131), (125, 15), (105, 52), (131, 91), (125, 48)]
landmark_est= {'l1': [[center_est[0][0], center_est[0][1]],  [[3, 0], [0, 3]]],
            'l2': [[center_est[1][0], center_est[1][1]],  [[3, 0], [0, 4]]],
            'l3': [[center_est[2][0], center_est[2][1]],  [[3, 0], [0, 3]]],
            'l4': [[center_est[3][0], center_est[3][1]],  [[1, 0], [0, 5]]],
            'l5': [[center_est[4][0], center_est[4][1]],  [[5, 0], [0, 4]]],
            'l6': [[center_est[5][0], center_est[5][1]],  [[4, 0], [0, 2]]],
            'l7': [[center_est[6][0], center_est[6][1]],  [[3, 0], [0, 4]]],
            'l8': [[center_est[7][0], center_est[7][1]],  [[2, 0], [0, 2]]],
            'l9': [[center_est[8][0], center_est[8][1]],  [[1, 0], [0, 1]]],
            'l10': [[center_est[9][0], center_est[9][1]],  [[2, 0], [0, 2]]],
            'l11': [[center_est[10][0], center_est[10][1]],  [[2, 0], [0, 2]]],
            'l12': [[center_est[11][0], center_est[11][1]],  [[2, 0], [0, 2]]],
            'l13': [[center_est[12][0], center_est[12][1]],  [[2, 0], [0, 2]]]
            }
classes_est = {'l1': [np.array([0.7, 0.25, 0.05])],
            'l2': [np.array([0.35, 0.6, 0.05])],
            'l3': [np.array([0.6, 0.35, 0.05])],
            'l4': [np.array([0.25, 0.7, 0.05])],
            'l5': [np.array([0.15, 0.8, 0.05])],
            'l6': [np.array([0.82, 0.13, 0.05])],
            'l7': [np.array([0.65, 0.3, 0.05])],
            'l8': [np.array([0.22, 0.73, 0.05])],
            'l9': [np.array([0.72, 0.23, 0.05])],
            'l10': [np.array([0.64, 0.31, 0.05])],
            'l11': [np.array([0.12, 0.83, 0.05])],
            'l12': [np.array([0.02, 0.03, 0.95])],
            'l13': [np.array([0.21, 0.7, 0.09])]
            }
num_of_classes = 3                   

sensor_noise_cov = [[0.3, 0], [0, 0.3]]
"""
example of defining the sensor model:
sensor_model = [[P(person|person), 	P(person|car), 	P(person|bike)],
				[P(car|person), 	P(car|car), 	P(car|bike)],
				[P(bike|person),	P(bike|car), 	P(bike|bike)]]
"""
sensor_model = [[0.80, 0.18, 0.02],
				[0.23, 0.75, 0.02],
				[0.06, 0.04, 0.9]]  


# returns probability of classes such that class_id will have higher probability with a higher probability
def get_nn_prob(class_id):
    result = np.zeros((num_of_classes,))
    prob = np.random.normal(0.75, 0.1, 1)
    while prob>0.99 or prob<0.2:
        prob = np.random.normal(0.75, 0.1, 1)
    result[class_id-1] = prob
    count = 1
    for i in range(num_of_classes):
        if i!= class_id-1:
            count+=1
            remaining_prob = 1 - result.sum()
            prob = np.random.uniform(0.01,remaining_prob,1)
            if count==num_of_classes:
                result[i] = remaining_prob
            else:
                result[i] = prob
    return result

# function to find if given point lies inside a given rectangle or not. 
# FOV = [x1,y1,x2,y2]
def in_FOV(FOV, pos) : 
    if (pos[0] > FOV[0] and pos[0] < FOV[2] and 
        pos[1] > FOV[1] and pos[1] < FOV[3]) : 
        return True
    else : 
        return False

def generate_nn_output(data):
	center = [(9.6, 132), (25, 67), (24, 59), (16, 42), (43, 22), (97, 9), (75, 47), 
                  (100, 91), (104, 136), (128, 10), (100, 55), (135, 95), (129, 48)]
	landmark_gt = {'l1': [center[0][0], center[0][1]],
	            'l2': [center[1][0], center[1][1]],
	            'l3': [center[2][0], center[2][1]],
	            'l4': [center[3][0], center[3][1]],
	            'l5': [center[4][0], center[4][1]],
	            'l6': [center[5][0], center[5][1]],
	            'l7': [center[6][0], center[6][1]],
	            'l8': [center[7][0], center[7][1]],
	            'l9': [center[8][0], center[8][1]],
	            'l10': [center[9][0], center[9][1]],
	            'l11': [center[10][0], center[10][1]],
	            'l12': [center[11][0], center[11][1]],
	            'l13': [center[12][0], center[12][1]]
	            }
	landmark_class_gt ={'l1': [1, "person"],
	            'l2': [2, "walking_person"],
	            'l3': [1, "person"],
	            'l4': [2, "walking_person"],
	            'l5': [2, "walking_person"],
	            'l6': [1, "person"],
	            'l7': [1, "person"],
	            'l8': [2, "walking_person"],
	            'l9': [1, "person"],
	            'l10': [1, "person"],
	            'l11': [2, "walking_person"],
	            'l12': [3, "police_station"],
	            'l13': [2, "walking_person"]
	            } 
	drone_x =  data.pose.pose.position.x
	drone_y =  data.pose.pose.position.y 
	drone_z =  data.pose.pose.position.z
	FOV = [drone_x-drone_z*0.75, drone_y-drone_z*0.5, drone_x+drone_z*0.75, drone_y+drone_z*0.5]
	nn_output={}
	lm_with_noise = []
	for key in landmark_gt.keys():
		if in_FOV(FOV, landmark_gt[key]):
			lm_with_noise = np.random.multivariate_normal(landmark_gt[key], sensor_noise_cov, 1).reshape(2,).tolist()
			probabiltiy = get_nn_prob(landmark_class_gt[key][0])
			nn_output[key] = [landmark_class_gt[key],lm_with_noise, probabiltiy]
	print("\n\n\nthis is what camera sees inside function")
	# print(nn_output)
	return nn_output

def update_landmark_estimates(nn_output):
	global landmark_est
	for key in nn_output.keys():
		x_estimate = landmark_est[key][0]
		cov = landmark_est[key][1]
		obs = nn_output[key][1]
		x_estimate, cov = kf_update(cov, x_estimate, obs, sensor_noise_cov)
		landmark_est[key][0] = x_estimate
		landmark_est[key][1]= cov

def update_class_distribution(nn_output):
	global classes_est
	for key in nn_output.keys():
		class_id = nn_output[key][0][0] - 1
		cond_prob = np.empty((num_of_classes,), float)
		cond_prob[class_id] = nn_output[key][2][class_id]
		for i in range(num_of_classes):
			if i!=class_id:
				cond_prob[i] = sensor_model[class_id][i]
		classes_est[key] = classes_est[key]*cond_prob/(classes_est[key]*cond_prob).sum()


def show_landmark(data,key_id):
	global marker_1 
	marker_1.header = data.header
	marker_1.type = Marker.CYLINDER
	# marker_1.pose.position.x = lm_with_noise[0]
	# marker_1.pose.position.y = lm_with_noise[1]
	marker_1.pose.position.x = landmark_est[key_id][0][0]
	marker_1.pose.position.y = landmark_est[key_id][0][1]
	marker_1.pose.position.z  = 1
	# rospy.loginfo(marker_1.pose.position)
	marker_1.scale.x = 0.50
	marker_1.scale.y = 0.50
	marker_1.scale.z = 4
	marker_1.color=ColorRGBA(0.013, 0.01, 0.9, 0.8)
	marker_1.lifetime=rospy.Duration()
	marker_pub_1.publish(marker_1)





def detect_in_vid1(data):
	global got_image_1
	got_image_1 = True

def detect_in_vid2(data):
	global got_image_2
	got_image_2 = True

def detect_in_vid3(data):
	global got_image_3
	got_image_3 = True
	
def detect_in_vid4(data):
	global got_image_4
	got_image_4 = True
	
def detect_in_vid5(data):
	global got_image_5
	got_image_5 = True


def odom_cb1(data):
	global got_image_1
	global landmark_est
	global classes_est
	if got_image_1:
		nn_output = generate_nn_output(data)
		got_image_1 = False
		update_landmark_estimates(nn_output)
		update_class_distribution(nn_output)
		print(classes_est)
		show_landmark(data,'l4')
# 2
def odom_cb2(data):
	global got_image_2
	global landmark_est
	global classes_est
	if got_image_2:
		nn_output = generate_nn_output(data)
		got_image_2 = False
		update_landmark_estimates(nn_output)
		update_class_distribution(nn_output)
		print(nn_output)
		show_landmark(data,'l4')

# 3
def odom_cb3(data):
	global got_image_3
	global landmark_est
	global classes_est
	if got_image_3:
		nn_output = generate_nn_output(data)
		got_image_3 = False
		update_landmark_estimates(nn_output)
		update_class_distribution(nn_output)
		show_landmark(data,'l4')

# 4
def odom_cb4(data):
	global got_image_4
	global landmark_est
	global classes_est
	if got_image_4:
		nn_output = generate_nn_output(data)
		got_image_4 = False
		update_landmark_estimates(nn_output)
		update_class_distribution(nn_output)
		show_landmark(data,'l4')

# 5
def odom_cb5(data):
	global got_image_5
	global landmark_est
	global classes_est
	if got_image_5:
		nn_output = generate_nn_output(data)
		got_image_5 = False
		update_landmark_estimates(nn_output)
		update_class_distribution(nn_output)
		show_landmark(data,'l4')









if __name__=="__main__":
	


	rospy.init_node('neural_network', anonymous=True)



	marker_pub_1 = rospy.Publisher('/firefly1/camera/visualization_marker', Marker, queue_size = 10)
	# marker_pub_2 = rospy.Publisher('/firefly2/camera/visualization_marker', Marker, queue_size = 10)
	# marker_pub_3 = rospy.Publisher('/firefly3/camera/visualization_marker', Marker, queue_size = 10)
	# marker_pub_4 = rospy.Publisher('/firefly4/camera/visualization_marker', Marker, queue_size = 10)
	# marker_pub_5 = rospy.Publisher('/firefly5/camera/visualization_marker', Marker, queue_size = 10)


	vid_sub_1 = rospy.Subscriber("/firefly1/vi_sensor/left/image_raw", Image, detect_in_vid1, queue_size = 1)
	vid_sub_2 = rospy.Subscriber("/firefly2/vi_sensor/left/image_raw", Image, detect_in_vid2, queue_size = 1)
	vid_sub_3 = rospy.Subscriber("/firefly3/vi_sensor/left/image_raw", Image, detect_in_vid3, queue_size = 1)
	vid_sub_4 = rospy.Subscriber("/firefly4/vi_sensor/left/image_raw", Image, detect_in_vid4, queue_size = 1)
	vid_sub_5 = rospy.Subscriber("/firefly5/vi_sensor/left/image_raw", Image, detect_in_vid5, queue_size = 1)

	odom_sub_1 = rospy.Subscriber("/firefly1/odometry_sensor1/odometry", Odometry, odom_cb1, queue_size = 1)
	odom_sub_2 = rospy.Subscriber("/firefly2/odometry_sensor1/odometry", Odometry, odom_cb2, queue_size = 1)
	odom_sub_3 = rospy.Subscriber("/firefly3/odometry_sensor1/odometry", Odometry, odom_cb3, queue_size = 1)
	odom_sub_4 = rospy.Subscriber("/firefly4/odometry_sensor1/odometry", Odometry, odom_cb4, queue_size = 1)
	odom_sub_5 = rospy.Subscriber("/firefly5/odometry_sensor1/odometry", Odometry, odom_cb5, queue_size = 1)


	print("x\nx\nx\nx\nx\nx\nx\nIn nn sim \nx\nx\nx\nx\nx\nx\nx\n")



	rospy.spin()
