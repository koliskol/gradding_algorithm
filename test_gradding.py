from ast import If
import imp
import cv2
import numpy as np
from numpy import unique
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from sklearn.cluster import MeanShift
from sklearn.datasets import make_classification
from fastdtw import fastdtw
import more_itertools
import os
import math


def load_npy(add_temp):
    pose = np.load(add_temp)

    return pose

def load_template(add_temp, exc):
    #print(add_temp)
    files = os.listdir(add_temp)
    poses_tmp = []
    poses_angle_tmp = []
    for f in files:
        add_t = os.path.join(add_temp, f)
        #print(add_t)
        pose_tmp = load_npy(add_t)
        pose_tmp = np.asarray(pose_tmp)
        pose_angle_tmp = []
        for p in pose_tmp:
            p_angle = extract_angle(p, exc)
            pose_angle_tmp.append(p_angle)
        
        pose_angle_tmp = np.asarray(pose_angle_tmp)
        pose_angle_tmp = np.expand_dims(pose_angle_tmp, axis=2)
        #print(pose_tmp.shape)
        #print(pose_angle_tmp.shape)
        poses_tmp.append(pose_tmp)
        poses_angle_tmp.append(pose_angle_tmp)

    return poses_tmp, poses_angle_tmp

def compare_pose(ref_pose, cur_pose):
    
    dist_min = 9999
    er_min = 9999
    er = 0
    idx = 0
    id_ctr = 0

    for p in ref_pose:
        er = cosine(p, cur_pose)
        if(er < dist_min):
            er_min = er
            dist_min = er
            idx = id_ctr
        id_ctr = id_ctr + 1
        
    return er_min, idx

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang



def extract_angle(user_joints, type_ex):

    '''
            Index of the joint for each model
            ==== TF lite ====       ==== Mediapie =====
            'nose': 0,                0 : nose  
            'left_eye': 1,            5 : right eye
            'right_eye': 2,           2 : left eye
            'left_ear': 3,            8 : right ear
            'right_ear': 4,           7 : left ear
            'left_shoulder': 5,       12 : right shoulder
            'right_shoulder': 6,      11 : left shoulder
            'left_elbow': 7,          14 : right elbow
            'right_elbow': 8,         13 : left elbow
            'left_wrist': 9,          16 : right wrist
            'right_wrist': 10,        15 : left wrist
            'left_hip': 11,           23 : left hip
            'right_hip': 12,          24 : right hip
            'left_knee': 13,          25 : left knee
            'right_knee': 14,         26 : right knee
            'left_ankle': 15,         27 : left ankle
            'right_ankle': 16         28 : right ankle
    '''

    angle_ex = []

     #if(side == "right"):
    left_trunk = getAngle(user_joints[5], user_joints[11], user_joints[13]) #trunk = shoulder, hip,knee
    right_trunk = getAngle(user_joints[6], user_joints[12], user_joints[14])
    left_leg = getAngle(user_joints[11], user_joints[13], user_joints[15]) #leg = hip, knee, ankle
    right_leg = getAngle(user_joints[12], user_joints[14], user_joints[16])
    left_arm = getAngle(user_joints[7], user_joints[5], user_joints[11]) #arm = elbow, shoulder, hip
    right_arm = getAngle(user_joints[8], user_joints[6], user_joints[12])
    left_hand = getAngle(user_joints[5], user_joints[7], user_joints[9]) #hand = shoulder, elbow, wrist
    right_hand = getAngle(user_joints[6], user_joints[8], user_joints[10])
    left_backbone = getAngle(user_joints[0], user_joints[11], user_joints[15])
    right_backbone = getAngle(user_joints[0], user_joints[12], user_joints[16])
    

    if(type_ex == "kneepushup" or type_ex == "birddog" or type_ex == "sumosquat" or type_ex == "squat"):
        
       
        angle_ex.append(left_trunk)
        angle_ex.append(right_trunk)
        angle_ex.append(left_leg)
        angle_ex.append(right_leg)
        angle_ex.append(left_arm)
        angle_ex.append(right_arm)
       
    elif type_ex == "reversefly":

        angle_ex.append(left_trunk)
        angle_ex.append(right_trunk)
        angle_ex.append(left_arm)
        angle_ex.append(right_arm)
        angle_ex.append(left_hand)
        angle_ex.append(right_hand)

    elif type_ex == "superman":
        angle_ex.append(left_arm)
        angle_ex.append(right_arm)
        angle_ex.append(left_trunk)
        angle_ex.append(right_trunk)
    
    elif type_ex == "reverselunge":
        angle_ex.append(left_trunk)
        angle_ex.append(right_trunk)
        angle_ex.append(left_leg)
        angle_ex.append(right_leg)
        angle_ex.append(left_hand)
        angle_ex.append(right_hand)
    
    elif type_ex == "fullplank":
        angle_ex.append(left_arm)
        angle_ex.append(right_arm)
        angle_ex.append(left_backbone)
        angle_ex.append(right_backbone)
    
    elif type_ex == "sideplank":
        angle_ex.append(left_arm)
        angle_ex.append(right_arm)
        angle_ex.append(left_backbone)
        angle_ex.append(right_backbone)
        angle_ex.append(left_trunk)
        angle_ex.append(right_trunk)
        angle_ex.append(left_leg)
        angle_ex.append(right_leg)
        angle_ex.append(left_hand)
        angle_ex.append(right_hand)
    
    
    return angle_ex


def gradingDTW_peak(user_joints, templ_joints):

    first_user_joint = user_joints[0]

    s1, s2 = first_user_joint.shape
    first_user_joint = np.reshape(first_user_joint, (1, s1*s2))

    errs_max = []
    idxx_max = []
    errs_all = []
    idxx_all = []
    X = []

    ctr = 0
    th = 0.003
    len_ = len(user_joints)

    for tmp in user_joints:
        tmp_  = np.reshape(tmp, (1, s1*s2))
        er, idx = compare_pose(first_user_joint, tmp_)
        errs_all.append(er)
        idxx_all.append(ctr)

        
        if(er > th):
            errs_max.append(er)
            idxx_max.append(ctr)
            X.append([er, (int(ctr) / len_)])
            
        
        ctr = ctr+1

    c = (np.diff(np.sign(np.diff(errs_max))) < 0).nonzero()[0] + 1    # local max

    idxx_max = np.asarray(idxx_max)
    errs_max = np.asarray(errs_max)
    idxx_all = np.asarray(idxx_all)
    errs_all = np.asarray(errs_all)

    model = MeanShift()
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    id_peak = []
    
    for cluster in clusters:
        id_peak.append(idxx_max[np.where(errs_max == np.max(errs_max[np.where(yhat == cluster)]))][0])
 
    id_peak = np.sort(id_peak)
    id_peak = np.array(id_peak)
  

    id_cluster = []
    draw_id_cluster = []
    id_cluster.append((0, id_peak[0]))
    draw_id_cluster.append(0)

    for id in range(0, len(id_peak)-1):
        next_id = id_peak[id] + int((id_peak[id+1] - id_peak[id])/2)
        id_cluster.append((id_peak[id], next_id))

        draw_id_cluster.append(id_peak[id])
        draw_id_cluster.append(next_id)
    
    id_cluster.append((id_peak[len(id_peak)-1], len(user_joints)-1))

    #calculate the Grade pose ===========================================
    error_sum = 0
    error_ctr = 0

    tmpl_j = np.asarray(templ_joints)
    f, j, c = tmpl_j.shape
    tmpl_j = np.reshape(tmpl_j, (f, j*c))

    min_avg = 9999

    for c in id_cluster:
        vid_chk = []
        #print("pair {} and {}".format(c[0], c[1]))
        for p in range(c[0], c[1]):
            vid_chk.append(user_joints[p])
        
        vid_chk = np.asarray(vid_chk)
        f, j, c = vid_chk.shape
        vid_chk = np.reshape(vid_chk, (f, j*c))
        error, _ = fastdtw(tmpl_j, vid_chk, dist=cosine)
        if(min_avg >= error):
            min_avg = error

    return min_avg

def gradingDTW_sliding_window(user_joints, templ_joints, wind, steps):

    min_avg = 9999

    lst_ft = list(range(len(user_joints)))
    res = list(more_itertools.windowed(lst_ft, wind, step=steps))

    tmpl_j = np.asarray(templ_joints)    
    #print(tmpl_j.shape)
    f, j, c = tmpl_j.shape
    tmpl_j = np.reshape(tmpl_j, (f, j*c))

    for i in range(0, len(res)):
        usr_j = []
        for r in res[i]:
            usr_j.append(user_joints[r])
        
        usr_j = np.asarray(usr_j)
        s = usr_j.shape

        if(len(s) >= 3):
            f, j, c = usr_j.shape
            usr_j = np.reshape(usr_j, (f, j*c))
            error, _ = fastdtw(tmpl_j, usr_j, dist=cosine)
            if(min_avg >= error):
                min_avg = error
    

    return min_avg

def grading_all(pose_user, poses_template, peak = False):
    wind = 20
    steps = int(wind/2)

    print(pose_user.shape)
    print(np.asanyarray(poses_template).shape)

    if(peak == False):

        print("=================== grading sliding window ===============================")
        avg_min = 9999
        for i in range(0, len(poses_template)):
            templ_joints = poses_template[i]
            
            avg_err = gradingDTW_sliding_window(pose_user, templ_joints, wind, steps)
            if(avg_min >= avg_err):
                avg_min = avg_err
        print("average min sliding window: ", avg_min)
        
    else:
        avg_min = 9999
        print("================== grading peak =========================================")
        for i in range(0, len(poses_template)):
            templ_joints = poses_template[i]
            avg_err = gradingDTW_peak(pose_user, templ_joints)
            if(avg_min >= avg_err):
                avg_min = avg_err

        print("average min peak: ", avg_min)
    
    return avg_min

def testMultiplePersonInput():
    #Example using many input array, for testing only ===================================================================
    status_grade = "Good Pose Excercise"
    excercise = "kneepushup"
    add = "Pose_test/kneepushup/"
    list_pose = os.listdir(add)
    print(list_pose)

    for l in list_pose:
        pose_example = os.path.join(add, l)
        pose_good = load_npy(pose_example)
        pose_good = np.asarray(pose_good)
        print(pose_good.shape)

        pose_angle_good = []

        for pose in pose_good:
        
            p_angle = extract_angle(pose, excercise)
            pose_angle_good.append(p_angle)
                
        pose_angle_good = np.asarray(pose_angle_good)
        pose_angle_good = np.expand_dims(pose_angle_good, axis=2)
        print(pose_angle_good.shape)

        folder_tmp = 'Pose_template/'+excercise+'/'
        poses_template, poses_angle_template = load_template(folder_tmp, excercise)

        print("========================== Good Pose =====================================")
        avg_min = grading_all(pose_angle_good, poses_angle_template, peak = False) #using sliding window
        print("grading score: ", avg_min)
        th_excelent = 0.75
        th_low = 1.10
        if avg_min <= th_excelent:
            status_grade = "Excelent Pose Excercise"
        elif avg_min > th_excelent and avg_min < th_low:
            status_grade = "Good Pose Excercise"
        else:
            status_grade = "Fair Pose Excercise"
        
        print("Status Grade: ", status_grade)

        

def testOnePersonInput(poseInput):

        status_grade = "Good Pose Excercise" 
        print(poseInput.shape)
        pose_angle_good = []

        for pose in poseInput:
        
            p_angle = extract_angle(pose, excercise)
            pose_angle_good.append(p_angle)
                
        pose_angle_good = np.asarray(pose_angle_good)
        pose_angle_good = np.expand_dims(pose_angle_good, axis=2)
        print(pose_angle_good.shape)

        folder_tmp = 'Pose_template/'+excercise+'/'
        poses_template, poses_angle_template = load_template(folder_tmp, excercise)

        print("========================== Good Pose =====================================")
        avg_min = grading_all(pose_angle_good, poses_angle_template, peak = False) #using sliding window
        print("grading score: ", avg_min)

        th_excelent = 0.75
        th_low = 1.10
        if avg_min <= th_excelent:
            status_grade = "Excelent Pose Excercise"
        elif avg_min > th_excelent and avg_min < th_low:
            status_grade = "Good Pose Excercise"
        else:
            status_grade = "Fair Pose Excercise"

        return status_grade


#how to use testOnePersonInput ==========================================================================
excercise = "kneepushup"
add = "kneePushup_test.npy"
pose_Input= load_npy(add)
pose_Input = np.asarray(pose_Input)

status_grade = testOnePersonInput(pose_Input)
print("Status Grade: ", status_grade)


