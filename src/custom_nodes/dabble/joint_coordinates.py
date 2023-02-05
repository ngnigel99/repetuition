"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import cv2
import math
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import os.path
import statistics


# GLOBAL VARIABLES
# font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# colors (BGR color format)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREY = (178, 178, 178)

# threshold
THRESHOLD = 0.6

# KEYPOINT AND RESPECTIVE IDs
# https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# DEBUGGING TOOLS ENABLE/DISABLE
BODY_COORDINATE = False
DEBUG_CONSOLE = False
DEBUG_CONSOLE_DISPLAY_PARAMETERS = [
    ["LS: ", 0.03], ["RS: ", 0.08], ["LE: ", 0.13], ["RE: ", 0.18], ["LW: ", 0.23], ["RW: ", 0.28], [
        "LH: ", 0.33], ["RH: ", 0.38], ["LK: ", 0.43], ["RK: ", 0.48], ["LA: ", 0.53], ["RA: ", 0.58]
]


# check spine alignment
def check_spine_alignment(right_shoulder, right_hip, right_knee, right_ankle):
    if right_shoulder and right_hip and right_knee and right_ankle:
        angle = get_angle(right_shoulder, right_hip, right_knee)
        # write to a text file for debugging
        # with open("angle.txt", "a") as f:
        #   f.write(str(angle) + "\n")
        '''
      if angle < 0:
         return False
      angle = get_angle(right_hip, right_knee, right_ankle)
      if angle < 0:
         return False
      angle = get_angle(right_shoulder, right_hip, right_ankle)
      if angle < 0:
         return False
      angle = get_angle(right_shoulder, right_knee, right_ankle)
      if angle < 0:
         return False
      else:
         return True
         '''

# get angle between 3 points given x, y coordinates
def get_angle(a: tuple, b: tuple, c: tuple):
    return math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))

def get_distance(a: tuple, b: tuple):
    distance = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    # with open("distance.txt", "a") as f:
    #     f.write(str(distance) + "\n")
    return distance

def getAngleRange(inputFilePath):
    """
    given a text file containing angles, return range of acceptable points.
    this removes outliers, and also demarcates the range of acceptable reps.
    """
    with open(inputFilePath) as f:
        data = f.readlines()
        data = [float(x.strip()) for x in data]
        arr = np.array(data).reshape(-1, 1)
        arr = StandardScaler().fit_transform(arr)
        arr = arr[arr[:, 0] <= 1.5]
        arr = arr[arr[:, 0] >= -1.5]

        max_point = np.max(arr)
        min_point = np.min(arr)
        print(min_point, max_point)
        return min_point, max_point

def writeAngleCal(outputFilePath, knee, hip, shoulder):
    with open(outputFilePath, "a") as f:
        angle = get_angle(knee, hip, shoulder)
        f.write(str(angle) + "\n")

# given an input calibration file, return a tuple of min, max points with scaling
def map_keypoint_to_image_coords(keypoint, image_size):
    """Second helper function to convert relative keypoint coordinates to
    absolute image coordinates.
    Keypoint coords ranges from 0 to 1
    where (0, 0) = image top-left, (1, 1) = image bottom-right.

    Args:
       bbox (List[float]): List of 2 floats x, y (relative)
       image_size (Tuple[int, int]): Width, Height of image

    Returns:
       List[int]: x, y in integer image coords
    """

    # (keypoint: List[float], image_size: Tuple[int, int]) -> List[int]

    width, height = image_size[0], image_size[1]
    x, y = keypoint
    x *= width
    y *= height
    # because coordinates need to be type:int for cv2 object
    return int(x), int(y)

def draw_debug_text(img, coordinates: tuple, color_code, img_size: tuple, keypoint: int):
    """
    Helper function to draw the coordinate texts for joints

    img: image to draw console and coordinates on
    coordinates: int tuple of joint coordinates
    color_code: color of preference for text
    img_size: int tuple image dimensions
    keypoint: GLOBAL VARIABLE for joint keypoint
    """

    x, y = coordinates[0], coordinates[1]
    x_y_str = f"({x}, {y})"

    if BODY_COORDINATE:
        cv2.putText(
            img=img,
            text=x_y_str,
            org=(x, y),
            fontFace=FONT,
            fontScale=0.4,
            color=color_code,
            thickness=2,
        )

    if DEBUG_CONSOLE:

        lookup = keypoint-5

        cv2.putText(
            img=img,
            text=DEBUG_CONSOLE_DISPLAY_PARAMETERS[lookup][0] + x_y_str,
            org=(map_keypoint_to_image_coords(
                (0.81, DEBUG_CONSOLE_DISPLAY_PARAMETERS[lookup][1]), img_size)),
            fontFace=FONT,
            fontScale=0.4,
            color=color_code,
            thickness=2,
        )

def draw_debug_console(img, start_point=(0.8, 0), end_point=(1, 0.6), color_code=GREY, thickness=-1):
    """
    Draw box for metadata
    """
    img_size = (img.shape[1], img.shape[0])
    start_point = (map_keypoint_to_image_coords(start_point, img_size))
    end_point = (map_keypoint_to_image_coords(end_point, img_size))

    cv2.rectangle(
        img=img,
        pt1=start_point,
        pt2=end_point,
        color=color_code,
        thickness=thickness
    )

def draw_timer_box(img, current_time: int, end_time: int, img_size: tuple):
    '''
    Draw the black box backgroud for count and timer and, print updated timer values
    '''
    time_left = end_time - current_time

    cv2.rectangle(
        img=img,
        pt1=(map_keypoint_to_image_coords((0, 0), img_size)),
        pt2=(map_keypoint_to_image_coords((0.33, 0.13), img_size)),
        color=BLACK,
        thickness=-1
    )

    if time_left > 0:
        cv2.putText(
            img=img,
            text="TIMER: " + str(int(time_left)),
            org=(map_keypoint_to_image_coords((0.01, 0.06), img_size)),
            fontFace=FONT,
            fontScale=1,
            color=WHITE,
            thickness=2,
        )
        end_timer = False

    else:
        cv2.putText(
            img=img,
            text="TIMER: Time's up",
            org=(map_keypoint_to_image_coords((0.01, 0.06), img_size)),
            fontFace=FONT,
            fontScale=1,
            color=WHITE,
            thickness=2,
        )
        end_timer = True

    return end_timer

def draw_counter_text(img, img_size, count: int):

    cv2.putText(
        img=img,
        text="COUNT: " + str(count),
        org=(map_keypoint_to_image_coords((0.01, 0.12), img_size)),
        fontFace=FONT,
        fontScale=1,
        color=WHITE,
        thickness=2,
    )

    if count == 60:
        cv2.putText(
            img=img,
            text="COUNT: 60 - Max",
            org=(map_keypoint_to_image_coords((0.01, 0.12), img_size)),
            fontFace=FONT,
            fontScale=1,
            color=WHITE,
            thickness=2,
        )


def draw_calibrate_box(img, img_size):
    '''
    Draw the black box & text for calibration phase
    '''

    cv2.rectangle(
        img=img,
        pt1=(map_keypoint_to_image_coords((0.55, 0.92), img_size)),
        pt2=(map_keypoint_to_image_coords((1.0, 1.0), img_size)),
        color=BLACK,
        thickness=-1
    )

    cv2.putText(
        img=img,
        text="CALIBRATION PHASE",
        org=(map_keypoint_to_image_coords((0.56, 0.98), img_size)),
        fontFace=FONT,
        fontScale=0.9,
        color=WHITE,
        thickness=2,
    )


def draw_ippt_box(img, img_size):
    '''
    Draw the black box & text for IPPT phase
    '''

    cv2.rectangle(
        img=img,
        pt1=(map_keypoint_to_image_coords((0.72, 0.92), img_size)),
        pt2=(map_keypoint_to_image_coords((1.0, 1.0), img_size)),
        color=BLACK,
        thickness=-1
    )

    cv2.putText(
        img=img,
        text="IPPT PHASE",
        org=(map_keypoint_to_image_coords((0.73, 0.98), img_size)),
        fontFace=FONT,
        fontScale=0.9,
        color=WHITE,
        thickness=2,
    )


# Given a text file, fit and return a scaler
def get_scaler(file_name):
    with open(file_name) as f:
        data = f.readlines()
        data = [float(x.strip()) for x in data]
        data = np.array(data).reshape(-1, 1)
        sta = StandardScaler()
        data = sta.fit_transform(data)
        return sta

def check_spine_alignment(right_shoulder, right_hip, right_knee, minNoise, maxNoise, minRange, maxRange, scaler):
    if right_shoulder and right_hip and right_knee:
        angle = get_angle(right_knee, right_hip, right_shoulder)
        angle = np.array(angle).reshape(-1, 1)
        # use scaler to transform data
        angle = scaler.transform(angle)
        if angle > maxNoise or angle < minNoise:
            return True  # ignore noise
        if angle > maxRange or angle < minRange:
            return False
        return True

def depth_file_denoizer(coordinates_txt_file: str):
    '''
    Parse all the depth values in a txt file and replace them with only the denoised maximum and minimum in the txt file
    (To be used for depth calibration)
    '''
    with open(coordinates_txt_file, 'r') as file:
        data = [float(line) for line in file]
        data = data[0:]
        left_cutoff, right_cutoff = int(len(data) * 0.1), int(len(data) * 0.9)
        data = data[left_cutoff:right_cutoff]
        mean = sum(data)/len(data)
        stdev = statistics.stdev(data)
        upper_bound = mean + stdev
        lower_bound = mean - stdev

    denoized_data = [x for x in data if lower_bound <= x <= upper_bound]

    open(coordinates_txt_file, 'w').close()
    with open(coordinates_txt_file, 'w') as file:
        file.write(str(max(denoized_data))+"\n"+str(min(denoized_data)) + "\n")

def check_orientation(img_size:tuple, right_wrist, right_shoulder, right_ankle, left_wrist, left_shoulder, left_ankle) -> bool:
    """
    Check if the user's right side is facing the camera.
    Shoulders and wrists should be to the right 40% of the img,
    ankles should be lower than the shoulders and wrist and on the left 40% of img
    
    inputs: all respective keypoints
    """

    left_limit, right_limit = int(0.4 * img_size[0]), int(0.6 * img_size[0])

    if (right_wrist[0] >= right_limit) and (left_wrist[0] >= right_limit) and (right_shoulder[0] >= right_limit) and (left_shoulder[0] >= right_limit):
        if (right_ankle[0] <= left_limit) and (left_ankle[0] <= left_limit):
            if (right_ankle[1] > right_shoulder[1]) and (right_ankle[1] > left_shoulder[1]):
                if (left_ankle[1] > right_shoulder[1]) and (left_ankle[1] > left_shoulder[1]):
                    return True

    return False

class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")

        # system attributes
        self.isCalibrated = True

        # time
        self.start_time = 0
        self.end_time = 0
        self.timer = True
        self.timer_has_started = False
        self.timer_has_ended = False

        # pushup attributes
        self.pushupTopHeight = 0
        self.pushupBottomHeight = 0
        self.isHighEnough = False
        self.isLowEnough = False
        self.pushupCount = 0
        self.counterGUI = False

        # pushup scaling
        self.sta = StandardScaler()  # for scaling values
        # self.scaler = get_scaler('proper_angle.txt') @Nigel

        # check if user is facing the right
        self.orientationisright = False

        # Check if the system has been calibrated to start testing, else calibrate first
        if os.path.isfile('distance.txt'):
            print('Distance calibrated! IPPT in progress...')
            self.isCalibrated = True
            # Get calibrated max min values
            depth_file_denoizer('distance.txt')
            distanceFile = open('distance.txt', 'r')
            Lines = [float(line) for line in distanceFile]
            Lines = Lines[0:]
            difference = Lines[0] - Lines[1]
            self.pushupTopHeight = Lines[0] - 0.3*difference
            self.pushupBottomHeight = Lines[1] + 0.2*difference
            print('Max = ' + str(self.pushupTopHeight))
            print('Min = ' + str(self.pushupBottomHeight))
        else:
            self.isCalibrated = False
            print('Calibrating distance now...')

        # calibration data for angle
        self.angleCalibrated = False
        if os.path.exists("angle.txt"):
            print('Angle calibrated! IPPT in progress...')
            self.angleCalibrated = True
            self.angleScaler = get_scaler("angle.txt")
            self.minNoise = -1.5    # scaled so 1.5 s.d.
            self.maxNoise = 1.5
            self.minRange, self.maxRange = getAngleRange("angle.txt")
        else:
            print("Calibrating angle now...")
            self.defaultAngleScaler = get_scaler("defaultAngle.txt")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys
                "img", "keypoints", "keypoint_scores"

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        # variables
        # get required inputs from pipeline
        img = inputs["img"]
        keypoints = inputs["keypoints"]
        keypoint_scores = inputs["keypoint_scores"]
        # --> vector coordinates for lines connecting joints
        keypoint_conns = inputs["keypoint_conns"]

        # derived inputs from raw inputs
        img_size = (img.shape[1], img.shape[0])

        # pre-declaring co-ordinates
        left_shoulder = None
        right_shoulder = None
        left_elbow = None
        right_elbow = None
        left_wrist = None
        right_wrist = None
        left_hip = None
        right_hip = None
        left_knee = None
        right_knee = None
        left_ankle = None
        right_ankle = None

        # detecting keypoints on screen
        if len(keypoints) >= 1:  # check if can do == 1
            the_keypoints = keypoints[0]
            the_keypoint_scores = keypoint_scores[0]

            if DEBUG_CONSOLE:
                draw_debug_console(img)
            if self.timer and self.timer_has_started:
                self.timer_has_ended = draw_timer_box(
                    img, time.time(), self.end_time, img_size)
            if self.counterGUI:
                draw_counter_text(img, img_size, self.pushupCount)
            if self.isCalibrated == True:
                draw_ippt_box(img, img_size)
            if self.isCalibrated == False:
                draw_calibrate_box(img, img_size)

            for i, keypoints in enumerate(the_keypoints):
                keypoint_score = the_keypoint_scores[i]

                if keypoint_score >= THRESHOLD:
                    threshold_color = GREEN
                else:
                    threshold_color = RED

                if i == KP_LEFT_SHOULDER:
                    left_shoulder = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, left_shoulder,
                                    threshold_color, img_size, KP_LEFT_SHOULDER)
                elif i == KP_RIGHT_SHOULDER:
                    right_shoulder = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, right_shoulder,
                                    threshold_color, img_size, KP_RIGHT_SHOULDER)
                elif i == KP_LEFT_ELBOW:
                    left_elbow = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, left_elbow,
                                    threshold_color, img_size, KP_LEFT_ELBOW)
                elif i == KP_RIGHT_ELBOW:
                    right_elbow = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, right_elbow,
                                    threshold_color, img_size, KP_RIGHT_ELBOW)
                elif i == KP_LEFT_WRIST:
                    left_wrist = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, left_wrist,
                                    threshold_color, img_size, KP_LEFT_WRIST)
                elif i == KP_RIGHT_WRIST:
                    right_wrist = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, right_wrist,
                                    threshold_color, img_size, KP_RIGHT_WRIST)
                elif i == KP_LEFT_HIP:
                    left_hip = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(
                        img, left_hip, threshold_color, img_size, KP_LEFT_HIP)
                elif i == KP_RIGHT_HIP:
                    right_hip = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(
                        img, right_hip, threshold_color, img_size, KP_RIGHT_HIP)
                elif i == KP_LEFT_KNEE:
                    left_knee = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(
                        img, left_knee, threshold_color, img_size, KP_LEFT_KNEE)
                elif i == KP_RIGHT_KNEE:
                    right_knee = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, right_knee,
                                    threshold_color, img_size, KP_RIGHT_KNEE)
                elif i == KP_LEFT_ANKLE:
                    left_ankle = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, left_ankle,
                                    threshold_color, img_size, KP_LEFT_ANKLE)
                elif i == KP_RIGHT_ANKLE:
                    right_ankle = (map_keypoint_to_image_coords(
                        keypoints.tolist(), img_size))
                    draw_debug_text(img, right_ankle,
                                    threshold_color, img_size, KP_RIGHT_ANKLE)

            if right_wrist and right_shoulder:
                wrist_to_shoulder_distance = get_distance(right_wrist, right_shoulder)

            if self.orientationisright:

                if self.isCalibrated == True:
                    # Run actual test
                    # check whether keypoints are  not aligned in a straight line. if so, increment count in output for debug
                    if wrist_to_shoulder_distance <= self.pushupBottomHeight:
                        self.isLowEnough = True
                        print('Low enough')

                    if self.isLowEnough and wrist_to_shoulder_distance >= self.pushupTopHeight:
                        self.isHighEnough = True
                        print('High enough')

                        if wrist_to_shoulder_distance >= self.pushupTopHeight:
                            self.isHighEnough = True

                        if self.isLowEnough and self.isHighEnough and self.timer_has_ended == False:
                            self.isHighEnough = False
                            self.isLowEnough = False
                            self.pushupCount += 1
                            print(self.pushupCount)

                        if self.pushupCount == 1:
                            self.start_time = time.time()
                            self.end_time = self.start_time + 30
                            self.timer = True
                            self.timer_has_started = True
                            self.counterGUI = True
                    else:
                        # Run calibration
                        # print out on a text file called distance.txt
                        with open("distance.txt", "a") as f:
                            if wrist_to_shoulder_distance > 0:
                                f.write(str(wrist_to_shoulder_distance) + "\n")

                    if right_shoulder and right_hip and right_knee:
                        if self.angleCalibrated:
                            spine_aligned = check_spine_alignment(
                                right_shoulder, right_hip, right_knee, self.minNoise, self.maxNoise, self.minRange, self.maxRange, self.angleScaler)
                        else:   # default values w/o user calibration
                            writeAngleCal("angle.txt", right_knee,
                                        right_hip, right_shoulder)
                            spine_aligned = check_spine_alignment(
                                right_shoulder, right_hip, right_knee, -1.5, 1.5, -1.19, 1.19, self.defaultAngleScaler)
                        if spine_aligned == False:
                            print("Spine not aligned")  # debug
                            self.spineAligned = False

                    if self.pushupCount == 1:
                        self.start_time = time.time()
                        self.end_time = self.start_time + 40
                        self.timer = True
                        self.timer_has_started = True
                        self.counterGUI = True
                else:
                    # Run distance calibration
                    # print out on a text file called distance.txt
                    with open("distance.txt", "a") as f:
                        if wrist_to_shoulder_distance > 0:
                            f.write(str(wrist_to_shoulder_distance) + "\n")

            elif right_wrist and right_shoulder and right_ankle and left_wrist and left_shoulder and left_ankle:
                self.orientationisright = check_orientation(img_size, right_wrist, right_shoulder, right_ankle, left_wrist, left_shoulder, left_ankle)
                    
            
            print(self.orientationisright)

        return {}

        # result = do_something(inputs["in1"], inputs["in2"])
        # outputs = {"out1": result}
        # return outputs
