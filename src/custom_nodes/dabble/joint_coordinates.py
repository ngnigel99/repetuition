"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

# GLOBAL CONSTANTS
## font
FONT = cv2.FONT_HERSHEY_SIMPLEX

## colors (BGR color format)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREY = (178, 178, 178)

## threshold
THRESHOLD = 0.6

# KEYPOINT AND RESPECTIVE IDs
## https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
DEBUG_CONSOLE_DISPLAY_PARAMETERS = [
   ["LS: ", 0.03], ["RS: ", 0.08], ["LE: ", 0.13], ["RE: ", 0.18], ["LW: ", 0.23], ["RW: ", 0.28]
]

# DEBUGGING TOOLS ENABLE/DISABLE
BODY_COORDINATE = True
DEBUG_CONSOLE = True

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
   return int(x), int(y) ## because coordinates need to be type:int for cv2 object

def draw_text(img, coordinates: tuple, color_code, img_size: tuple, keypoint: int):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """

   x, y = coordinates[0], coordinates[1]
   x_y_str = f"({x}, {y})"

   if BODY_COORDINATE:
      cv2.putText(
         img=img,
         text=x_y_str,
         org=(x, y),
         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
         fontScale=0.4,
         color=color_code,
         thickness=2,
      )
   
   if DEBUG_CONSOLE:

      lookup = keypoint-5

      cv2.putText(
         img=img,
         text=DEBUG_CONSOLE_DISPLAY_PARAMETERS[lookup][0] + x_y_str,
         org=(map_keypoint_to_image_coords((0.81, DEBUG_CONSOLE_DISPLAY_PARAMETERS[lookup][1]), img_size)),
         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
         fontScale=0.4,
         color=color_code,
         thickness=2,
      )

def draw_box(img, start_point=(0.8,0), end_point=(1,0.3), color=GREY, thickness=-1):
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
      color=color,
      thickness=thickness
   )

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

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
      """This node does ___.

      Args:
         inputs (dict): Dictionary with keys
            "img", "keypoints", "keypoint_scores"

      Returns:
         outputs (dict): Dictionary with keys "__".
      """

      # variables
      ## get required inputs from pipeline
      img = inputs["img"]
      keypoints = inputs["keypoints"]
      keypoint_scores = inputs["keypoint_scores"]
      keypoint_conns = inputs["keypoint_conns"] #--> vector coordinates for lines connecting joints

      ## derived inputs from raw inputs
      img_size = (img.shape[1], img.shape[0])

      ## pre-declaring co-ordinates
      left_shoulder = None
      right_shoulder = None
      left_elbow = None
      right_elbow = None
      left_wrist = None
      right_wrist = None

      # detecting keypoints on screen
      if len(keypoints) >= 1: ### check if can do == 1
         the_keypoints = keypoints[0]
         the_keypoint_scores = keypoint_scores[0]

         if DEBUG_CONSOLE: draw_box(img)

         for i, keypoints in enumerate(the_keypoints):
            keypoint_score = the_keypoint_scores[i]

            if keypoint_score >= THRESHOLD:
               threshold_color = GREEN
            else: threshold_color = RED

            if i == KP_LEFT_SHOULDER:
               left_shoulder = (map_keypoint_to_image_coords(keypoints.tolist(), img_size))
               draw_text(img, left_shoulder, threshold_color, img_size, KP_LEFT_SHOULDER)
            elif i == KP_RIGHT_SHOULDER:
               right_shoulder = (map_keypoint_to_image_coords(keypoints.tolist(), img_size))
               draw_text(img, right_shoulder, threshold_color, img_size, KP_RIGHT_SHOULDER)
            elif i == KP_LEFT_ELBOW:
               left_elbow = (map_keypoint_to_image_coords(keypoints.tolist(), img_size))
               draw_text(img, left_elbow, threshold_color, img_size, KP_LEFT_ELBOW)
            elif i == KP_RIGHT_ELBOW:
               right_elbow = (map_keypoint_to_image_coords(keypoints.tolist(), img_size))
               draw_text(img, right_elbow, threshold_color, img_size, KP_RIGHT_ELBOW)
            elif i == KP_LEFT_WRIST:
               left_wrist = (map_keypoint_to_image_coords(keypoints.tolist(), img_size))
               draw_text(img, left_wrist, threshold_color, img_size, KP_LEFT_WRIST)
            elif i == KP_RIGHT_WRIST:
               right_wrist = (map_keypoint_to_image_coords(keypoints.tolist(), img_size))
               draw_text(img, right_wrist, threshold_color, img_size, KP_RIGHT_WRIST)

      return {}



      # result = do_something(inputs["in1"], inputs["in2"])
      # outputs = {"out1": result}
      # return outputs