"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
THRESHOLD = -1

# https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8

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
   return int(x), int(y)

def draw_text(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.4,
      color=color_code,
      thickness=2,
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

      ## derived inputs from raw inputs
      img_size = (img.shape[1], img.shape[0])

      # detecting keypoints on screen
      if len(keypoints) >= 1: ### check if can do == 1
         the_keypoints = keypoints[0]
         the_keypoint_scores = keypoint_scores[0]
         left_shoulder = None
         right_shoulder = None
         left_elbow = None
         right_elbow = None

         for i, keypoints in enumerate(the_keypoints):
            keypoint_score = the_keypoint_scores[i]

            if keypoint_score >= THRESHOLD:
               x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
               x_y_str = f"({x}, {y}"

               if i == KP_LEFT_SHOULDER:
                  left_shoulder = keypoints
                  the_color = YELLOW
               elif i == KP_RIGHT_SHOULDER:
                  right_shoulder = keypoints
                  the_color = YELLOW
               elif i == KP_LEFT_ELBOW:
                  left_elbow = keypoints
                  the_color = YELLOW
               elif i == KP_RIGHT_ELBOW:
                  right_elbow = keypoints
                  the_color = YELLOW
               else:
                  the_color = WHITE
               
               draw_text(img, x, y, x_y_str, the_color)

      return {}



      # result = do_something(inputs["in1"], inputs["in2"])
      # outputs = {"out1": result}
      # return outputs
