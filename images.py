import mediapipe as mp
from pathlib import Path

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def process_images(images):
  """
  
  """
  assert images
  processed_images = []
  
  # Build model path relative to this file to avoid Windows '\' escape issues
  model_path = r'C:\Users\maite\OneDrive\Desktop\LSA_deteccion_estatica\hand_landmarker.task'
  
  BaseOptions = mp.tasks.BaseOptions
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # Create a hand landmarker instance with the image mode:
  options = HandLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=VisionRunningMode.IMAGE,
      num_hands = 1,
      min_hand_detection_confidence=0.5
      )
  
  with HandLandmarker.create_from_options(options) as landmarker:
    
    for image_path in images: 
      # Load the input image from an image file.
      mp_image = mp.Image.create_from_file(image_path)

      # SOLUCIÓN 2: Borramos la línea duplicada que causaba el error
      # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image) 
      
      # Perform hand landmarks detection on the provided single image.
      hand_landmarker_result = landmarker.detect(mp_image)
    
      annotated_image = draw_landmarks_on_image(mp_image, hand_landmarker_result)
    
      processed_images.append(annotated_image)
      
  return processed_images