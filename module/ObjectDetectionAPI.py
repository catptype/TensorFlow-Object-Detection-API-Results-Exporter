import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
from PIL import Image

## Avoid out of memory by setting GPU memory consumption growth
gpu_list = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)

class ObjectDetectionAPI():
    """
    A class for performing object detection on an image using a TensorFlow Object Detection API.
    
    Attributes:
        image_path (str): The path to the input image.
        model: A TensorFlow object detection model.
        detections (dict): A dictionary containing detection results.
    """
    def __init__(self, image_path, model):
        """
        Initializes the ObjectDetectionAPI instance.

        Args:
            image_path (str): The path to the input image.
            model: A TensorFlow Object Detection API model.
        """
        self.image = Image.open(image_path)
        self.model = model
        self.detections = self.__predict_object()

    def __predict_object(self):
        """
        Private method to perform object detection on the input image using the model.

        Returns:
            dict: A dictionary containing detection results.
        """
        # Convert the PIL image to a NumPy array
        image_np = np.array(self.image.convert('RGB'))

        # Convert the NumPy array to a TensorFlow tensor
        image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                
        # Run inference
        image, shapes = self.model.preprocess(image_tensor)
        prediction_dict = self.model.predict(image, shapes)
        detections = self.model.postprocess(prediction_dict, shapes)

        # Post-process the detection results (Convert TensorFlow tensors to NumPy arrays)
        num_detections = int(detections.pop('num_detections')) # Only num_detections has different shape
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections