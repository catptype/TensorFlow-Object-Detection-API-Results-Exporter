import sys
sys.dont_write_bytecode = True

from PIL import Image
from object_detection.utils import visualization_utils as vis_utils

class ROIExporter():
    """
    A class for exporting images with region-of-interest (ROI) visualizations.

    Attribute:
        image_roi (PIL.Image.Image): The image with ROI visualizations.
    """
    def __init__(self, input_image, detections, category_idx, threshold):
        """
        Initializes the ROIExporter instance.

        Args:
            input_image (numpy.ndarray): The input image as a NumPy array.
            detections (dict): A dictionary containing object detection results.
            category_idx (dict): A dictionary mapping class IDs to class names.
            threshold (float): The confidence threshold for object detection.
        """
        self.image_roi = self.__generate_image_ROI(input_image, detections, category_idx, threshold)

    def __generate_image_ROI(self, input_image, detections, category_idx, threshold):        
        """
        Generates an image with region-of-interest (ROI) visualizations.

        Args:
            input_image (numpy.ndarray): The input image as a NumPy array.
            detections (dict): A dictionary containing object detection results.
            category_idx (dict): A dictionary mapping class IDs to class names.
            threshold (float): The confidence threshold for object detection.

        Returns:
            PIL.Image.Image: The image with ROI visualizations.
        """
        # Visualize boxes and labels on the input image
        vis_utils.visualize_boxes_and_labels_on_image_array(
            input_image,
            detections['detection_boxes'], # Boxes
            detections['detection_classes'] + 1, # Label index with offset +1
            detections['detection_scores'], # Classification score
            category_idx,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100, # or detections['num_detections']
            min_score_thresh=threshold,
            line_thickness=2,
            agnostic_mode=False)

        image_roi = Image.fromarray(input_image)
        return image_roi
    
    def write_roi(self, destination, image_format, quality=95):
        """
        Writes the image with ROI visualizations to a file.

        Args:
            destination (str): The path to save the image.
            image_format (str): The image format (e.g., '.jpg', '.png').
            quality (int): The image quality (only applicable for JPEG format).
        """
        if image_format == ".jpg":
            self.image_roi.save(destination, 
                                format='JPEG', 
                                subsampling=0, 
                                quality=quality,
                                optimize=True)
        
        elif image_format == ".png":
            self.image_roi.save(destination, 
                                format='PNG', 
                                subsampling=0, 
                                quality=quality,
                                optimize=True)
        
        else:
            print("Format is not available")