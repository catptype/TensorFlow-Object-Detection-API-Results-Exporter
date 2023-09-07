import sys
sys.dont_write_bytecode = True

import os
import shutil
import time
import numpy as np
import tensorflow as tf

from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from .ObjectDetectionAPI import ObjectDetectionAPI
from .ROIExporter import ROIExporter
from .XMLExporter import XMLExporter

## Avoid out of memory by setting GPU memory consumption growth
gpu_list = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)

class ResultExporter():
    """
    A class for exporting object detection results including ROI images and XML annotations.

    Attributes:
        model: The TensorFlow object detection model.
        category_idx: A dictionary mapping class IDs to class names.
        image_paths (list): List of image file paths to process.
    """
    def __init__(self):
        """
        Initializes the ResultExporter instance.

        - Initializes the model and category index.
        - Creates export directories.
        """
        self.model, self.category_idx = self.__initializing_model()
        self.__prepare_path(os.path.join("Image","export_result"))
        self.__prepare_path(os.path.join("Image","object_found"))
        self.__prepare_path(os.path.join("Image","object_not_found"))

    def __prepare_path(self, path):
        """
        Creates a directory if it does not exist.

        Args:
            path (str): The directory path to create.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def __load_image_paths(self):
        """
        Loads image paths from the "Image" directory.

        Returns:
            list: List of image file paths.
        """
        image_paths = [os.path.join("Image", file) for file in os.listdir("Image") if os.path.isfile(os.path.join("Image", file))]
        image_paths = [path for path in image_paths if any(ext in path for ext in ('.jpg', '.png'))]
        return image_paths

    def __model_selection(self, model_dir):
        """
        Selects a model from the "Model" directory.

        Args:
            model_dir (str): The directory containing the models.

        Returns:
            str: The selected model directory name.
        """
        print("Scanning for models... ", end="")
        legit_model_list = [model for model in os.listdir(model_dir)
                            if os.path.exists(os.path.join(model_dir, model, 'pipeline.config')) and
                            os.path.exists(os.path.join(model_dir, model, 'checkpoint', 'ckpt-0.index')) and
                            os.path.exists(os.path.join(model_dir, model, 'label_map.pbtxt'))]
        num_models = len(legit_model_list)
        print(f"Found {num_models}")
        
        if len(legit_model_list) == 0:
            return None
        
        if len(legit_model_list) == 1:
            return legit_model_list[0]
        
        while True:
            for idx, model in enumerate(legit_model_list):
                print(f"{idx}: {model}")
            try:
                model_num = int(input("Select a model: "))         
                if 0 <= model_num < num_models:
                    return legit_model_list[model_num]
                else:
                    print("Invalid input. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def __initializing_model(self):
        """
        Initializes the detection model and category index.

        Returns:
            tuple: A tuple containing the detection model and category index.
        """
        # Initialize the detection model and category index
        model = self.__model_selection("Model")

        # Model paths
        cfg_path     = os.path.join("Model", model, 'pipeline.config')
        ckpt_path    = os.path.join("Model", model, 'checkpoint', 'ckpt-0')
        label_path   = os.path.join("Model", model, 'label_map.pbtxt')

        # Load model
        print('Loading model... ', end='')
        start_time      = time.time()

        # Load pipeline config and build a detection model
        configs         = config_util.get_configs_from_pipeline_file(cfg_path)
        model_config    = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt            = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(ckpt_path).expect_partial()

        # Create a category index
        category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Took {elapsed_time:.2f} seconds")
        return detection_model, category_index
    
    def export(self, threshold=0.5, roi=True, xml=True):
        """
        Exports object detection results including ROI images and XML annotations.

        Args:
            threshold (float): The confidence threshold for object detection.
            roi (bool): Whether to export ROI images.
            xml (bool): Whether to export XML annotations.
        """
        image_paths = self.__load_image_paths()
        progress_bar = tf.keras.utils.Progbar(len(image_paths), 
                                              stateful_metrics=["progress"],
                                              width=20, 
                                              interval=0.1, 
                                              unit_name='image')
        for image_path in image_paths:
            # Prepare filename and its extension
            filename = os.path.basename(image_path)
            _, ext = os.path.splitext(filename)

            # Calling ObjectDetectionAPI class
            obj_detect = ObjectDetectionAPI(image_path, self.model)
            detections = obj_detect.detections

            if max(detections['detection_scores']) < threshold:
                source = os.path.join("Image", filename)
                destination = os.path.join("Image", "object_not_found", filename)
                shutil.move(source, destination)
                progress_bar.add(1)
                continue

            # Export image with ROI
            if roi:
                image = obj_detect.image
                image_np = np.array(image.convert('RGB'))
                filename_roi = os.path.join("Image", "export_result", filename.replace(f"{ext}", f"_ROI{ext}"))
                exporter = ROIExporter(input_image=image_np, 
                                        detections=detections, 
                                        category_idx=self.category_idx, 
                                        threshold=threshold)
                exporter.write_roi(filename_roi, 
                                    image_format=ext, 
                                    quality=image.info.get('quality', 95))
            
            # Export XML
            if xml:
                filename_xml = os.path.join("Image", "export_result", filename.replace(ext, ".xml"))
                exporter = XMLExporter(image_filename=filename, 
                                        detections=detections, 
                                        category_idx=self.category_idx, 
                                        threshold=threshold)
                exporter.write_xml(filename_xml)

            source = os.path.join("Image", filename)
            destination = os.path.join("Image", "object_found", filename)
            shutil.move(source, destination)
            
            progress_bar.add(1)
