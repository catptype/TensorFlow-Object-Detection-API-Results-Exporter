import sys
sys.dont_write_bytecode = True

import xml.etree.ElementTree as ET

class XMLExporter():
    """
    A class for exporting object detection results in XML format.

    Attributes:
        image_filename (str): The filename of the image being annotated.
        xml_tree (xml.etree.ElementTree.ElementTree): The XML tree containing the annotations.
    """
    def __init__(self, image_filename, detections, category_idx, threshold):
        """
        Initializes the XMLExporter instance.

        Args:
            image_filename (str): The filename of the input image.
            detections (dict): A dictionary containing object detection results.
            category_idx (dict): A dictionary mapping class IDs to class names.
            threshold (float): The confidence threshold for object detection.
        """
        self.image_filename = image_filename
        self.xml_tree = self.__create_xml_tree(detections, category_idx, threshold)

    def __create_xml_tree(self, detections, category_idx, threshold):
        """
        Creates an XML ElementTree representing the annotations.

        Args:
            detections (dict): A dictionary containing object detection results.
            category_idx (dict): A dictionary mapping class IDs to class names.
            threshold (float): The confidence threshold for object detection.

        Returns:
            xml.etree.ElementTree.ElementTree: The XML ElementTree representing the annotations.
        """
        boundary_boxes = []
        for box, class_idx, score in zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
            if score < threshold:
                continue

            class_name = category_idx[class_idx + 1]['name']
            
            # Boundary box coordiates
            ymin, xmin, ymax, xmax = box

            boundary_boxes.append({
                "classname": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "score": score,
            })

        xml_tree = self.__processing_xml_annotation(boundary_boxes)
        return xml_tree

    def __processing_xml_annotation(self, objects):
        """
        Processes and constructs the XML annotations.

        Args:
            objects (list): A list of dictionaries representing detected objects.

        Returns:
            xml.etree.ElementTree.ElementTree: The XML ElementTree representing the annotations.
        """
        root = ET.Element("annotation")
        
        # Create basic image information
        filename_elem      = ET.SubElement(root, "filename")
        filename_elem.text = self.image_filename

        # Object annotations
        for obj in objects:
            # Tree preparation
            obj_elem     = ET.SubElement(root, "object")
            classname_elem = ET.SubElement(obj_elem, "classname")
            bndbox_elem    = ET.SubElement(obj_elem, "bndbox")
            xmin_elem        = ET.SubElement(bndbox_elem, "xmin")
            ymin_elem        = ET.SubElement(bndbox_elem, "ymin")
            xmax_elem        = ET.SubElement(bndbox_elem, "xmax")
            ymax_elem        = ET.SubElement(bndbox_elem, "ymax")
            score_elem     = ET.SubElement(obj_elem, "score")
            
            # Data text
            ## Class name
            classname_elem.text = obj["classname"]

            ## Bounding box coordinates
            xmin_elem.text = str(obj["xmin"])
            ymin_elem.text = str(obj["ymin"])
            xmax_elem.text = str(obj["xmax"])
            ymax_elem.text = str(obj["ymax"])

            ## Detection score
            score_elem.text = str(obj["score"])

        # Create and return the ElementTree
        tree = ET.ElementTree(root)
        return tree
    
    def write_xml(self, destination):
        """
        Writes the XML annotations to a file.

        Args:
            destination (str): The path to save the XML file.
        """
        self.xml_tree.write(destination)