# TensorFlow Object Detection API Results Exporter

## Introduction

<b>TensorFlow Object Detection API Results Exporter</b> is a Python project designed to address a specific need in the field of object detection. It extends the functionality of the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) by providing the capability to export images with Region of Interest (ROI) annotations and boundary box detections as XML data.

It utilizes portions of [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), an open-source project developed by the TensorFlow team at Google, which is licensed under the Apache License Version 2.0. I have not modified code and have only imported modules directly for the purpose of object detection.

For more information about [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and its license, please visit  [here](https://github.com/tensorflow/models/blob/master/LICENSE).

## Table of Contents
- [Imported modules](#imported-modules)
- [Usage](#usage)
- [Output](#output)
- [License](#license)

## Imported modules

- [object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [official](https://github.com/tensorflow/models/tree/master/official)

## Usage

Follow these steps to use the TensorFlow Object Detection API Results Exporter:

1. Put your `.jpg` and `.png` files into the `Image` directory.
2. Place your Object Detection API model in the `Model` directory.
3. Add a `label_map.pbtxt` file to the corresponding Object Detection API model directory.
4. Run the `TensorFlow Object Detection API Results Exporter.ipynb` notebook.

After completing the third step, your directory structure should resemble the following:
```
.
├── Image
│   ├── *.jpg
│   ├── *.png
│   └── ...
├── Model
│   ├── Object Detection API model
│   │   ├── checkpoint
│   │   │   ├── checkpoint
│   │   │   ├── ckpt-0.data-00000-of-00001
│   │   │   └── ckpt-0.index
│   │   ├── saved_model
│   │   │   ├── fingerprint.pb
│   │   │   └── saved_model.pb
│   │   ├── label_map.pbtxt
│   │   └── pipeline.config
...
```

## Output

After running the notebook, the `Image` directory will contain an `export_result` subdirectory with exported images and XML annotations.
```
.
├── Images
│   ├── export_result
│   │     ├── *.jpg
│   │     ├── *.png
│   │     ├── *.xml
│   │     └── ...
...
```

## License

This project is also licensed under the Apache License Version 2.0. See the [LICENSE](LICENSE) file for details.