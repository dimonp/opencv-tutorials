# OpenCV Learning Project

A collection of examples and experiments developed while learning **OpenCV** and exploring computer vision techniques.

## 📌 About

This project serves as a personal learning repository for exploring OpenCV through practical examples. Each example demonstrates a particular feature, algorithm, or image processing technique.

The repository evolves as new concepts are explored and implemented, providing a reference for future experiments and projects.

## 🎯 Goals

* Explore the core features of OpenCV.
* Learn image and video processing techniques.
* Understand fundamental computer vision algorithms.
* Experiment with neural network models and their integration into computer vision workflows.
* Build a collection of practical examples for future reference.

## 🛠️ Technologies

* **C++** — primary programming language.
* **OpenCV** — computer vision, image processing, and deep learning inference.
* **CMake** — build system configuration.
* **Deep Learning** — experimentation with neural networks and pre-trained models.
* **YOLO (You Only Look Once)** — real-time object detection using pre-trained neural network models.
* **ONNX** — a model format for interoperability between deep learning frameworks and inference engines, where supported by the selected models and OpenCV configuration.

## 📂 Project Structure

```text
opencv-learning/
├── src/
│   └── examples/    # OpenCV examples and experiments
├── data/            # Input images and neural network models
├── CMakeLists.txt   # CMake build configuration
└── README.md        # Project documentation
```

### Directories and files

* **`src/examples/`** — source code for individual examples demonstrating OpenCV functionality, image processing techniques, neural network inference, and object detection.
* **`data/`** — input images, pre-trained neural network models, YOLO weights, and other resources required by the examples.
* **`CMakeLists.txt`** — CMake configuration used to configure and build the project.

## ⚙️ Requirements

Before building the project, make sure the following tools are installed:

* A C++ compiler supporting the C++ standard required by the project.
* [CMake](https://cmake.org/) — for configuring and building the project.
* [OpenCV](https://opencv.org/) — installed and available to CMake.

The required C++ standard and OpenCV version depend on the current project configuration.

## 🚀 Building the Project

From the project root directory, create a build directory and configure the project:

```bash
cmake -S . -B build
```

Build the configured targets:

```bash
cmake --build build
```

These commands assume that the dependencies are installed and that `CMakeLists.txt` correctly defines the project targets.

## ▶️ Running Examples

The available examples depend on the targets defined in `CMakeLists.txt`.

After building, run an example using the executable generated for that target. The exact executable name and location depend on the build configuration and operating system.

Some examples may require input images or neural network models from the `data/` directory. Make sure the required resources are available at the paths expected by the source code.

## 📚 Topics

The repository explores various computer vision concepts and techniques, including:

* Image loading, display, and manipulation.
* Image transformations and geometric operations.
* Color spaces and color channel processing.
* Image filtering, thresholding, and edge detection.
* Contour detection and shape analysis.
* Video processing and camera input.
* Neural network inference using OpenCV's DNN module.
* Deep learning model loading and execution.
* YOLO-based object detection.
* Real-time object detection in images and video streams.
* Working with pre-trained models and model input/output processing.

## 🧠 Deep Learning and YOLO

This project includes experiments with deep learning techniques and pre-trained neural network models for computer vision tasks.

A particular focus is on **YOLO (You Only Look Once)**, a family of neural network architectures designed for efficient object detection. The examples may explore loading models, preparing input data, running inference, processing detection results, and visualizing detected objects.

OpenCV's **DNN module** can be used to load and execute compatible neural network models, including supported YOLO variants.

Depending on the model and its format, additional preprocessing or postprocessing may be required. Model compatibility depends on the YOLO version, exported format, and OpenCV version.

Neural network models and related resources are stored in the `data/` directory.

## 📝 Notes

* This is a learning-oriented project focused on practical experimentation with OpenCV and computer vision.
* Examples may cover both traditional image processing algorithms and deep learning-based approaches.
* Pre-trained neural network models may require specific input dimensions, preprocessing steps, and output decoding logic.
* YOLO model compatibility depends on the architecture, model version, export format, and available OpenCV functionality.
* Some examples may require additional model files or configuration files in the `data/` directory.

## 📈 Future Development

The repository will continue to grow with new examples, experiments, and implementations as I explore additional OpenCV features and computer vision techniques.

---

**OpenCV Learning Project** — learning computer vision through practical C++ examples.
