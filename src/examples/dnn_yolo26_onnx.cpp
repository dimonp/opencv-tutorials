#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // Specify paths to the YOLO26 ONNX model and the test image
    const std::string modelPath = "data/yolo26n.onnx";
    const std::string imagePath = "data/cars.jpg";
    cv::dnn::Net net;

    // Load the pretrained YOLO26 ONNX network with exception handling
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty()) {
            std::cerr << "Error: Model file opened, but the network is empty!" << std::endl;
            return -1;
        }
        std::cout << "YOLO26 ONNX model loaded successfully!" << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception while loading the model: " << e.msg << std::endl;
        return -1;
    }

    // Load the input image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not read the image!" << std::endl;
        return -1;
    }

    // Save original dimensions to rescale bounding boxes later
    int imgWidth = image.cols;
    int imgHeight = image.rows;

    // Preprocess the image to match YOLO26 640x640 layout
    cv::Mat blob;
    cv::Size targetSize(640, 640);

    cv::dnn::blobFromImage(
        image,
        blob,
        1.0 / 255.0,         // Scale factor to [0.0, 1.0]
        targetSize,          // Exact input size required by YOLO26
        cv::Scalar(0, 0, 0), // No mean subtraction
        true,                // Swap Red and Blue channels (BGR -> RGB)
        false                // Do not crop, resize directly
    );

    // Pass the processed blob to the network
    net.setInput(blob);

    const int CAR_CLASS_ID = 2;               // Class ID for 'car' in COCO dataset
    const float CONFIDENCE_THRESHOLD = 0.35f; // Minimum score to filter noise detections
    int carCount = 0;

    try {
        // Run the forward pass
        cv::Mat output = net.forward();

        // YOLO26 output structure is End-to-End NMS-Free: [1 x 300 x 6]
        // Elements per prediction: [x_min, y_min, x_max, y_max, confidence, class_id]
        int numPredictions = output.size[1];
        int elementSize = output.size[2];

        // Reshape output to [300 x 6] matrix for easier iteration
        cv::Mat data = output.reshape(1, numPredictions);

        // Parse native predictions directly (No NMS needed!)
        for (size_t i = 0; i < numPredictions; ++i) {
            float* rowData = data.ptr<float>(i);

            float confidence = rowData[4];
            int classId = static_cast<int>(rowData[5]);

            // Filter by confidence score and check if the class is a car
            if (confidence >= CONFIDENCE_THRESHOLD && classId == CAR_CLASS_ID) {
                carCount++;

                // Extract direct bounding box corners (coordinates are typically in 640x640 space)
                float x_min = rowData[0];
                float y_min = rowData[1];
                float x_max = rowData[2];
                float y_max = rowData[3];

                // Scale coordinates back to match the original input image size
                int left   = static_cast<int>(x_min * (imgWidth / 640.0));
                int top    = static_cast<int>(y_min * (imgHeight / 640.0));
                int right  = static_cast<int>(x_max * (imgWidth / 640.0));
                int bottom = static_cast<int>(y_max * (imgHeight / 640.0));

                int width  = right - left;
                int height = bottom - top;
                cv::Rect box(left, top, width, height);

                // 8. Draw bounding box rectangle
                cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

                // Prepare and draw label text string
                std::string label = "Car: " + std::to_string(static_cast<int>(confidence * 100)) + "%";
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int textTop = std::max(box.y, labelSize.height);
                cv::rectangle(image, cv::Point(box.x, textTop - labelSize.height),
                              cv::Point(box.x + labelSize.width, textTop + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);

                cv::putText(image, label, cv::Point(box.x, textTop), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception during inference: " << e.msg << std::endl;
        return -1;
    }

    std::cout << "Successfully found " << carCount << " cars in the image." << std::endl;

    // Display the result in a UI window
    std::string windowName = "YOLO26 Car Detection Results (NMS-Free)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, image);

    std::cout << "Press any key in the window to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
