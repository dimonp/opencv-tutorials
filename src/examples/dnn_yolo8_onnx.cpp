#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

int main() {
    // Specify paths to the YOLOv8 ONNX model and the test image
    const std::string modelPath = "data/yolo8n.onnx";
    const std::string imagePath = "data/cars.jpg";
    cv::dnn::Net net;

    // Load the pretrained ONNX network with exception handling
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty()) {
            std::cerr << "Error: Model file opened, but the network is empty!" << std::endl;
            return -1;
        }
        std::cout << "ONNX model loaded successfully!" << std::endl;
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

    // Preprocess the image to strictly match YOLOv8 640x640 layout
    cv::Mat blob;
    cv::Size targetSize(640, 640);

    cv::dnn::blobFromImage(
        image,
        blob,
        1.0 / 255.0,    // Scale factor to [0.0, 1.0]
        targetSize,     // Exact input size required by YOLOv8
        cv::Scalar(0, 0, 0), // No mean subtraction
        true,           // Swap Red and Blue channels (BGR -> RGB)
        false           // Do not crop, resize directly
    );

    // Pass the processed blob to the network
    net.setInput(blob);

    // Vector containers for holding detection results before Non-Maximum Suppression (NMS)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    const int CAR_CLASS_ID = 2; // Class ID for 'car' in COCO dataset
    const float CONFIDENCE_THRESHOLD = 0.35f; // Minimum score to consider a detection
    const float NMS_THRESHOLD = 0.45f;        // Overlap threshold for filtering duplicates

    try {
        // Run the forward pass
        cv::Mat output = net.forward();

        // YOLOv8 output structure is usually [1 x 84 x 8400]
        int dimensions = output.size[1];
        int rows = output.size[2];

        // Reshape and transpose to make it row-by-row parseable [8400 x 84]
        cv::Mat data = output.reshape(1, dimensions);
        cv::Mat transposedData;
        cv::transpose(data, transposedData);

        // Parse predictions row by row
        for (size_t i = 0; i < rows; ++i) {
            float* rowData = transposedData.ptr<float>(i);
            float carScore = rowData[4 + CAR_CLASS_ID];

            // Filter out everything below our confidence threshold
            if (carScore >= CONFIDENCE_THRESHOLD) {
                // Extract bounding box coordinates (normalized to 640x640 space)
                float cx = rowData[0];
                float cy = rowData[1];
                float w  = rowData[2];
                float h  = rowData[3];

                // Scale coordinates back to match the original input image size
                int left   = static_cast<int>((cx - w / 2.0) * (imgWidth / 640.0));
                int top    = static_cast<int>((cy - h / 2.0) * (imgHeight / 640.0));
                int width  = static_cast<int>(w * (imgWidth / 640.0));
                int height = static_cast<int>(h * (imgHeight / 640.0));

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(carScore);
            }
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception during inference: " << e.msg << std::endl;
        return -1;
    }

    // Apply Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    std::cout << "Successfully found " << indices.size() << " cars in the image." << std::endl;

    // Draw the final boxes onto the original image
    for (size_t idx : indices) {
        cv::Rect box = boxes[idx];
        float confidence = confidences[idx];

        // Draw bounding box rectangle
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        // Prepare and draw label text string
        std::string label = "Car: " + std::to_string(static_cast<int>(confidence * 100)) + "%";
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // Ensure background text box doesn't go out of frame bounds
        int top = std::max(box.y, labelSize.height);
        cv::rectangle(image, cv::Point(box.x, top - labelSize.height),
                      cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);

        // Write the text description
        cv::putText(image, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Display the result in a UI window instead of writing to a file
    std::string windowName = "YOLOv8 Car Detection Results";

    // Create a resizable window to handle large high-resolution images properly
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, image);

    std::cout << "Press any key in the image window to close the application..." << std::endl;

    // Wait indefinitely until a keyboard key is pressed
    cv::waitKey(0);

    // Cleanup highgui windows resource properly
    cv::destroyAllWindows();

    return 0;
}
