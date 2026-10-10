#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Constants for configuration
const float CONFIDENCE_THRESHOLD = 0.25f;
const float MASK_THRESHOLD = 0.5f; // Threshold for binarizing the final mask
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

int main() {
    // 1. Load the segmentation network using OpenCV DNN module
    // This model must be exported as a segmentation variant (e.g., yolo26n-seg.onnx)
    cv::dnn::Net net = cv::dnn::readNetFromONNX("data/yolo26n-seg.onnx");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 2. Read the input image
    std::string imagePath = "data/cars.jpg";
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return -1;
    }

    int imgWidth = image.cols;
    int imgHeight = image.rows;

    // 3. Preprocess the image into a 4D blob
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    // 4. Run the forward pass (Segmentation returns multiple output layers)
    std::vector<cv::Mat> outputs;
    std::vector<std::string> outputLayerNames = net.getUnconnectedOutLayersNames();
    net.forward(outputs, outputLayerNames);

    // Identify output tensors based on their expected dimensions
    // outputs[0] usually holds detections/coefficients, outputs[1] holds mask prototypes
    cv::Mat detOutput = outputs[0];
    cv::Mat protoOutput = outputs[1];

    // RF-DETR NMS-free segmentation output structure: [1, 300, 6 + 32]
    // 6 elements for bbox, score, class; 32 elements for mask coefficients
    int numProposals = detOutput.size[1];
    int numCoefficients = detOutput.size[2] - 6; // Usually 32

    // Reshape prototype tensor to [32, 160x160] for matrix multiplication
    int protoChannels = protoOutput.size[1]; // 32
    int protoHeight = protoOutput.size[2];   // 160
    int protoWidth = protoOutput.size[3];    // 160
    cv::Mat protoMasks = protoOutput.reshape(1, protoChannels); // Size: [32, 25600]

    // Create an overlay matrix for colorful transparent contours/masks
    cv::Mat overlay = image.clone();

    // 5. Post-processing the predictions
    float* detData = (float*)detOutput.data;
    int rowSize = detOutput.size[2]; // Total elements per row (6 + 32 = 38)

    for (int i = 0; i < numProposals; ++i) {
        int index = i * rowSize;
        float x1 = detData[index + 0];
        float y1 = detData[index + 1];
        float x2 = detData[index + 2];
        float y2 = detData[index + 3];
        float score = detData[index + 4];
        int classId = static_cast<int>(detData[index + 5]);

        // Filter out predictions with low confidence scores
        if (score >= CONFIDENCE_THRESHOLD) {
            // Scale bounding box coordinates back to the original image dimensions
            int left   = static_cast<int>((x1 / INPUT_WIDTH) * imgWidth);
            int top    = static_cast<int>((y1 / INPUT_HEIGHT) * imgHeight);
            int right  = static_cast<int>((x2 / INPUT_WIDTH) * imgWidth);
            int bottom = static_cast<int>((y2 / INPUT_HEIGHT) * imgHeight);

            // Constrain boxes to image boundaries
            left   = std::max(0, std::min(left, imgWidth - 1));
            top    = std::max(0, std::min(top, imgHeight - 1));
            right  = std::max(0, std::min(right, imgWidth - 1));
            bottom = std::max(0, std::min(bottom, imgHeight - 1));

            int width  = right - left;
            int height = bottom - top;
            if (width <= 0 || height <= 0) continue;

            // --- EXTRACTING MASK CONTOUR ---
            // Extract the 32 mask coefficients for the current object
            cv::Mat maskCoeffs(1, numCoefficients, CV_32F, &detData[index + 6]);

            // Multiply coefficients by prototypes to generate the raw mask: [1, 32] * [32, 25600] = [1, 25600]
            cv::Mat rawMask = maskCoeffs * protoMasks;
            rawMask = rawMask.reshape(1, protoHeight); // Reshape back to [160, 160]

            // Apply Sigmoid activation layer to map raw values to probabilities [0, 1]
            cv::Mat sigmoidMask;
            cv::exp(-rawMask, sigmoidMask);
            sigmoidMask = 1.0 / (1.0 + sigmoidMask);

            // Crop the mask area that corresponds to the object bounding box (normalized coordinates)
            float x1_norm = x1 / INPUT_WIDTH;
            float y1_norm = y1 / INPUT_HEIGHT;
            float x2_norm = x2 / INPUT_WIDTH;
            float y2_norm = y2 / INPUT_HEIGHT;

            cv::Rect roi(
                static_cast<int>(x1_norm * protoWidth),
                static_cast<int>(y1_norm * protoHeight),
                static_cast<int>((x2_norm - x1_norm) * protoWidth),
                static_cast<int>((y2_norm - y1_norm) * protoHeight)
            );

            // Boundary checks for the ROI inside the prototype mask dimensions
            roi.x = std::max(0, std::min(roi.x, protoWidth - 1));
            roi.y = std::max(0, std::min(roi.y, protoHeight - 1));
            roi.width = std::max(1, std::min(roi.width, protoWidth - roi.x));
            roi.height = std::max(1, std::min(roi.height, protoHeight - roi.y));

            cv::Mat croppedMask = sigmoidMask(roi);

            // Resize the cropped mask to match the exact absolute size of the bounding box
            cv::Mat resizedMask;
            cv::resize(croppedMask, resizedMask, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

            // Binarize the mask into a binary threshold image (0 or 255)
            cv::Mat binaryMask;
            cv::threshold(resizedMask, binaryMask, MASK_THRESHOLD, 255, cv::THRESH_BINARY);
            binaryMask.convertTo(binaryMask, CV_8U);

            // Find external contours within the binary mask
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Random color generation for each distinct object instance
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

            // Shift contour points to match their absolute location inside the global image frame
            for (size_t c = 0; c < contours.size(); ++c) {
                for (size_t p = 0; p < contours[c].size(); ++p) {
                    contours[c][p].x += left;
                    contours[c][p].y += top;
                }
                // Draw the solid filled mask area onto the overlay matrix
                cv::drawContours(overlay, contours, static_cast<int>(c), color, cv::FILLED);
                // Draw the explicit outer contour edge line on the primary image
                cv::drawContours(image, contours, static_cast<int>(c), color, 2);
            }

            // Draw bounding box and text label
            cv::Rect box(left, top, width, height);
            cv::rectangle(image, box, color, 2);

            std::string label = "ID " + std::to_string(classId) + " (" + cv::format("%.2f", score) + ")";
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int labelTop = std::max(top, labelSize.height);

            cv::rectangle(image, cv::Point(left, labelTop - labelSize.height),
                          cv::Point(left + labelSize.width, labelTop + baseLine), color, cv::FILLED);
            cv::putText(image, label, cv::Point(left, labelTop), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

    // Blend the solid colored overlay into the main image with 40% transparency
    float alpha = 0.4f;
    cv::addWeighted(overlay, alpha, image, 1.0f - alpha, 0.0, image);

    // 6. Display the result
    std::string windowName = "RF-DETR / YOLO26-Seg Instance Segmentation";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, image);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
