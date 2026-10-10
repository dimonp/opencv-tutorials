#include <cstddef>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Constants for YOLOv8 input and filtering thresholds
const cv::Size YOLO_INPUT_SIZE(640, 640);
const float CONFIDENCE_THRESHOLD = 0.25f;
const float SCORE_THRESHOLD = 0.45f;

// Structure to hold letterbox transformation mapping parameters
struct LetterboxParams {
    float scale;     // Resizing scale factor
    int top_pad;     // Top padding size in pixels
    int left_pad;    // Left padding size in pixels
};

// Modified Letterbox function that perfectly matches Ultralytics preprocessing logic
cv::Mat letterbox(const cv::Mat& source, const cv::Size& target_size, cv::Scalar padding_color, LetterboxParams& params) {
    int w = source.cols;
    int h = source.rows;

    // Calculate the scaling ratio (we scale based on the limiting dimension)
    float r = std::min((float)target_size.width / w, (float)target_size.height / h);

    // Compute the new unpadded inner image dimensions
    int new_unpad_w = std::round(w * r);
    int new_unpad_h = std::round(h * r);

    // Resize the original image using linear interpolation (standard for YOLOv8)
    cv::Mat resized;
    if (w != new_unpad_w || h != new_unpad_h) {
        cv::resize(source, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = source.clone();
    }

    // Calculate remaining padding spacing to hit the exact 640x640 target size
    int dw = target_size.width - new_unpad_w;
    int dh = target_size.height - new_unpad_h;

    // Divide padding by 2 to perfectly center the inner image inside the canvas
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    // Save parameters to map coordinates back accurately during post-processing
    params.scale = r;
    params.top_pad = top;
    params.left_pad = left;

    // Pad the borders of the image using the standardized YOLO gray color (114, 114, 114)
    cv::Mat result;
    cv::copyMakeBorder(
        resized,
        result,
        top, bottom, left, right,
        cv::BORDER_CONSTANT,
        padding_color);

    return result;
}

int main() {
    const std::string image_path = "data/cars.jpg";
    const std::string model_path = "data/yolo26n.onnx";

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load the image from path: " << image_path << std::endl;
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Apply the accurate centered Letterbox preprocessing
    LetterboxParams params;
    // (114, 114, 114) is the exact default background padding color used by Ultralytics
    cv::Mat letterboxed_img = letterbox(img, YOLO_INPUT_SIZE, cv::Scalar(114, 114, 114), params);

    // Create blob. swapRB=true switches OpenCV's BGR layout to YOLO's native RGB layout
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed_img, blob, 1.0 / 255.0, YOLO_INPUT_SIZE, cv::Scalar(), true, false);
    net.setInput(blob);

    // Run Inference
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Postprocessing
    cv::Mat output = outputs[0];
    output = output.reshape(1, output.size[1]);
    cv::transpose(output, output);

    int rows = output.rows;
    int total_columns = output.cols;
    int num_classes = total_columns - 4;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (size_t i = 0; i < rows; ++i) {
        float* data = (float*)output.ptr(i);

        float cx = data[0];
        float cy = data[1];
        float w  = data[2];
        float h  = data[3];

        float* classes_scores = data + 4;
        cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

        if (max_class_score > CONFIDENCE_THRESHOLD) {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id_point.x);

            // Accurate coordinate re-mapping considering center shifting and scaling
            // Subtract the padding offset, then divide by the scale factor to map to the original resolution
            int left   = int((cx - 0.5f * w - params.left_pad) / params.scale);
            int top    = int((cy - 0.5f * h - params.top_pad) / params.scale);
            int width  = int(w / params.scale);
            int height = int(h / params.scale);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, indices);

    // 7. Print Results
    std::cout << "\n--- Detection Results ---" << std::endl;
    for (size_t idx : indices) {
        int class_id = class_ids[idx];
        float confidence = confidences[idx];
        cv::Rect box = boxes[idx];

        std::cout << "Class ID: " << class_id
                  << " | Confidence: " << std::fixed << std::setprecision(2) << confidence
                  << " | Bounding Box: [x: " << box.x << ", y: " << box.y
                  << ", w: " << box.width << ", h: " << box.height << "]" << std::endl;
    }

    return 0;
}
