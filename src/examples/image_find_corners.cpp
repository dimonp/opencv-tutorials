#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Enumeration for convenient switching between algorithms
enum Method {
    METHOD_FAST,
    METHOD_HARRIS,
    METHOD_SHI_TOMASI,
    METHOD_MIN_EIGEN_VAL,
    METHOD_PRE_CORNER_DETECT, // Added Pre Corner Detect method
    METHOD_COUNT              // Total number of methods
};

// Returns the method name as a string for on-screen display
std::string getMethodName(Method method) {
    switch (method) {
        case METHOD_FAST:              return "FAST (Corner Detector)";
        case METHOD_HARRIS:            return "Harris Corner Detector";
        case METHOD_SHI_TOMASI:        return "Shi-Tomasi (Good Features)";
        case METHOD_MIN_EIGEN_VAL:     return "Min Eigen Value Detector";
        case METHOD_PRE_CORNER_DETECT: return "Pre Corner Detect Map";
        default:                       return "Unknown";
    }
}

// Custom function to detect Harris corners and convert them to cv::KeyPoint
void detectHarrisCorners(const cv::Mat& grayFrame, std::vector<cv::KeyPoint>& keypoints) {
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;

    cv::Mat dst = cv::Mat::zeros(grayFrame.size(), CV_32FC1);
    cv::cornerHarris(grayFrame, dst, blockSize, apertureSize, k);

    cv::Mat dst_norm;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int)dst_norm.at<float>(j, i) > thresh) {
                keypoints.push_back(cv::KeyPoint(static_cast<float>(i), static_cast<float>(j), 5.0f));
            }
        }
    }
}

// Custom function to detect Shi-Tomasi corners using cv::goodFeaturesToTrack
void detectShiTomasiCorners(const cv::Mat& grayFrame, std::vector<cv::KeyPoint>& keypoints) {
    std::vector<cv::Point2f> corners;
    int maxCorners = 500;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;

    cv::goodFeaturesToTrack(grayFrame, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector);

    for (const auto& pt : corners) {
        keypoints.push_back(cv::KeyPoint(pt, 6.0f));
    }
}

// Custom function to extract corners using direct cv::cornerMinEigenVal block computation
void detectMinEigenValCorners(const cv::Mat& grayFrame, std::vector<cv::KeyPoint>& keypoints) {
    int blockSize = 3;
    int apertureSize = 3;
    int thresh = 150;

    cv::Mat dst = cv::Mat::zeros(grayFrame.size(), CV_32FC1);
    cv::cornerMinEigenVal(grayFrame, dst, blockSize, apertureSize);

    cv::Mat dst_norm;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int)dst_norm.at<float>(j, i) > thresh) {
                keypoints.push_back(cv::KeyPoint(static_cast<float>(i), static_cast<float>(j), 5.0f));
            }
        }
    }
}

// Custom function to process and display keypoints using cv::preCornerDetect
void detectPreCornerFeatures(const cv::Mat& grayFrame, std::vector<cv::KeyPoint>& keypoints) {
    int apertureSize = 3; // Sobel aperture size (3, 5, or 7)
    int thresh = 180;      // Threshold to filter structural extrema

    cv::Mat dst = cv::Mat::zeros(grayFrame.size(), CV_32FC1);
    // Computes topological features for corner detection (partial derivatives combination)
    cv::preCornerDetect(grayFrame, dst, apertureSize);

    cv::Mat dst_norm;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Basic thresholding to gather pixels with strong structural corner responses
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int)dst_norm.at<float>(j, i) > thresh) {
                keypoints.push_back(cv::KeyPoint(static_cast<float>(i), static_cast<float>(j), 4.0f));
            }
        }
    }
}

int main() {
    // 1. Initialize video capture from default webcam (index 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video camera!" << std::endl;
        return -1;
    }

    // 2. Initialize feature detectors
    cv::Ptr<cv::ORB> orbDetector = cv::ORB::create();
    cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create();
    cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create();

    Method currentMethod = METHOD_FAST;
    cv::Mat frame, grayFrame, outputFrame;
    std::vector<cv::KeyPoint> keypoints;

    std::cout << "Controls:\n"
              << "[Space]  - Switch feature detection method\n"
              << "[ESC]    - Exit the program\n\n";

    const std::string windowName = "Feature Detection Demo";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame; // Fetch a new frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        // Convert frame to grayscale as detectors operate on single-channel images
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        keypoints.clear();

        // 3. Select and execute the chosen detection method
        switch (currentMethod) {
            case METHOD_FAST:
                fastDetector->detect(grayFrame, keypoints);
                break;
            case METHOD_HARRIS:
                detectHarrisCorners(grayFrame, keypoints);
                break;
            case METHOD_SHI_TOMASI:
                detectShiTomasiCorners(grayFrame, keypoints);
                break;
            case METHOD_MIN_EIGEN_VAL:
                detectMinEigenValCorners(grayFrame, keypoints);
                break;
            case METHOD_PRE_CORNER_DETECT:
                detectPreCornerFeatures(grayFrame, keypoints);
                break;
            default:
                break;
        }

        // 4. Draw detected keypoints over the original colored frame
        cv::drawKeypoints(frame, keypoints, outputFrame, cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // 5. Overlay text status onto the output frame
        std::string statusText = "Method: " + getMethodName(currentMethod);
        std::string countText  = "Points: " + std::to_string(keypoints.size());

        cv::putText(outputFrame, statusText, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::putText(outputFrame, countText, cv::Point(20, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        // Display the output frame in the window
        cv::imshow(windowName, outputFrame);

        // 6. Handle keyboard inputs
        int key = cv::waitKey(30); // Wait for 30 ms

        if (key == 27) { // 27 is the ASCII code for ESC
            break;
        }
        else if (key == 32) { // 32 is the ASCII code for SPACE
            // Cycle through available methods
            currentMethod = static_cast<Method>((currentMethod + 1) % METHOD_COUNT);
            std::cout << "Switched to: " << getMethodName(currentMethod) << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
