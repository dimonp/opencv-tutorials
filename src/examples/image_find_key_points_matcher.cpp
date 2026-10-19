#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Initialize video capture from default webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video camera!" << std::endl;
        return -1;
    }

    // 2. Initialize feature pipeline components
    // We use GFTT to detect stable corners, and ORB to compute descriptors for them
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(400, 0.01, 10);
    cv::Ptr<cv::ORB> descriptorExtractor = cv::ORB::create();

    // Brute-Force matcher with HAMMING distance (since ORB produces binary descriptors)
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::Mat objectFrame, objectDescriptors;
    std::vector<cv::KeyPoint> objectKeypoints;
    bool hasTargetObject = false;

    std::cout << "Controls:\n"
              << "[Space] - Capture the current frame as the TARGET object to track\n"
              << "[ESC]   - Exit the program\n\n";

    const std::string windowName = "Feature Matching Demo";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::Mat currentFrame, grayCurrent, currentDescriptors;
    std::vector<cv::KeyPoint> currentKeypoints;

    while (true) {
        cap >> currentFrame;
        if (currentFrame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        cv::cvtColor(currentFrame, grayCurrent, cv::COLOR_BGR2GRAY);

        // If the user hasn't captured a target object yet, just show the camera feed with keypoints
        if (!hasTargetObject) {
            detector->detect(grayCurrent, currentKeypoints);
            cv::Mat previewFrame;
            cv::drawKeypoints(currentFrame, currentKeypoints, previewFrame, cv::Scalar(0, 255, 0));

            cv::putText(previewFrame, "Press [SPACE] to capture target object", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            cv::imshow(windowName, previewFrame);
        }
        else {
            // STEP 1 & 2: Detect and compute features for the live camera scene
            currentKeypoints.clear();
            detector->detect(grayCurrent, currentKeypoints);
            descriptorExtractor->compute(grayCurrent, currentKeypoints, currentDescriptors);

            cv::Mat matchOutputImg;

            // Check if both images actually contain valid descriptors to prevent crashes
            if (!objectDescriptors.empty() && !currentDescriptors.empty()) {
                // STEP 3: K-Nearest Neighbors matching (k=2) to apply Lowe's ratio test
                std::vector<std::vector<cv::DMatch>> knnMatches;
                matcher->knnMatch(objectDescriptors, currentDescriptors, knnMatches, 2);

                // Filter matches using Lowe's ratio test to eliminate ambiguous points
                std::vector<cv::DMatch> goodMatches;
                const float ratioThreshold = 0.75f; // Closeness factor between best and second-best match

                for (size_t i = 0; i < knnMatches.size(); i++) {
                    if (knnMatches[i].size() >= 2) {
                        if (knnMatches[i][0].distance < ratioThreshold * knnMatches[i][1].distance) {
                            goodMatches.push_back(knnMatches[i][0]);
                        }
                    }
                }

                // Draw the side-by-side comparison screen with connection lines
                cv::drawMatches(objectFrame, objectKeypoints, currentFrame, currentKeypoints, goodMatches, matchOutputImg,
                                cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), std::vector<char>(),
                                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                std::string matchCountText = "Good Matches: " + std::to_string(goodMatches.size());
                cv::putText(matchOutputImg, matchCountText, cv::Point(20, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            } else {
                // Fallback layout if no features match up at all
                cv::hconcat(objectFrame, currentFrame, matchOutputImg);
            }

            cv::imshow(windowName, matchOutputImg);
        }

        // Handle keyboard keys
        int key = cv::waitKey(30);
        if (key == 27) { // ESC key
            break;
        }
        else if (key == 32) { // SPACE key -> Save target image context
            objectFrame = currentFrame.clone();
            cv::Mat grayObject;
            cv::cvtColor(objectFrame, grayObject, cv::COLOR_BGR2GRAY);

            // Compute reference descriptors for the saved object
            objectKeypoints.clear();
            detector->detect(grayObject, objectKeypoints);
            descriptorExtractor->compute(grayObject, objectKeypoints, objectDescriptors);

            if (objectKeypoints.empty()) {
                std::cout << "Capture failed: No keypoints found in the current frame. Try again!" << std::endl;
                hasTargetObject = false;
            } else {
                std::cout << "Target captured! Keypoints: " << objectKeypoints.size() << std::endl;
                hasTargetObject = true;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
