#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

// Enumeration for unified keypoint/blob/region detectors
enum Method {
    METHOD_GFTT = 0, // Good Features To Track (Shi-Tomasi wrapper)
    METHOD_ORB,
    METHOD_SIFT,
    METHOD_AKAZE,
    METHOD_BRISK,
    METHOD_COUNT     // Total number of methods
};

// Returns the method name as a string for on-screen display
std::string getMethodName(Method method) {
    switch (method) {
        case METHOD_GFTT:  return "GFTT (Good Features To Track / Shi-Tomasi)";
        case METHOD_ORB:   return "ORB (Oriented FAST & Rotated BRIEF)";
        case METHOD_SIFT:  return "SIFT (Scale-Invariant Blob Detector)";
        case METHOD_AKAZE: return "AKAZE (Accelerated KAZE)";
        case METHOD_BRISK: return "BRISK (Scale-Invariant Binary)";
        default:           return "Unknown";
    }
}

int main() {
    // 1. Initialize video capture from default webcam (index 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video camera!" << std::endl;
        return -1;
    }

    // 2. Initialize unified keypoint detectors using smart pointers
    // GFTT dynamically instantiates the Shi-Tomasi algorithm wrapped into Feature2D API
    cv::Ptr<cv::GFTTDetector> gfttDetector = cv::GFTTDetector::create(
        500,   // maxCorners: maximum number of features to retain
        0.01,  // qualityLevel: threshold multiplier for eigenvalues
        10.0,  // minDistance: minimum space between detected coordinates
        3,     // blockSize: size of the averaging neighborhood
        false, // useHarrisDetector: false forces pure Shi-Tomasi (min eigen val) criteria
        0.04   // k: Harris free parameter (ignored when useHarrisDetector is false)
    );

    cv::Ptr<cv::ORB> orbDetector = cv::ORB::create();
    cv::Ptr<cv::SIFT> siftDetector = cv::SIFT::create();
    cv::Ptr<cv::xfeatures2d::AKAZE> akazeDetector = cv::xfeatures2d::AKAZE::create();
    cv::Ptr<cv::xfeatures2d::BRISK> briskDetector = cv::xfeatures2d::BRISK::create();

    Method currentMethod = METHOD_GFTT;
    cv::Mat frame, grayFrame, outputFrame;
    std::vector<cv::KeyPoint> keypoints;

    std::cout << "Controls:\n"
              << "[Space]  - Switch keypoint detection method\n"
              << "[ESC]    - Exit the program\n\n";

    const std::string windowName = "Unified Feature Detection Demo";
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

        // 3. Select and execute the chosen keypoint detector using identical syntax
        switch (currentMethod) {
            case METHOD_GFTT:
                gfttDetector->detect(grayFrame, keypoints);
                break;
            case METHOD_ORB:
                orbDetector->detect(grayFrame, keypoints);
                break;
            case METHOD_SIFT:
                siftDetector->detect(grayFrame, keypoints);
                break;
            case METHOD_AKAZE:
                akazeDetector->detect(grayFrame, keypoints);
                break;
            case METHOD_BRISK:
                briskDetector->detect(grayFrame, keypoints);
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
