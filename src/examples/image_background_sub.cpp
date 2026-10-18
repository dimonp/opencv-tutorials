#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>

int main() {
    // 1. Initialize video capture from the default webcam (index 0)
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open the camera!" << std::endl;
        return -1;
    }

    // 2. Create the Background Subtractor object using the MOG2 algorithm
    // history = 500 frames to learn the background model
    // varThreshold = 16 (Mahalanobis distance threshold to filter noise)
    // detectShadows = true (enables shadow detection, marked in gray)
    cv::Ptr<cv::BackgroundSubtractorMOG2> pBackSub = cv::createBackgroundSubtractorMOG2(
        500, 16, true);

    cv::Mat frame;  // Holds the source camera frame
    cv::Mat fgMask; // Holds the foreground mask output

    std::cout << "Application started. Press 'ESC' to exit." << std::endl;

    while (true) {
        // Grab the current frame
        capture >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty or stream has ended." << std::endl;
            break;
        }

        // 3. Apply the background subtraction model to the current frame
        // The apply() method updates the background model and generates the binary fgMask
        pBackSub->apply(frame, fgMask);

        // (Optional) Remove small noise using morphological opening
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);

        // 4. Display the results in windows
        cv::imshow("Original Frame", frame);   // Source video
        cv::imshow("FG Mask", fgMask);         // Motion mask (White = object, Gray = shadow, Black = background)

        // Wait for a key press (27 is the ASCII code for the ESC key)
        char key = (char)cv::waitKey(30);
        if (key == 27) {
            break;
        }
    }

    // Release resources and close windows
    capture.release();
    cv::destroyAllWindows();
    return 0;
}
