#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera (index 0)
    cv::VideoCapture cap(0);

    // Check if camera is opened
    if (!cap.isOpened()) {
        std::cerr << "Error:: Could not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Video stream", cv::WINDOW_AUTOSIZE);

    while (true) {
        // Read a frame from the camera
        // Operator >> captures a frame from the camera
        cap >> frame;

        // Check if frame is empty
        if (frame.empty()) {
            std::cerr << "Error:: frame is empty" << std::endl;
            break;
        }

        // Show the frame in the window
        cv::imshow("Video stream", frame);

        // Wait for 30 ms and check if 'Esc' key is pressed
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    // Release the camera and close the window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
