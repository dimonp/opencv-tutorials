#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera (index 0)
    cv::VideoCapture cap(0);

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat processedFrame;

    // Create a window to display the output
    cv::namedWindow("Video Processing", cv::WINDOW_AUTOSIZE);

    while (true) {
        // Capture the next frame from the camera
        cap >> frame;

        // Break the loop if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        // Convert the frame from BGR to Grayscale
        cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);

        // Apply Gaussian Blur to reduce noise (kernel size must be odd)
        cv::GaussianBlur(processedFrame, processedFrame, cv::Size(7, 7), 0);

        // Convert back to BGR so we can draw colorful shapes on top
        cv::cvtColor(processedFrame, processedFrame, cv::COLOR_GRAY2BGR);

        // Draw a green rectangle in the center area of the frame
        cv::rectangle(processedFrame,
                      cv::Point(150, 150),
                      cv::Point(350, 350),
                      cv::Scalar(0, 255, 0),
                      2);

        // 5. Add status text onto the processed frame
        cv::putText(processedFrame,
                    "Status: Active",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(255, 0, 0),
                    2);

        // Display the newly modified frame
        cv::imshow("Video Processing", processedFrame);

        // Wait for 30ms; if 'Esc' is pressed, exit the loop
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    // Release the camera and destroy all created windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
