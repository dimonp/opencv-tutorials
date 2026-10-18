#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <string>

// Enum to keep track of the currently selected color
enum ColorMode {
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    COLOR_COUNT = 3 // Used for cycling via modulo
};

int main() {
    // 1. Initialize video capture from the default webcam
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open the camera!" << std::endl;
        return -1;
    }

    cv::Mat frame, hsv, mask;
    ColorMode current_mode = BLUE;
    std::string mode_text = "Tracking: BLUE";

    std::cout << "Application started." << std::endl;
    std::cout << "Press 'SPACE' to switch colors (Blue -> Green -> Red)." << std::endl;
    std::cout << "Press 'ESC' to exit." << std::endl;

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        // 2. Convert the image from BGR to HSV color space
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 3. Define bounds and process mask based on the selected color mode
        cv::Scalar bgr_color; // Color for the drawing rectangle

        if (current_mode == BLUE) {
            cv::Scalar lower_bound(100, 100, 100);
            cv::Scalar upper_bound(140, 255, 255);
            cv::inRange(hsv, lower_bound, upper_bound, mask);
            bgr_color = cv::Scalar(255, 0, 0); // Blue in BGR
            mode_text = "Tracking: BLUE";
        }
        else if (current_mode == GREEN) {
            cv::Scalar lower_bound(35, 100, 100);
            cv::Scalar upper_bound(85, 255, 255);
            cv::inRange(hsv, lower_bound, upper_bound, mask);
            bgr_color = cv::Scalar(0, 255, 0); // Green in BGR
            mode_text = "Tracking: GREEN";
        }
        else if (current_mode == RED) {
            // Red hue wraps around 0 and 180, so we need two intervals combined
            cv::Mat mask1, mask2;

            // Lower red range (0 - 10)
            cv::Scalar lower_bound1(0, 100, 100);
            cv::Scalar upper_bound1(10, 255, 255);
            cv::inRange(hsv, lower_bound1, upper_bound1, mask1);

            // Upper red range (170 - 180)
            cv::Scalar lower_bound2(170, 100, 100);
            cv::Scalar upper_bound2(180, 255, 255);
            cv::inRange(hsv, lower_bound2, upper_bound2, mask2);

            // Combine both masks into one
            cv::bitwise_or(mask1, mask2, mask);

            bgr_color = cv::Scalar(0, 0, 255); // Red in BGR
            mode_text = "Tracking: RED";
        }

        // 4. Morphological operations to clean up the binary mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // 5. Find contours of the detected color segments
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. Track the largest object of the selected color
        double max_area = 0;
        int largest_contour_index = -1;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                largest_contour_index = i;
            }
        }

        // 7. Draw bounding box if object is found
        if (largest_contour_index != -1 && max_area > 500) {
            cv::Rect bounding_rect = cv::boundingRect(contours[largest_contour_index]);
            cv::rectangle(frame, bounding_rect, bgr_color, 2, cv::LINE_AA);
            cv::putText(frame, "Object Found", cv::Point(bounding_rect.x, bounding_rect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2);
        }

        // 8. Overlay the current tracking mode text onto the video frame
        cv::putText(frame, mode_text, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2);

        // 9. Display the windows
        cv::imshow("Multi-Color Tracking", frame);
        cv::imshow("Active Color Mask", mask);

        // 10. Handle key events
        int key = cv::waitKey(30);

        if (key == 27) { // ESC key to exit
            break;
        }
        else if (key == 32) { // SPACE key to switch color mode
            // Increment mode and cycle back using modulo
            current_mode = static_cast<ColorMode>((current_mode + 1) % COLOR_COUNT);
            std::cout << "Switched mode!" << std::endl;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}
