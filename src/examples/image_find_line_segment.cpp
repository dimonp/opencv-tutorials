#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Load the source image
    cv::Mat src = cv::imread("data/objects1.jpg", cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not load the image! Check the file path." << std::endl;
        return -1;
    }

    // 2. Preprocessing: Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 3. Preprocessing: Apply Gaussian Blur to reduce noise and avoid false lines
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // 4. Edge Detection: Apply Canny operator
    // Hough transform operates on binary images containing edges (white pixels)
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 200, 3);

    // 5. Line Segment Detection: Apply Probabilistic Hough Transform (HoughLinesP)
    // Each line segment is stored as a Vec4i: (x_start, y_start, x_end, y_end)
    std::vector<cv::Vec4i> lines;

    double rho = 1;              // Distance resolution of the accumulator in pixels
    double theta = CV_PI / 180;  // Angle resolution of the accumulator in radians (1 degree)
    int threshold = 50;          // Minimum number of intersections to detect a line
    double minLineLength = 50;   // Minimum number of pixels making up a line segment
    double maxLineGap = 10;      // Maximum allowed gap between points on the same line to link them

    cv::HoughLinesP(edges, lines, rho, theta, threshold, minLineLength, maxLineGap);

    // 6. Draw the detected line segments on the original image
    cv::Mat result = src.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];

        // Draw each line segment in green color with a thickness of 2 pixels
        cv::line(result, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    // 7. Display the results
    cv::imshow("Source Image", src);
    cv::imshow("Canny Edges", edges);
    cv::imshow("Detected Line Segments", result);

    cv::waitKey(0); // Wait indefinitely for a key press
    cv::destroyAllWindows();
    return 0;
}
