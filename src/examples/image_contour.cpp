#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Load the input image
    cv::Mat src = cv::imread("data/cars.jpg");
    if (src.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Step 1: Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Step 2: Apply blurring to reduce high-frequency noise
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    // Step 3: Apply Canny edge detector to get a binary edge image
    cv::Mat edges;
    cv::Canny(gray, edges, 100, 200);

    // Variables to store contours and hierarchy tree information
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // Step 4: Find all contours in the binary image
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a copy of the source image to draw contours on
    cv::Mat drawing = src.clone();

    // Step 5: Draw the detected contours
    cv::drawContours(drawing, contours, -1, cv::Scalar(0, 255, 0), 2);

    // Step 6: Create resizable windows so they don't exceed screen boundaries
    // WINDOW_NORMAL allows the window to be resized and forces it to fit the screen
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Edges (Canny)", cv::WINDOW_NORMAL);
    cv::namedWindow("Detected Contours", cv::WINDOW_NORMAL);

    // Optional: Explicitly set a preferred size for the windows (e.g., 800x600)
    // You can remove or adjust these lines depending on your screen resolution
    cv::resizeWindow("Original Image", 800, 600);
    cv::resizeWindow("Edges (Canny)", 800, 600);
    cv::resizeWindow("Detected Contours", 800, 600);

    // Step 7: Display the results in the prepared windows
    cv::imshow("Original Image", src);
    cv::imshow("Edges (Canny)", edges);
    cv::imshow("Detected Contours", drawing);

    // Wait until the user presses any key
    cv::waitKey(0);
    return 0;
}
