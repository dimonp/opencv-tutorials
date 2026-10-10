#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::string image_path = "data/test.png";

    // IMREAD_COLOR loads the image in default BGR format
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Could not open or find the image: " << image_path << std::endl;
        return -1;
    }

    // Process the image: Convert the color space from BGR to Grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Create windows and display both images
    cv::imshow("Original Image", image);
    cv::imshow("Grayscale Image", gray_image);

    // Wait for a keystroke in any window before closing (0 means wait indefinitely)
    // Without this, the windows would close instantly
    cv::waitKey(0);

    cv::destroyAllWindows();
}
