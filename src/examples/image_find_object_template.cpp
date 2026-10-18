#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Load the source image and the template image
    // The template must be smaller than or equal to the source image
    cv::Mat img = cv::imread("data/objects.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("data/cube.png", cv::IMREAD_COLOR);

    if (img.empty() || templ.empty()) {
        std::cerr << "Error: Could not load images! Check the file paths." << std::endl;
        return -1;
    }

    // 2. Create a matrix to store the comparison results
    // The result matrix size depends on the source and template sizes
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    cv::Mat result;
    result.create(result_rows, result_cols, CV_32FC1);

    // 3. Perform template matching
    // TM_CCOEFF_NORMED is one of the most robust methods (values range from -1 to 1)
    // Higher values mean better match for this specific method
    int match_method = cv::TM_CCOEFF_NORMED;
    cv::matchTemplate(img, templ, result, match_method);

    // 4. Find the best match position in the result matrix
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::Point matchLoc;

    // minMaxLoc finds the minimum and maximum values and their positions in the matrix
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    // For TM_SQDIFF and TM_SQDIFF_NORMED, the best match is the minimum value (minLoc)
    // For other methods (like TM_CCOEFF_NORMED), the best match is the maximum value (maxLoc)
    if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED) {
        matchLoc = minLoc;
    } else {
        matchLoc = maxLoc;
    }

    // (Optional) Define a threshold to verify if the object was actually found
    double threshold = 0.8; // 80% confidence match
    if (maxVal >= threshold) {
        std::cout << "Object found with confidence: " << maxVal * 100 << "%" << std::endl;

        // 5. Draw a bounding box around the detected object
        // The rectangle starts at matchLoc and has the same width/height as the template
        cv::Point bottom_right = cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows);
        cv::rectangle(img, matchLoc, bottom_right, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    } else {
        std::cout << "Object not found. Best match confidence was only: " << maxVal * 100 << "%" << std::endl;
    }

    // 6. Display the result
    cv::imshow("Detected Object", img);
    cv::imshow("Template Used", templ);

    cv::waitKey(0); // Wait indefinitely for a key press
    cv::destroyAllWindows();
    return 0;
}
