#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // 1. Create a blank black image: 450 pixels high and 900 pixels wide (Wide & Short layout)
    // Parameters: (rows/height, cols/width, type)
    cv::Mat image = cv::Mat::zeros(450, 900, CV_8UC3);

    // Define colors in BGR format
    cv::Scalar red(0, 0, 255);
    cv::Scalar green(0, 255, 0);
    cv::Scalar blue(255, 0, 0);
    cv::Scalar yellow(0, 255, 255);
    cv::Scalar white(255, 255, 255);
    cv::Scalar magenta(255, 0, 255);
    cv::Scalar cyan(255, 255, 0);
    cv::Scalar orange(0, 165, 255);

    // ==========================================
    // ROW 1: Top elements
    // ==========================================

    // 2. Draw a line
    cv::line(image, cv::Point(50, 40), cv::Point(850, 40), red, 3);

    // 3. Draw a rectangle (outline only)
    cv::rectangle(image, cv::Point(50, 80), cv::Point(220, 200), green, 3);

    // 4. Draw a filled rectangle
    cv::rectangle(image, cv::Point(260, 80), cv::Point(430, 200), blue, cv::FILLED);

    // 5. Draw a circle
    cv::circle(image, cv::Point(530, 140), 60, yellow, 4);

    // 6. Draw an ellipse
    cv::ellipse(image, cv::Point(740, 140), cv::Size(80, 40), 30, 0, 360, white, 2);

    // 7. Draw an Arc (an unclosed ellipse segment)
    cv::ellipse(image, cv::Point(135, 310), cv::Size(60, 60), 0, 0, 270, magenta, 3);

    // 8. Draw a Sector (a filled pie-like slice)
    cv::ellipse(image, cv::Point(345, 310), cv::Size(60, 60), 0, 45, 315, cyan, cv::FILLED);

    // 9. Draw a Polygon (a closed shape defined by points)
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(490, 350));
    pts.push_back(cv::Point(530, 270));
    pts.push_back(cv::Point(610, 280));
    pts.push_back(cv::Point(630, 350));
    pts.push_back(cv::Point(560, 380));
    std::vector<std::vector<cv::Point>> pp{pts};
    cv::polylines(image, pp, true, white, 3, cv::LINE_AA);

    // 10. Draw an Arrowed Line
    cv::arrowedLine(image, cv::Point(690, 310), cv::Point(850, 310), orange, 3, cv::LINE_AA, 0, 0.2);


    // ==========================================
    // TEXT AND WINDOW DISPLAY
    // ==========================================

    // 11. Render text at the bottom center
    cv::putText(image, "OpenCV Wide Layout", cv::Point(330, 420),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv::LINE_AA);

    // 12. Display the result in a window
    std::string windowName = "Wide Geometric Drawing";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);

    // Wait for any key press to close the window
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
