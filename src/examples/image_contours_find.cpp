#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <string>

// Global structures to share data between main and callbacks
cv::Mat src, grayImg, edges;
const std::string windowName = "Object Detection Tuning";
std::vector<std::vector<cv::Point>> filteredContours;
int currentIdx = 0;
int currentMethod = 2;

void updateContours_A()
{
    int cannyLow = cv::getTrackbarPos("Canny Low", windowName);
    int cannyHigh = cv::getTrackbarPos("Canny High", windowName);

    cv::Mat thresholdImg;

    cv::threshold(
        grayImg,               // Source image (single-channel, grayscale)
        thresholdImg,          // Destination image of the same size and type as src
        100.0,              // Threshold value used to classify pixel intensities
        255.0,              // Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV
        cv::THRESH_BINARY |   // Thresholding type (e.g., THRESH_BINARY, THRESH_OTSU)
        cv::THRESH_OTSU
    );

    // Canny Edge Detection
    cv::Canny(thresholdImg, edges, cannyLow, cannyHigh);
}

void updateContours_B()
{
    int cannyLow = cv::getTrackbarPos("Canny Low", windowName);
    int cannyHigh = cv::getTrackbarPos("Canny High", windowName);

    cv::Mat thresholdImg;

    cv::adaptiveThreshold(
        grayImg,                     // Source 8-bit single-channel image
        thresholdImg,                // Destination image of the same size and type as src
        255.0       ,           // Non-zero value assigned to pixels for which the condition is satisfied
        cv::ADAPTIVE_THRESH_MEAN_C, // Adaptive thresholding algorithm (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
        cv::THRESH_BINARY, // Thresholding type (must be THRESH_BINARY or THRESH_BINARY_INV)
        11,                    // Size of a pixel neighborhood used to calculate a threshold value (odd number: 3, 5, 7, etc.)
        5.0                            // Constant subtracted from the mean or weighted mean
    );

    // Canny Edge Detection
    cv::Canny(thresholdImg, edges, cannyLow, cannyHigh);
}

// Function to update contours based on current trackbar positions
void updateContours_C()
{
    // Safely fetch values from trackbars
    int blurSizeInt = cv::getTrackbarPos("Blur Size", windowName);
    int cannyLow = cv::getTrackbarPos("Canny Low", windowName);
    int cannyHigh = cv::getTrackbarPos("Canny High", windowName);

    cv::Mat blurred;

    // Adaptive Blur
    int kSize = 2 * blurSizeInt + 1;
    cv::GaussianBlur(
        grayImg,
        blurred,
        cv::Size(kSize,
        kSize),
        0);

    // Canny Edge Detection
    cv::Canny(blurred, edges, cannyLow, cannyHigh);
}

void findContours()
{
    int minAreaThresh = cv::getTrackbarPos("Min Area", windowName);

    // Find raw contours
    std::vector<std::vector<cv::Point>> rawContours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
        edges,
        rawContours,
        hierarchy,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    // Filter contours by minimum area
    filteredContours.clear();
    for (const auto& contour : rawContours) {
        if (cv::contourArea(contour) >= minAreaThresh) {
            filteredContours.push_back(contour);
        }
    }

    // Reset index if it goes out of bounds after filtering
    if (currentIdx >= static_cast<int>(filteredContours.size())) {
        currentIdx = 0;
    }
}

void drawContours()
{
    cv::Mat displayMat = src.clone();

    // Draw all detected objects with a thin gray line
    cv::drawContours(displayMat, filteredContours, -1, cv::Scalar(128, 128, 128), 1);

    // Highlight the active object if any exist
    if (!filteredContours.empty()) {
        cv::drawContours(displayMat, filteredContours, currentIdx, cv::Scalar(0, 255, 0), 3);

        double area = cv::contourArea(filteredContours[currentIdx]);
        std::string text = "Object: " + std::to_string(currentIdx + 1) + " / " + std::to_string(filteredContours.size());
        std::string areaText = "Area: " + std::to_string(static_cast<int>(area)) + " px";
        cv::putText(displayMat, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(displayMat, areaText, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    } else {
        cv::putText(displayMat, "No objects match filter", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow(windowName, displayMat);
}


void onTrackbarChange(int, void*)
{
    switch (currentMethod)
    {
        case 0:
            updateContours_A(); // Method A: OTSU
            break;

        case 1:
            updateContours_B(); // Method B: Adaptive threshold
            break;

        case 2:
            updateContours_C(); // Method C: Gaussian blur
            break;
    }

    findContours();
    drawContours();
}

void on_radio_change(int state, void* userdata)
{
    if (state != 1 || userdata == nullptr)
        return;

    const std::string& buttonName =
        *static_cast<std::string*>(userdata);

    if (buttonName == "A")
        currentMethod = 0;
    else if (buttonName == "B")
        currentMethod = 1;
    else if (buttonName == "C")
        currentMethod = 2;
    else
        return;

    onTrackbarChange(0, nullptr);

    cv::displayStatusBar(
        windowName,
        "Selected method: " + buttonName,
        2000
    );
}

int main()
{
    std::string imagePath = "data/objects.jpg";
    src = cv::imread(imagePath);

    if (src.empty()) {
        std::cout << "Could not load the image: " << imagePath << std::endl;
        return -1;
    }

    cv::cvtColor(src, grayImg, cv::COLOR_BGR2GRAY);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // Create sliders using safe nullptr instead of variable pointers
    cv::createTrackbar("Blur Size", windowName, nullptr, 15, onTrackbarChange);
    cv::createTrackbar("Canny Low", windowName, nullptr, 255, onTrackbarChange);
    cv::createTrackbar("Canny High", windowName, nullptr, 255, onTrackbarChange);
    cv::createTrackbar("Min Area", windowName, nullptr, 5000, onTrackbarChange);

    // Set initial configuration positions
    cv::setTrackbarPos("Blur Size", windowName, 1);
    cv::setTrackbarPos("Canny Low", windowName, 25);
    cv::setTrackbarPos("Canny High", windowName, 65);
    cv::setTrackbarPos("Min Area", windowName, 100);

    std::string optA = "A";
    std::string optB = "B";
    std::string optC = "C";

    cv::createButton("method A", on_radio_change,
                    &optA, cv::QT_RADIOBOX, false);

    cv::createButton("method B", on_radio_change,
                    &optB, cv::QT_RADIOBOX, false);

    cv::createButton("method C", on_radio_change,
                    &optC, cv::QT_RADIOBOX, true);

    // Initial render
    onTrackbarChange(0, nullptr);

    std::cout << "1. Adjust sliders to isolate the object (green contour)." << std::endl;
    std::cout << "2. Press [Space] to cycle through found objects." << std::endl;
    std::cout << "3. Press [Esc] to exit." << std::endl;

        cv::displayStatusBar(windowName, "Инициализация интерфейса...", 1);

    while (true) {
        try {
            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
                std::cout << "Window closed by user. Exiting..." << std::endl;
                break;
            }
        }
        catch (const cv::Exception& e) {
            std::cout << "Window interface destroyed. Exiting..." << std::endl;
            break;
        }

        int key = cv::waitKey(10);

        if (key == 27) { // Esc
            break;
        }
        else if (key == 32) { // Space
            if (!filteredContours.empty()) {
                currentIdx = (currentIdx + 1) % filteredContours.size();
                // Force redraw with the new active index
                onTrackbarChange(0, nullptr);
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}
