#include <opencv2/opencv.hpp>
#include <ScreenCapture.h> // Library: screen_capture_lite
#include <iostream>
#include <cmath>

// Structure to track the mouse selection state
struct SelectionContext {
    cv::Rect roi;
    bool drawing = false;
    bool selected = false;
    cv::Mat image; // Holds the original pristine screenshot
};

// Callback function to handle mouse clicks and movements in OpenCV
void onMouse(int event, int x, int y, int flags, void* userdata)
{
    SelectionContext* context = reinterpret_cast<SelectionContext*>(userdata);

    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            context->drawing = true;
            context->selected = false; // Reset previous selection on a new click
            context->roi = cv::Rect(x, y, 0, 0);
            break;

        case cv::EVENT_MOUSEMOVE:
            if (context->drawing) {
                context->roi.width = x - context->roi.x;
                context->roi.height = y - context->roi.y;
            }
            break;

        case cv::EVENT_LBUTTONUP:
            if (context->drawing) {
                context->drawing = false;

                // Adjust coordinates if selecting from right to left or bottom to top
                if (context->roi.width < 0) {
                    context->roi.x += context->roi.width;
                    context->roi.width = std::abs(context->roi.width);
                }
                if (context->roi.height < 0) {
                    context->roi.y += context->roi.height;
                    context->roi.height = std::abs(context->roi.height);
                }

                // Keep ROI within frame boundaries to avoid out_of_range exceptions
                context->roi &= cv::Rect(0, 0, context->image.cols, context->image.rows);

                // Verify that the selected area is large enough
                if (context->roi.width > 5 && context->roi.height > 5) {
                    context->selected = true;
                }
            }
            break;
    }
}

int main()
{
    // Get the list of available monitors
    auto monitors = SL::Screen_Capture::GetMonitors();
    if (monitors.empty()) {
        std::cerr << "Monitors not found!" << std::endl;
        return -1;
    }
    // Select the main monitor
    auto main_monitor = monitors[0];

    SelectionContext context;
    bool screenshotCaptured = false;

    // Take exactly ONE single screenshot using SL::Screen_Capture
    // We create a temporary configuration that captures right away and immediately stops
    auto frameGrabber = SL::Screen_Capture::CreateCaptureConfiguration([]() {
        return SL::Screen_Capture::GetMonitors();
    })->onNewFrame([&](const SL::Screen_Capture::Image& img, const SL::Screen_Capture::Monitor& monitor) {
        // Triggered once when the very first frame is ready
        if (monitor.Id != main_monitor.Id || screenshotCaptured) {
            return;
        }

        int width = SL::Screen_Capture::Width(img);
        int height = SL::Screen_Capture::Height(img);
        int rowStride = img.RowStrideInBytes;

        // Safely wrap raw buffer with stride to prevent malloc/tcache memory corruption
        cv::Mat rawFrame(
            height,
            width,
            CV_8UC4,
            (void*)SL::Screen_Capture::StartSrc(img),
            rowStride
        );

        // Convert to standard standard 3-channel BGR format and save to context
        cv::cvtColor(rawFrame, context.image, cv::COLOR_BGRA2BGR);

        screenshotCaptured = true;
    })->start_capturing();

    // Loop briefly until the single screenshot is received from the library
    std::cout << "Taking a screenshot..." << std::endl;
    while (!screenshotCaptured) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Explicitly release the frame grabber because we don't need it anymore
    frameGrabber.reset();

    if (context.image.empty()) {
        std::cerr << "Error: Failed to capture the screen!" << std::endl;
        return -1;
    }

    // Setup OpenCV interactive window
    const std::string windowName = "Select Region of Interest (Press ESC to Exit)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowName, onMouse, &context);

    // Main GUI loop (only handles user rendering interactions on the static image)
    while (true) {
        // Create a temporary copy to display the drawing box without modifying the original screenshot
        cv::Mat displayImage = context.image.clone();

        // If the user is currently dragging the mouse, draw a temporary green box
        if (context.drawing) {
            cv::Rect currentRect = context.roi;
            if (currentRect.width < 0) { currentRect.x += currentRect.width; currentRect.width = std::abs(currentRect.width); }
            if (currentRect.height < 0) { currentRect.y += currentRect.height; currentRect.height = std::abs(currentRect.height); }

            cv::rectangle(displayImage, currentRect, cv::Scalar(0, 255, 0), 2);
        }

        // If selection is complete, draw the final red box and display the crop
        if (context.selected) {
            cv::rectangle(displayImage, context.roi, cv::Scalar(0, 0, 255), 2);

            // Crop and display the selected ROI in a static window
            cv::Mat croppedRoi = context.image(context.roi);
            cv::imshow("Selected ROI Fragment", croppedRoi);
        }

        // Render the image
        cv::imshow(windowName, displayImage);

        // Press 'ESC' to exit the application
        char key = (char)cv::waitKey(30);
        if (key == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
