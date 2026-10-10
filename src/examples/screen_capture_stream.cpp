#include <opencv2/opencv.hpp>
#include <ScreenCapture.h> // Library: screen_capture_lite (SL::Screen_Capture)
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <cmath>

// Structure to track the mouse selection state and store thread-safe image data
struct SelectionContext {
    cv::Rect roi;
    bool drawing = false;
    bool selected = false;

    cv::Mat image;         // Stores the clean current frame without annotations
    std::mutex imageMtx;   // Protects access to the image matrix across threads
};

// Callback function to handle mouse clicks and movements in OpenCV
void onMouse(int event, int x, int y, int flags, void* userdata)
{
    SelectionContext* context = reinterpret_cast<SelectionContext*>(userdata);

    // Lock mutex briefly to make sure we don't read corrupted dimensions if image updates
    std::lock_guard<std::mutex> lock(context->imageMtx);
    if (context->image.empty()) return;

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
    const std::string windowName = "Screen Selection";
    cv::namedWindow(windowName);

    // Register the mouse callback function
    cv::setMouseCallback(windowName, onMouse, &context);

    // Configure SL::Screen_Capture to grab all available monitors

    bool snapshot_taken = false;

    auto frameGrabber = SL::Screen_Capture::CreateCaptureConfiguration([&]() {
        return SL::Screen_Capture::GetMonitors();
    })->onNewFrame([&](const SL::Screen_Capture::Image& img, const SL::Screen_Capture::Monitor& monitor) {
        if (monitor.Id != main_monitor.Id) {
            return;
        }

        // This lambda triggers in a background thread whenever a new frame arrives
        int width = SL::Screen_Capture::Width(img);
        int height = SL::Screen_Capture::Height(img);

        // Lock mutex to safely update the shared image matrix
        std::lock_guard<std::mutex> lock(context.imageMtx);

        // Wrap the raw buffer. screen_capture_lite provides raw BGRA/RGBA strided images.
        // We clone it immediately into our context matrix to manage memory safely.
        cv::Mat rawFrame(
            height,
            width,
            CV_8UC4,
            (void*)SL::Screen_Capture::StartSrc(img),
            img.RowStrideInBytes
        );

        // Convert to standard OpenCV 3-channel BGR format
        cv::cvtColor(rawFrame, context.image, cv::COLOR_BGRA2BGR);

    })->start_capturing();

    // Set frame capture interval (e.g., ~30 FPS -> 33 ms)
    frameGrabber->setFrameChangeInterval(std::chrono::milliseconds(33));

    // Wait a brief moment for the first frame to populate from the background thread
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    cv::Mat displayImage;

    while (true) {
        // Thread-safe duplication of the background-captured frame for current iteration
        {
            std::lock_guard<std::mutex> lock(context.imageMtx);
            if (!context.image.empty()) {
                displayImage = context.image.clone();
            }
        }

        if (displayImage.empty()) {
            std::cout << "Waiting for the first screen frame..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        // If the user is currently dragging the mouse, draw a temporary green bounding box
        if (context.drawing) {
            cv::Rect currentRect = context.roi;
            if (currentRect.width < 0) { currentRect.x += currentRect.width; currentRect.width = std::abs(currentRect.width); }
            if (currentRect.height < 0) { currentRect.y += currentRect.height; currentRect.height = std::abs(currentRect.height); }

            cv::rectangle(displayImage, currentRect, cv::Scalar(0, 255, 0), 2);
        }

        // If selection is complete, draw the final red bounding box and extract ROI
        if (context.selected) {
            cv::rectangle(displayImage, context.roi, cv::Scalar(0, 0, 255), 2);

            // Thread-safe crop of the pristine region from original frame
            cv::Mat croppedRoi;
            {
                std::lock_guard<std::mutex> lock(context.imageMtx);
                // Ensure context.roi is valid against current image state
                context.roi &= cv::Rect(0, 0, context.image.cols, context.image.rows);
                if (context.roi.width > 0 && context.roi.height > 0) {
                    croppedRoi = context.image(context.roi).clone();
                }
            }

            if (!croppedRoi.empty()) {
                cv::imshow("Selected ROI", croppedRoi);
            }
        }

        // Show the updated image frame
        cv::imshow(windowName, displayImage);

        // Press 'ESC' to exit the loop
        char key = (char)cv::waitKey(30);
        if (key == 27) break;
    }

    // Grabber automatically stops capturing when destructor is called,
    // but explicit termination ensures smooth context cleanup before window destruction
    frameGrabber.reset();
    cv::destroyAllWindows();
    return 0;
}
