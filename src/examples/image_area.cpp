#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Structure to track the mouse selection state
struct SelectionContext {
    cv::Rect roi;
    bool drawing = false;
    bool selected = false;
    cv::Mat image; // Stores the clean current frame without annotations
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
    // Initialize the selection context
    SelectionContext context;

    // Placeholder: create a blank image for testing (replace with your ScreenCapture logic)
    context.image = cv::Mat::zeros(600, 800, CV_8UC3);
    cv::putText(context.image, "Hold Left Mouse Button and drag to select ROI",
                cv::Point(50, 300), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    const std::string windowName = "Screen Selection";
    cv::namedWindow(windowName);

    // Register the mouse callback function
    cv::setMouseCallback(windowName, onMouse, &context);

    while (true) {
        if (context.image.empty()) {
            std::cerr << "Error: Captured frame is empty!" << std::endl;
            break;
        }

        // 2. Create a temporary clone for visualization to prevent frame pollution
        cv::Mat displayImage = context.image.clone();

        // 3. If the user is currently dragging the mouse, draw a temporary bounding box
        if (context.drawing) {
            // Handle negative values temporarily during live drawing (moving left/up)
            cv::Rect currentRect = context.roi;
            if (currentRect.width < 0) { currentRect.x += currentRect.width; currentRect.width = std::abs(currentRect.width); }
            if (currentRect.height < 0) { currentRect.y += currentRect.height; currentRect.height = std::abs(currentRect.height); }

            cv::rectangle(displayImage, currentRect, cv::Scalar(0, 255, 0), 2);
        }

        // 4. If selection is complete, draw the final bounding box and extract ROI
        if (context.selected) {
            cv::rectangle(displayImage, context.roi, cv::Scalar(0, 0, 255), 2);

            // Crop and display the selected region in a separate window
            cv::Mat croppedRoi = context.image(context.roi);
            cv::imshow("Selected ROI", croppedRoi);
        }

        // Show the updated image frame
        cv::imshow(windowName, displayImage);

        // Press 'ESC' to exit the loop
        char key = (char)cv::waitKey(30);
        if (key == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
