#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>

// Function to draw HUD elements (runs on CPU)
void drawHUD(cv::Mat& frame, double fps, bool useGPU) {
    std::string modeText = useGPU ? "MODE: GPU (OpenCL)" : "MODE: CPU";
    std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));

    // Colors: Green for GPU, Red for CPU
    cv::Scalar color = useGPU ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    // 1. Draw a border frame around the screen edges
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(frame.cols - 10, frame.rows - 10), color, 2);

    // 2. Display text (Current Mode and FPS)
    cv::putText(frame, modeText, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    cv::putText(frame, fpsText, cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    cv::putText(frame, "Press SPACE to toggle mode. ESC to exit.", cv::Point(30, frame.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // 3. Draw a crosshair in the center of the frame
    cv::Point center(frame.cols / 2, frame.rows / 2);
    cv::circle(frame, center, 20, color, 2);
    cv::line(frame, cv::Point(center.x - 30, center.y), cv::Point(center.x + 30, center.y), color, 2);
    cv::line(frame, cv::Point(center.x, center.y - 30), cv::Point(center.x, center.y + 30), color, 2);
}

int main() {
    // Check if OpenCL is supported by the system
    if (!cv::ocl::haveOpenCL()) {
        std::cout << "Critical Error: OpenCL is not supported on this system!" << std::endl;
        return -1;
    }

    // Explicitly enable OpenCL usage (T-API)
    cv::ocl::setUseOpenCL(true);
    std::cout << "OpenCL Status: " << (cv::ocl::useOpenCL() ? "ENABLED" : "DISABLED") << std::endl;

    // Initialize the camera (0 is usually the default built-in webcam)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Configure frame resolution (HD resolution provides a visible workload benchmark)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::Mat frame, finalFrame;
    cv::UMat u_frame, u_blur; // UMat objects will be automatically processed on the AMD GPU

    bool useGPU = true;
    double fps = 0.0;

    const std::string windowName = "OpenCV Acceleration Demo";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        cap >> frame; // Capture frame from camera to system RAM
        if (frame.empty()) break;

        // Huge blur kernel size (larger sizes create a bigger performance gap between CPU and GPU)
        cv::Size blurSize(45, 45);

        if (useGPU) {
            // --- GPU MODE ---
            // Upload pixel data from CPU (Mat) to GPU VRAM (UMat)
            frame.copyTo(u_frame);

            // Image filtering runs on the AMD GPU via OpenCL kernels
            cv::GaussianBlur(u_frame, u_blur, blurSize, 0);

            // Download the processed frame back to RAM for rendering and HUD drawing
            u_blur.copyTo(finalFrame);
        } else {
            // --- CPU MODE ---
            // All computations are executed exclusively on the processor
            cv::GaussianBlur(frame, finalFrame, blurSize, 0);
        }

        // Draw HUD overlay on top of the processed frame (executed on CPU)
        drawHUD(finalFrame, fps, useGPU);

        // Display the final output window
        cv::imshow(windowName, finalFrame);

        // Calculate real-time FPS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        fps = 1.0 / diff.count();

        // Handle keyboard inputs
        int key = cv::waitKey(1);
        if (key == 27) { // ESC key to exit
            break;
        } else if (key == 32) { // SPACE key to toggle mode
            useGPU = !useGPU;
            std::cout << "Switching mode. Use GPU: " << (useGPU ? "YES" : "NO") << std::endl;
        }
    }

    // Release camera resources and close windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
