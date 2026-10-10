#include <iostream>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "ScreenCapture.h"

struct FrameData {
    cv::Mat mat;
    std::mutex mtx;
    std::atomic<bool> has_new_frame{false};
};

// Global safe storage for window positioning
std::atomic<int> win_x{0}, win_y{0}, win_w{0}, win_h{0};
std::atomic<bool> window_found{false};

std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
    return str;
}

#if defined(__linux__)
#include <X11/Xlib.h>
std::pair<int, int> GetAbsoluteX11WindowPosition(unsigned long window_handle) {
    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        return {0, 0};
    }

    Window window = (Window)window_handle;
    Window root = DefaultRootWindow(display);
    Window child;
    int root_x = 0, root_y = 0;

    XTranslateCoordinates(display, window, root, 0, 0, &root_x, &root_y, &child);
    XCloseDisplay(display);
    return {root_x, root_y};
}
#endif

int main() {
    FrameData shared_frame;
    std::atomic<bool> running{true};

    // 1. Find the target window globally
    auto windows = SL::Screen_Capture::GetWindows();
    std::string search_term = "opencv course";

    for (const auto& w : windows) {
        std::string title = to_lower(std::string(w.Name));
        std::cout << "Checking window: " << title << std::endl;
        if (title.find(search_term) != std::string::npos) {
#if defined(__linux__)
            auto pos = GetAbsoluteX11WindowPosition(w.Handle);
            win_x = pos.first;
            win_y = pos.second;
#else
            win_x = w.Position.x;
            win_y = w.Position.y;
#endif
            win_w = w.Size.x;
            win_h = w.Size.y;
            window_found = true;

            std::cout << "SUCCESS: Found target window!" << std::endl;
            std::cout << "Global Position: (" << win_x << "," << win_y << ") Size: " << win_w << "x" << win_h << std::endl;
            //break;
        }
    }

    if (!window_found) {
        std::cerr << "ERROR: Target window not found!" << std::endl;
        return -1;
    }

    // 2. Select ONLY the single monitor that holds this window
    auto capture_config = SL::Screen_Capture::CreateCaptureConfiguration([]() {
        auto all_monitors = SL::Screen_Capture::GetMonitors();
        std::vector<SL::Screen_Capture::Monitor> selected_monitor;

        for (const auto& m : all_monitors) {
            int m_left = m.OffsetX;
            int m_right = m.OffsetX + m.Width;
            int m_top = m.OffsetY;
            int m_bottom = m.OffsetY + m.Height;

            // Check if window top-left corner is inside this monitor boundaries
            if (win_x >= m_left && win_x < m_right && win_y >= m_top && win_y < m_bottom) {
                std::cout << "Selected Monitor: " << m.Name << " [" << m.Width << "x" << m.Height
                          << "] at Offset (" << m.OffsetX << "," << m.OffsetY << ")" << std::endl;
                selected_monitor.push_back(m);
                break;
            }
        }

        if (selected_monitor.empty() && !all_monitors.empty()) {
            std::cout << "Warning: Mirroring window mapping failed. Using primary monitor." << std::endl;
            selected_monitor.push_back(all_monitors[0]);
        }

        return selected_monitor;
    });

    // 3. Capture frame callback (X11 window coordinates are already monitor-relative!)
    capture_config->onNewFrame([&shared_frame](const SL::Screen_Capture::Image& img, const SL::Screen_Capture::Monitor& monitor) {
        int monitor_w = SL::Screen_Capture::Width(img);
        int monitor_h = SL::Screen_Capture::Height(img);

        // Map raw BGRA data into an OpenCV Matrix
        cv::Mat full_monitor_frame(monitor_h, monitor_w, CV_8UC4, (void*)SL::Screen_Capture::StartSrc(img));

        if (window_found) {
            // CRITICAL FIX: Under Linux X11 multi-monitor setups, win_x and win_y
            // are already relative to the monitor boundaries. Do NOT subtract monitor.OffsetX!
            int crop_x = std::max(0, std::min((int)win_x, monitor_w - 1));
            int crop_y = std::max(0, std::min((int)win_y, monitor_h - 1));

            // Calculate size, making sure we don't overflow the right/bottom edge of the screen
            int crop_w = std::min((int)win_w, monitor_w - crop_x);
            int crop_h = std::min((int)win_h, monitor_h - crop_y);

            // Safety padding: clip 4 pixels to strip X11 manager shadows or invisible resizing borders
            if (crop_w > 10 && crop_h > 10) {
                crop_w = std::max(1, crop_w - 4);
                crop_h = std::max(1, crop_h - 4);

                cv::Rect roi(crop_x, crop_y, crop_w, crop_h);

                std::lock_guard<std::mutex> lock(shared_frame.mtx);
                shared_frame.mat = full_monitor_frame(roi).clone(); // Deep copy the precise window rectangle
                shared_frame.has_new_frame = true;
                return;
            }
        }
    });

    // 4. Start the async backend threads
    auto capture_manager = capture_config->start_capturing();
    capture_manager->setFrameChangeInterval(std::chrono::milliseconds(30));

    cv::namedWindow("Window Capture Stream", cv::WINDOW_NORMAL);

    while (running) {
        cv::Mat frame_to_show;
        bool got_frame = false;

        if (shared_frame.has_new_frame) {
            std::lock_guard<std::mutex> lock(shared_frame.mtx);
            if (!shared_frame.mat.empty()) {
                frame_to_show = shared_frame.mat.clone();
                shared_frame.has_new_frame = false;
                got_frame = true;
            }
        }

        if (got_frame) {
            cv::Mat bgr_frame;
            cv::cvtColor(frame_to_show, bgr_frame, cv::COLOR_BGRA2BGR);
            cv::imshow("Window Capture Stream", bgr_frame);
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            running = false;
        }
    }

    capture_manager->pause();
    cv::destroyAllWindows();
    return 0;
}
