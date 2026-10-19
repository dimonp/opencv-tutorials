#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <string>

/*
• Otsu's Thresholding
	• Concept: Calculates a single global threshold value automatically by analyzing the image histogram.
	• Ideal for: Binarizing high-contrast images (e.g., separating black text from a white background) [result: 0].
	• Pros/Cons: Very fast, but fails under uneven or variable lighting conditions.
• Adaptive Thresholding
	• Concept: Computes a unique threshold for each pixel based on the brightness of its small local neighborhood.
	• Ideal for: Scanning text and detecting documents with shadows or gradients.
	• Pros/Cons: Robust to illumination changes, but introduces severe noise in uniformly flat areas.
• Canny Edge Detection
	• Concept: Uses structural gradients to locate intensity discontinuities (edges) and links them together.
	• Ideal for: Outlining object shapes, boundaries, and sketch-like features.
	• Pros/Cons: Excellent at structural extraction, but does not provide filled, solid regional masks natively.
• Watershed Segmentation
	• Concept: Treats image intensities as topographic relief and simulates flooding them with water from seed markers.
	• Ideal for: Isolating and splitting touching or overlapping objects (e.g., counting coins or cells).
	• Pros/Cons: High precision for complex shapes, but highly prone to over-segmentation without accurate markers.
• Mean Shift Filtering
	• Concept: An iterative clustering approach that smooths textures by shifting data points toward local mode densities.
	• Ideal for: Posterization, smoothing fine-grained noise, and cartoon-like image effects.
	• Pros/Cons: Preserves sharp boundaries beautifully, but has heavy CPU load and drops live framerates.
• Flood Fill
	• Concept: Starts at a specified seed pixel and colors connected neighboring regions that fall within a color tolerance.
	• Ideal for: "Magic Wand" selections, filling uniform spaces, and region growing.
	• Pros/Cons: Highly efficient, but leaks easily if object boundaries have even tiny gaps.
• GrabCut
	• Concept: An optimization graph-cut algorithm that builds Gaussian Mixture Models to iteratively isolate foreground from background.
	• Ideal for: Professional photo editing foreground extraction using a bounding box.
	• Pros/Cons: Exceptionally precise for complex objects, but too heavy for full-resolution real-time video feeds.
• K-Means Clustering
	• Concept: Partitioning pixels into K distinctive color groups by minimizing distances to cluster centroids.
	• Ideal for: Finding dominant colors, quantization, and color-based regional grouping.
	• Pros/Cons: Simple and effective color grouping, but ignores pixel spatial relationships and requires a predefined K value
*/

// Enum for all 7 segmentation methods
enum SegmentationMode {
    OTSU_THRESHOLD = 0,
    ADAPTIVE_THRESHOLD = 1,
    CANNY_EDGES = 2,
    WATERSHED_SEGMENT = 3,
    MEAN_SHIFT_FILTER = 4,
    FLOOD_FILL_MODE = 5,
    GRAB_CUT_MODE = 6,
    KMEANS_MODE = 7,
    MODE_COUNT = 8 // Total number of modes
};

int main() {
    // 1. Initialize video capture from the default webcam
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open the camera!" << std::endl;
        return -1;
    }

    cv::Mat frame, gray, result;
    SegmentationMode current_mode = OTSU_THRESHOLD;
    std::string mode_text = "";

    std::cout << "Application started." << std::endl;
    std::cout << "Press 'SPACE' to switch segmentation methods." << std::endl;
    std::cout << "Press 'ESC' to exit." << std::endl;

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        // Preprocessing: convert to grayscale (used for thresholding methods)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // 2. Apply the selected segmentation method
        switch (current_mode) {

            case OTSU_THRESHOLD: {
                mode_text = "Method: Otsu's Thresholding";
                cv::threshold(gray, result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                break;
            }

            case ADAPTIVE_THRESHOLD: {
                mode_text = "Method: Adaptive Thresholding";
                cv::adaptiveThreshold(gray, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv::THRESH_BINARY, 11, 2);
                break;
            }

            case CANNY_EDGES: {
                mode_text = "Method: Canny Edge Segmentation";
                cv::Mat edges;
                cv::Canny(gray, edges, 50, 150);
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::morphologyEx(edges, result, cv::MORPH_CLOSE, kernel);
                break;
            }

            case WATERSHED_SEGMENT: {
                mode_text = "Method: Watershed Segmentation";
                cv::Mat bin;
                cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

                cv::Mat dist;
                cv::distanceTransform(bin, dist, cv::DistanceTypes::DIST_L2, 3);
                cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

                cv::Mat sure_fg;
                cv::threshold(dist, sure_fg, 0.4, 255, cv::THRESH_BINARY);
                sure_fg.convertTo(sure_fg, CV_8U);

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(sure_fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                cv::Mat markers = cv::Mat::zeros(frame.size(), CV_32S);
                for (size_t i = 0; i < contours.size(); i++) {
                    cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
                }

                cv::Mat watershed_src = frame.clone();
                cv::watershed(watershed_src, markers);

                result = cv::Mat::zeros(frame.size(), CV_8UC3);
                for (int i = 0; i < markers.rows; i++) {
                    for (int j = 0; j < markers.cols; j++) {
                        int index = markers.at<int>(i, j);
                        if (index == -1) {
                            result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                        } else if (index <= 0) {
                            result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                        } else {
                            result.at<cv::Vec3b>(i, j) = cv::Vec3b((index * 35) % 255, (index * 85) % 255, (index * 135) % 255);
                        }
                    }
                }
                break;
            }

            case MEAN_SHIFT_FILTER: {
                mode_text = "Method: Mean Shift Filtering";
                int sp = 15;
                int sr = 30;
                cv::pyrMeanShiftFiltering(frame, result, sp, sr, 2);
                break;
            }

            case FLOOD_FILL_MODE: {
                mode_text = "Method: Flood Fill (Center Seed)";
                result = frame.clone();
                cv::Point seed(frame.cols / 2, frame.rows / 2);
                cv::Scalar newVal(0, 255, 255);
                cv::Scalar loDiff(20, 20, 20);
                cv::Scalar upDiff(20, 20, 20);
                cv::floodFill(result, seed, newVal, nullptr, loDiff, upDiff, 4);
                cv::drawMarker(result, seed, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 15, 2);
                break;
            }

            case GRAB_CUT_MODE: {
                mode_text = "Method: GrabCut (Heavy on CPU!)";
                cv::Mat small_frame;
                cv::resize(frame, small_frame, cv::Size(), 0.5, 0.5);

                cv::Mat bgdModel, fgdModel, mask;
                cv::Rect rect(small_frame.cols / 4, small_frame.rows / 4, small_frame.cols / 2, small_frame.rows / 2);

                cv::grabCut(small_frame, mask, rect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT);
                cv::Mat bin_mask = (mask == cv::GC_PR_FGD) | (mask == cv::GC_FGD);

                cv::Mat small_result = cv::Mat::zeros(small_frame.size(), small_frame.type());
                small_frame.copyTo(small_result, bin_mask);
                cv::resize(small_result, result, frame.size());

                cv::Rect large_rect(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2);
                cv::rectangle(frame, large_rect, cv::Scalar(255, 0, 0), 2);
                break;
            }

            case KMEANS_MODE: {
                mode_text = "Method: K-Means Clustering";

                // Downscale for performance optimization (K-means on full resolution can be laggy)
                cv::Mat small_frame;
                cv::resize(frame, small_frame, cv::Size(), 0.5, 0.5);

                // Step A: Reshape the image to a 1D list of pixels: Nx3 (channels)
                cv::Mat samples = small_frame.reshape(1, small_frame.total());
                samples.convertTo(samples, CV_32F); // Convert to float values

                // Step B: Set K (number of clusters/dominant colors)
                int K = 4;
                cv::Mat labels, centers;

                // Define criteria: Stop after 10 iterations or if accuracy epsilon reaches 1.0
                cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);

                // Step C: Run K-Means algorithm
                cv::kmeans(samples, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

                // Step D: Reconstruct the segmented image using cluster centers
                cv::Mat small_result(small_frame.size(), small_frame.type());
                for (int i = 0; i < small_frame.total(); i++) {
                    int cluster_idx = labels.at<int>(i);
                    // Dynamically map 1D array index back to 2D image coordinates
                    int r = i / small_frame.cols;
                    int c = i % small_frame.cols;

                    // Assign the cluster center color to the pixel
                    small_result.at<cv::Vec3b>(r, c) = centers.at<cv::Vec3f>(cluster_idx);
                }

                // Upscale the result back to original window size
                cv::resize(small_result, result, frame.size());
                break;
            }

            default:
                break;
        }

        // 3. Overlay the current method name onto the original frame
        cv::putText(frame, mode_text, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 4. Display the windows
        cv::imshow("Original Video Feed", frame);
        cv::imshow("Segmentation Result", result);

        // 5. Key event handling
        int key = cv::waitKey(30);
        if (key == 27) { // ESC key to exit
            break;
        }
        else if (key == 32) { // SPACE key to cycle through methods
            current_mode = static_cast<SegmentationMode>((current_mode + 1) % MODE_COUNT);
            cv::destroyWindow("Segmentation Result"); // Reset window geometry
            std::cout << "Switched to mode: " << current_mode << std::endl;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}
