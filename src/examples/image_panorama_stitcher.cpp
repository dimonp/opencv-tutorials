#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

int main() {
    // 1. Load two overlapping images
    cv::Mat img1 = cv::imread("data/photo_l.jpg");
    cv::Mat img2 = cv::imread("data/photo_r.jpg");

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return -1;
    }

    // Pack images into a vector
    std::vector<cv::Mat> imgs;
    imgs.push_back(img1);
    imgs.push_back(img2);

    // 2. Create a Stitcher object in panorama mode
    // (Modes: PANORAMA for standard scenes, SCANS for flat documents/maps)
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    // 3. Perform the stitching process
    cv::Mat panorama;
    cv::Stitcher::Status status = stitcher->stitch(imgs, panorama);

    // 4. Check the result status
    if (status != cv::Stitcher::OK) {
        std::cerr << "Stitching failed! Error code: " << int(status) << std::endl;
        // Code 1 = ERR_NEED_MORE_IMGS (not enough matching keypoints found)
        return -1;
    }

    std::cout << "Stitching completed successfully!" << std::endl;

    // 5. Save and display the result
    cv::imwrite("panorama_result.jpg", panorama);

    cv::imshow("Result Panorama", panorama);
    cv::waitKey(0);

    return 0;
}
