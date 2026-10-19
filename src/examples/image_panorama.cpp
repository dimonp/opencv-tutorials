#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/geometry/3d.hpp>

int main() {
    // 1. Load the two images (Left and Right)
    cv::Mat imgL = cv::imread("data/photo0_l.jpg");
    cv::Mat imgR = cv::imread("data/photo0_r.jpg");

    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return -1;
    }

    // 2. Convert images to grayscale for feature detection
    cv::Mat grayL, grayR;
    cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);

    // 3. Detect keypoints and compute descriptors using ORB
    cv::Ptr<cv::ORB> detector = cv::ORB::create(2000); // Find up to 2000 features
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;

    detector->detectAndCompute(grayL, cv::noArray(), keypointsL, descriptorsL);
    detector->detectAndCompute(grayR, cv::noArray(), keypointsR, descriptorsR);

    // 4. Match descriptors using Brute-Force Matcher with Hamming distance
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    // Use k=2 for Lowe's ratio test
    matcher.knnMatch(descriptorsR, descriptorsL, knnMatches, 2);

    // 5. Filter matches using Lowe's ratio test to filter out false matches
    std::vector<cv::DMatch> goodMatches;
    const float ratioThresh = 0.75f;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() >= 2 && knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    std::cout << "Number of good matches found: " << goodMatches.size() << std::endl;

    // Optional: Visual check. Draw and show the filtered matches.
    cv::Mat imgMatches;
    cv::drawMatches(imgR, keypointsR, imgL, keypointsL, goodMatches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Good Matches (Right to Left)", imgMatches);

    // We need at least 4 matches to calculate the Homography matrix
    if (goodMatches.size() < 4) {
        std::cerr << "Error: Not enough matches to compute homography (needed at least 4)." << std::endl;
        cv::waitKey(0);
        return -1;
    }

    // 6. Extract coordinates of matched keypoints
    std::vector<cv::Point2f> ptsR, ptsL;
    for (const auto& match : goodMatches) {
        ptsR.push_back(keypointsR[match.queryIdx].pt);
        ptsL.push_back(keypointsL[match.trainIdx].pt);
    }

    // 7. Find the Homography matrix using RANSAC
    // This matrix transforms coordinates from the Right image space to the Left image space
    cv::Mat H = cv::findHomography(ptsR, ptsL, cv::RANSAC);

    if (H.empty()) {
        std::cerr << "Error: Homography estimation failed." << std::endl;
        cv::waitKey(0);
        return -1;
    }

    // 8. Warp the Right image to match the perspective of the Left image
    // Create a large canvas to fit both images (width of Left + width of Right)
    cv::Mat result;
    cv::Size panoramaSize(imgL.cols + imgR.cols, imgL.rows);
    cv::warpPerspective(imgR, result, H, panoramaSize);

    // 9. Overlay the Left image onto the warped canvas
    // Copy the Left image directly into the top-left region of the result canvas
    cv::Mat half(result, cv::Rect(0, 0, imgL.cols, imgL.rows));
    imgL.copyTo(half);

    // 10. Display and save the manual panorama result
    cv::imshow("Manual Panorama Result", result);
    cv::imwrite("manual_panorama.jpg", result);
    cv::waitKey(0);

    return 0;
}
