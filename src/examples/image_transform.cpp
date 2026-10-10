#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>

int main() {
    // Load the input image
    cv::Mat src = cv::imread("data/test.png");

    // Check if the image is loaded successfully
    if (src.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Create destinations for the transformed images
    cv::Mat dst_warp, dst_warp_rotate, dst_perspective;

    // =========================================================================
    // METHOD 1: Affine Transformation using 3 corresponding points
    // =========================================================================

    // Define arrays of 3 points for source and destination
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    // Define 3 points in the source image
    srcTri[0] = cv::Point2f(0.0f, 0.0f);
    srcTri[1] = cv::Point2f(src.cols - 1.0f, 0.0f);
    srcTri[2] = cv::Point2f(0.0f, src.rows - 1.0f);

    // Define where those 3 points should map to in the destination image
    dstTri[0] = cv::Point2f(src.cols * 0.0f, src.rows * 0.33f);
    dstTri[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = cv::Point2f(src.cols * 0.15f, src.rows * 0.70f);

    // Get the 2x3 Affine Transformation Matrix
    cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transformation to the source image
    cv::warpAffine(src, dst_warp, warp_mat, src.size());

    // =========================================================================
    // METHOD 2: Affine Transformation using Rotation and Scaling
    // =========================================================================

    // Define the center of rotation (middle of the warped image)
    cv::Point center = cv::Point(dst_warp.cols / 2, dst_warp.rows / 2);

    // Define rotation angle in degrees (positive means counter-clockwise)
    double angle = -50.0;

    // Define scale factor (0.6 means shrinking the image to 60% of its size)
    double scale = 0.6;

    // Get the 2x3 rotation matrix
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);

    // Apply the rotation transformation to the warped image
    cv::warpAffine(dst_warp, dst_warp_rotate, rot_mat, dst_warp.size());

    // =========================================================================
    // METHOD 3: Perspective Transformation using 4 corresponding points
    // =========================================================================

    // Define arrays of 4 points for source and destination
    cv::Point2f srcQuad[4];
    cv::Point2f dstQuad[4];

    // Define 4 corners of the source image (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    srcQuad[0] = cv::Point2f(0.0f, 0.0f);
    srcQuad[1] = cv::Point2f(src.cols - 1.0f, 0.0f);
    srcQuad[2] = cv::Point2f(src.cols - 1.0f, src.rows - 1.0f);
    srcQuad[3] = cv::Point2f(0.0f, src.rows - 1.0f);

    // Define where those 4 corners should map to (creating a trapezoid/perspective effect)
    dstQuad[0] = cv::Point2f(src.cols * 0.15f, src.rows * 0.20f);
    dstQuad[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.10f);
    dstQuad[2] = cv::Point2f(src.cols * 0.70f, src.rows * 0.90f);
    dstQuad[3] = cv::Point2f(src.cols * 0.25f, src.rows * 0.80f);

    // Get the 3x3 Perspective Transformation Matrix
    cv::Mat perspective_mat = cv::getPerspectiveTransform(srcQuad, dstQuad);

    // Apply the Perspective Transformation to the source image
    cv::warpPerspective(src, dst_perspective, perspective_mat, src.size());

    // =========================================================================
    // Display results
    // =========================================================================
    cv::imshow("Source Image", src);
    cv::imshow("Warped Image (Affine)", dst_warp);
    cv::imshow("Warped and Rotated Image (Affine)", dst_warp_rotate);
    cv::imshow("Perspective Transformed Image", dst_perspective);

    // Wait for a key press to close windows
    cv::waitKey(0);

    return 0;
}
