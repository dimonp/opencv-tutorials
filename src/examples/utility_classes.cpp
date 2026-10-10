#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{

    {
        cv::Point p1;
        cv::Point p2(10, 20);
        cv::Point p3 = {30, 40};

        std::cout << "p2.x = " << p2.x << ", p2.y = " << p2.y << std::endl;
    }

    {
        cv::Point2d p4;
        cv::Point2d p5(10.0, 20.0);
        cv::Point2d p6 = {30.0, 40.0};

        std::cout << "p5.x = " << p5.x << ", p5.y = " << p5.y << std::endl;
    }

    {
        cv::Size s1;
        cv::Size s2(640, 480);
        cv::Size s3 = {800, 600};

        std::cout << "Width: " << s2.width << ", Height: " << s2.height << std::endl;
        std::cout << "Area: " << s2.area() << " pixels" << std::endl; // 307200

        if (s1.empty()) {
            std::cout << "Size is empty" << std::endl;
        }
    }

    {
        cv::Rect r1;
        cv::Rect r2(10, 20, 30, 40);
        cv::Rect r3 = {50, 60, 70, 80};

        std::cout << "x: " << r2.x << ", y: " << r2.y << ", width: " << r2.width << ", height: " << r2.height << std::endl;
    }

    {
        cv::RotatedRect r4;
        cv::RotatedRect r5(cv::Point2f(10, 20), cv::Size2f(30, 40), 50);
        cv::RotatedRect r6 = {cv::Point2f(50, 60), cv::Size2f(70, 80), 90};

        std::cout << "x: " << r5.center.x << ", y: " << r5.center.y << ", width: " << r5.size.width << ", height: " << r5.size.height << std::endl;
    }

    {
        cv::Scalar s1;
        cv::Scalar s2(10, 20, 30, 40);
        cv::Scalar s3 = {50, 60, 70, 80};

        std::cout << "s2.val[0] = " << s2.val[0] << ", s2.val[1] = " << s2.val[1] << std::endl;
    }

    {
        cv::Range r1;
        cv::Range r2(10, 20);
        cv::Range r3 = {50, 60};

        std::cout << "r2.start = " << r2.start << ", r2.end = " << r2.end << std::endl;
    }

    {
        cv::TermCriteria t1;
        cv::TermCriteria t2(cv::TermCriteria::COUNT, 10, 0.01);
        cv::TermCriteria t3 = {cv::TermCriteria::EPS, 20, 0.001};

        std::cout << "Type: " << t2.type << ", Max Count: " << t2.maxCount << ", Epsilon: " << t2.epsilon << std::endl;
    }

    {
        cv::Vec3d v1;
        cv::Vec3d v2(10, 20, 30);
        cv::Vec3d v3 = {40, 50, 60};

        std::cout << "v2[0] = " << v2[0] << ", v2[1] = " << v2[1] << std::endl;
    }

    {
        cv::Mat m1;
        cv::Mat m2(480, 640, CV_8UC3, cv::Scalar(255, 0, 0));
        std::cout << "Rows: " << m2.rows << ", Cols: " << m2.cols << std::endl;
        cv::transpose(m2, m1);
        std::cout << "Rows: " << m1.rows << ", Cols: " << m1.cols << std::endl;


        cv::Mat m3 = cv::Mat::zeros(3, 3, CV_32FC1); // Matrix of zeros 3x3
        std::cout << "m3 = " << m3 << std::endl;

        cv::Mat m4 = cv::Mat::eye(3, 3, CV_32FC1);   // Identity matrix 3x3
        std::cout << "m4 = " << m4 << std::endl;

        cv::Mat m5 = m2; // Shallow copying
        cv::Mat m6 = m2.clone(); // Deep copying
    }

    {
        cv::Matx33d m1;
        cv::Matx33d m2(1, 2, 3, 4, 5, 6, 7, 8, 9);
        cv::Matx33d m3 = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        std::cout << "m2(0, 1) = " << m2(0, 1) << std::endl;
    }

}
