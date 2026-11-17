#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>   // Needed for cvtColor, warpAffine, addWeighted, fillConvexPoly
#include <opencv2/geometry.hpp>  // OpenCV 5.0+ module: Needed for Subdiv2D
#include <opencv2/objdetect.hpp> // Needed for CascadeClassifier (Face detection)
#include <opencv2/face.hpp>      // Needed for FacemarkLBF (Landmarks detection)
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

// Warps a single triangle from the source image to the destination image safely
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2) {
    // 1. Calculate the initial bounding rectangles for both triangles
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // 2. Define image boundaries to prevent out-of-range memory access
    Rect img1Rect(0, 0, img1.cols, img1.rows);
    Rect img2Rect(0, 0, img2.cols, img2.rows);

    // 3. CRITICAL PROTECTION: Crop the rectangles using intersection operator '&'
    // This guarantees the ROI will never exceed the physical size of the image matrices
    r1 = r1 & img1Rect;
    r2 = r2 & img2Rect;

    // Safety check in case the triangle area is completely collapsed or outside
    if (r1.width <= 0 || r1.height <= 0 || r2.width <= 0 || r2.height <= 0) {
        return;
    }

    // Offset vertices based on the top-left corner of their updated cropped bounding boxes
    vector<Point2f> t1Cropped, t2Cropped;
    vector<Point> t2CroppedInt;
    for(int i = 0; i < 3; i++) {
        t1Cropped.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
        t2Cropped.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
        t2CroppedInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y));
    }

    // Crop the source image segment using the safe bounding box
    Mat img1Cropped;
    img1(r1).copyTo(img1Cropped);

    // Compute the Affine Transform matrix for the cropped coordinates
    Mat warpMat = getAffineTransform(t1Cropped, t2Cropped);

    // Apply the Affine Transform to the cropped source image segment
    Mat img2Cropped = Mat::zeros(r2.height, r2.width, img1Cropped.type());
    warpAffine(img1Cropped, img2Cropped, warpMat, img2Cropped.size(), INTER_LINEAR, BORDER_REFLECT_101);

    // Create a precise binary mask for the specific triangle area
    Mat mask = Mat::zeros(r2.height, r2.width, CV_8UC3);
    fillConvexPoly(mask, t2CroppedInt, Scalar(1.0, 1.0, 1.0), LINE_AA, 0);

    // Copy the warped pixels onto the target image grid using the mask
    multiply(img2Cropped, mask, img2Cropped);
    multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Cropped;
}

// Automatically detects 68 facial landmarks and adds full image boundary points
// bool getFaceLandmarks(const Mat &img, vector<Point2f> &landmarks, CascadeClassifier &faceDetector, Ptr<Facemark> &facemark) {
//     Mat gray;
//     cvtColor(img, gray, COLOR_BGR2GRAY); // CascadeClassifier operates on grayscale images

//     // Detect face bounding rectangles within the image
//     vector<Rect> faces;
//     faceDetector.detectMultiScale(gray, faces);

//     if (faces.empty()) return false; // Return false if no faces are visible

//     // Extract facial feature shapes
//     vector<vector<Point2f>> detectedShapes;
//     if (facemark->fit(img, faces, detectedShapes)) {
//         // Double check to ensure the vector contains valid points for the first face
//         if (detectedShapes.empty() || detectedShapes[0].empty()) {
//             return false;
//         }

//         // Correct Type Mapping: Extract 1D point vector of the first face found
//         landmarks = detectedShapes[0];

//         Size size = img.size();
//         // Add image frame corner coordinates to anchor the background structure during morphing
//         landmarks.push_back(Point2f(0.f, 0.f));
//         landmarks.push_back(Point2f((float)size.width, 0.f));
//         landmarks.push_back(Point2f(0.f, (float)size.height));
//         landmarks.push_back(Point2f((float)size.width, (float)size.height));
//         return true;
//     }
//     return false;
// }
bool getFaceLandmarks(const Mat &img, vector<Point2f> &landmarks, CascadeClassifier &faceDetector, Ptr<Facemark> &facemark) {
    if (img.empty()) return false;

    landmarks.clear();

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    vector<Rect> faces;
    faceDetector.detectMultiScale(gray, faces, 1.1, 3, 0, Size(50, 50));

    if (faces.empty()) {
        cout << "[FAIL] No face found by CascadeClassifier" << endl;
        return false;
    }

    // КРИТИЧЕСКИЙ ШАГ ДЛЯ OPENCV:
    // Создаем вектор для 1 лица и ПРИНУДИТЕЛЬНО выделяем в нем память под 68 точек.
    // Без этого facemark->fit() в некоторых версиях OpenCV вернет 0 точек!
    vector<vector<Point2f>> detectedShapes(1);
    detectedShapes[0].resize(68);

    if (facemark->fit(img, faces, detectedShapes)) {
        // Проверяем, записал ли детектор туда реальные данные
        if (detectedShapes.empty() || detectedShapes[0].empty()) {
            cout << "[FAIL] Landmark detector returned empty shape vectors" << endl;
            return false;
        }

        cout << "[SUCCESS] Extracted " << detectedShapes[0].size() << " facial points." << endl;

        // Копируем 68 точек лица
        for (size_t i = 0; i < detectedShapes[0].size(); i++) {
            landmarks.push_back(detectedShapes[0][i]);
        }

        // Добавляем 4 угловые точки кадра в самый конец
        Size size = img.size();
        landmarks.push_back(Point2f(0.f, 0.f));
        landmarks.push_back(Point2f((float)size.width, 0.f));
        landmarks.push_back(Point2f(0.f, (float)size.height));
        landmarks.push_back(Point2f((float)size.width, (float)size.height));

        cout << "[TOTAL] Total points vector size: " << landmarks.size() << " (68 face + 4 corners)" << endl;
        return true;
    }
    return false;
}

// Generates a single morphed frame based on the dynamic alpha interpolation value
Mat generateMorphFrame(const Mat &img1, const Mat &img2, const vector<Point2f> &points1, const vector<Point2f> &points2, float alpha) {
    Size size = img1.size();

    // 1. Calculate dynamic intermediate keypoint coordinates using safety clamping
    vector<Point2f> pointsMorph;
    for(size_t i = 0; i < points1.size(); i++) {
        float x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
        float y = (1 - alpha) * points1[i].y + alpha * points2[i].y;

        // Clamp boundaries to keep points strictly within structural limitations
        x = std::max(0.f, std::min(x, size.width - 1.f));
        y = std::max(0.f, std::min(y, size.height - 1.f));

        pointsMorph.push_back(Point2f(x, y));
    }

    // 2. Setup boundary boxes for verification and extended Subdiv2D tracking limits
    Rect imageRect(0, 0, size.width, size.height);
    Rect subdivRect(-10, -10, size.width + 20, size.height + 20); // 10px margin to avoid out-of-range crashes

    // Initialize Delaunay Triangulation using OpenCV 5 architecture
    Subdiv2D subdiv(subdivRect);
    for(auto& p : pointsMorph) {
        subdiv.insert(p);
    }

    // Retrieve generated triangulation layouts
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    Mat imgMorph1 = Mat::zeros(size, img1.type());
    Mat imgMorph2 = Mat::zeros(size, img2.type());

    // 3. Process every individual triangle layout
    for(size_t i = 0; i < triangleList.size(); i++) {
        Vec6f t = triangleList[i];

        // Correct Type Mapping: Extract float values from Vec6f sequence to construct Point2f objects
        vector<Point2f> tMorph = {
            Point2f(t[0], t[1]),
            Point2f(t[2], t[3]),
            Point2f(t[4], t[5])
        };

        // Correct API Call: Perform point-by-point validation inside the image area instead of passing vector
        if (!imageRect.contains(tMorph[0]) ||
            !imageRect.contains(tMorph[1]) ||
            !imageRect.contains(tMorph[2])) {
            continue;
        }

        // Trace Delaunay points back to their corresponding indexes in source vectors
        vector<Point2f> t1(3), t2(3);
        for(int j = 0; j < 3; j++) {
            for(size_t k = 0; k < pointsMorph.size(); k++) {
                if(abs(tMorph[j].x - pointsMorph[k].x) < 1.0 && abs(tMorph[j].y - pointsMorph[k].y) < 1.0) {
                    t1[j] = points1[k];
                    t2[j] = points2[k];
                    break;
                }
            }
        }

        // Warp triangles from original structures to fit the intermediate mesh layout
        warpTriangle(const_cast<Mat&>(img1), imgMorph1, t1, tMorph);
        warpTriangle(const_cast<Mat&>(img2), imgMorph2, t2, tMorph);
    }

    // 4. Blend the two final warped texture configurations pixel by pixel
    Mat imgMorph = Mat::zeros(size, img1.type());
    addWeighted(imgMorph1, 1.0 - alpha, imgMorph2, alpha, 0, imgMorph);

    return imgMorph;
}

int main() {
    // Load input target images (ensure both are identical sizes)
    Mat img1 = imread("data/face1.jpg");
    Mat img2 = imread("data/face2.jpg");
    if(img1.empty() || img2.empty()) {
        cout << "Error: Could not load image files." << endl;
        return -1;
    }

    // Initialize detection configurations
    CascadeClassifier faceDetector("data/haarcascade_frontalface_alt2.xml");

    Ptr<Facemark> facemark = FacemarkKazemi::create();
    facemark->loadModel("data/face_landmark_model.dat");

    // Ptr<Facemark> facemark = FacemarkLBF::create();
    // facemark->loadModel("data/lbfmodel.yaml");

    // Extract facial landmarks once before initiating presentation sequence
    vector<Point2f> points1, points2;
    if (!getFaceLandmarks(img1, points1, faceDetector, facemark) ||
        !getFaceLandmarks(img2, points2, faceDetector, facemark)) {
        cout << "Error: Face features or landmark detection failed." << endl;
        return -1;
    }

// --- VISUALIZATION OF DETECTED LANDMARKS ---
// Create copies of the original images to draw on (so we don't mess up the source textures)
Mat img1Contours = img1.clone();
Mat img2Contours = img2.clone();

// Draw the 68 face landmarks as small solid circles directly on the faces
// We loop up to 'points1.size() - 4' to skip the 4 corner anchor points
for (size_t i = 0; i < points1.size() - 4; i++) {
    // Green dots for Image 1, thickness = -1 means filled circle, radius = 3
    circle(img1Contours, points1[i], 3, Scalar(0, 255, 0), -1);

    // Red dots for Image 2
    circle(img2Contours, points2[i], 3, Scalar(0, 0, 255), -1);
}

// Draw the 4 image corner points using blue color just for verification
for (size_t i = points1.size() - 4; i < points1.size(); i++) {
    circle(img1Contours, points1[i], 8, Scalar(255, 0, 0), -1);
    circle(img2Contours, points2[i], 8, Scalar(255, 0, 0), -1);
}

// Display the detected landmarks windows
namedWindow("Detected Landmarks - Image 1", WINDOW_AUTOSIZE);
namedWindow("Detected Landmarks - Image 2", WINDOW_AUTOSIZE);
imshow("Detected Landmarks - Image 1", img1Contours);
imshow("Detected Landmarks - Image 2", img2Contours);

cout << "Landmarks windows opened. Press ANY KEY in one of the windows to start the morphing animation." << endl;
waitKey(0); // Pause execution until user presses any key

// Destroy the static visualization windows to clear the screen for animation
destroyWindow("Detected Landmarks - Image 1");
destroyWindow("Detected Landmarks - Image 2");
// -------------------------------------------

    string windowName = "OpenCV 5 Face Morphing Animation";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    int totalFrames = 60; // Frame iterations per transition direction
    cout << "Starting OpenCV 5 animation loop. Press 'ESC' to exit." << endl;

    // Endless visual demonstration loop
    while(true) {
        // Forward Phase Transition Sequence: Face 1 -> Face 2
        for (int i = 0; i <= totalFrames; i++) {
            float alpha = (float)i / totalFrames;
            Mat frame = generateMorphFrame(img1, img2, points1, points2, alpha);

            imshow(windowName, frame);
            if (waitKey(30) == 27) return 0; // 30ms window delay (~33 FPS). Exit code 27 equals ESC key
        }

        // Stay on target Face 2 destination for 1000 milliseconds
        if (waitKey(1000) == 27) return 0;

        // Backward Phase Transition Sequence: Face 2 -> Face 1
        for (int i = totalFrames; i >= 0; i--) {
            float alpha = (float)i / totalFrames;
            Mat frame = generateMorphFrame(img1, img2, points1, points2, alpha);

            imshow(windowName, frame);
            if (waitKey(30) == 27) return 0;
        }

        // Stay on source Face 1 destination for 1000 milliseconds
        if (waitKey(1000) == 27) return 0;
    }

    return 0;
}
