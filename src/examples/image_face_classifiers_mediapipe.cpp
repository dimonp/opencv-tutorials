#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/geometry.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ============================================================
// CONFIGURATION
// ============================================================

static const string DETECTOR_MODEL =
    "data/face_detection_short_range.onnx";

static const string LANDMARK_MODEL_192 =
    "data/face_mesh_Nx3x192x192.onnx";

static const string LANDMARK_MODEL_256 =
    "data/face_landmarker_Nx3x256x256.onnx";

static const int DETECTOR_SIZE = 128;

static const int DEFAULT_LANDMARK_SIZE = 192;

static const int NUM_ANCHORS = 896;
static const int NUM_FACE_KEYPOINTS = 6;

static const int MIN_LANDMARKS = 468;
static const int MAX_LANDMARKS = 478;

static const float DETECTION_THRESHOLD = 0.65f;
static const float PRESENCE_THRESHOLD = 0.50f;
static const float NMS_THRESHOLD = 0.30f;

// ============================================================
// DATA STRUCTURES
// ============================================================

struct FaceDetection
{
    Rect2f box;
    array<Point2f, 6> keypoints;
    float score = 0.0f;
};

struct FaceROI
{
    Point2f center;
    float side;
    float angle;
};

struct FaceLandmarks
{
    vector<Point3f> points;
    float presence = 0.0f;
};

// ============================================================
// FACIAL LANDMARK INDICES
// ============================================================

const vector<int> FACE_OVAL = {
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
};

const vector<int> LEFT_EYE = {
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
};

const vector<int> RIGHT_EYE = {
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
};

const vector<int> OUTER_LIPS = {
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87,
    178, 88, 95, 78
};

const vector<int> INNER_LIPS = {
    78, 191, 80, 81, 82, 13,
    312, 311, 310, 415, 308,
    324, 318, 402, 317, 14,
    87, 178, 88, 95
};

const vector<int> NOSE = {
    168, 6, 197, 195, 5, 4,
    1, 19, 94, 2, 98, 97,
    326, 327, 294, 278, 344
};

const vector<int> LEFT_IRIS = {
    468, 469, 470, 471, 472
};

const vector<int> RIGHT_IRIS = {
    473, 474, 475, 476, 477
};

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

float sigmoid(float x)
{
    x = max(-80.0f, min(80.0f, x));
    return 1.0f / (1.0f + exp(-x));
}

Point2f transformPoint(
    const Mat& matrix,
    const Point2f& point)
{
    double x =
        matrix.at<double>(0, 0) * point.x +
        matrix.at<double>(0, 1) * point.y +
        matrix.at<double>(0, 2);

    double y =
        matrix.at<double>(1, 0) * point.x +
        matrix.at<double>(1, 1) * point.y +
        matrix.at<double>(1, 2);

    return Point2f(
        static_cast<float>(x),
        static_cast<float>(y)
    );
}

Mat createRGBBlob(
    const Mat& image,
    float scale,
    const Size& size,
    const Scalar& mean)
{
    return blobFromImage(
        image,
        scale,
        size,
        mean,
        true,
        false,
        CV_32F
    );
}

// ============================================================
// BLAZEFACE ANCHOR GENERATION
// ============================================================

vector<Point2f> generateAnchors()
{
    vector<Point2f> anchors;

    vector<int> strides = { 8, 16, 16, 16 };

    int index = 0;

    while (index < static_cast<int>(strides.size()))
    {
        int last = index;

        while (
            last < static_cast<int>(strides.size()) &&
            strides[last] == strides[index])
        {
            last++;
        }

        int repeats = 2 * (last - index);
        int cells = DETECTOR_SIZE / strides[index];

        for (int y = 0; y < cells; y++)
        {
            for (int x = 0; x < cells; x++)
            {
                Point2f anchor(
                    (x + 0.5f) / cells,
                    (y + 0.5f) / cells
                );

                for (int k = 0; k < repeats; k++)
                    anchors.push_back(anchor);
            }
        }

        index = last;
    }

    if (anchors.size() != NUM_ANCHORS)
        throw runtime_error("Incorrect BlazeFace anchor count!");

    return anchors;
}

// ============================================================
// WEIGHTED NON-MAXIMUM SUPPRESSION
// ============================================================

float calculateIoU(
    const Rect2f& a,
    const Rect2f& b)
{
    float x1 = max(a.x, b.x);
    float y1 = max(a.y, b.y);

    float x2 = min(a.x + a.width, b.x + b.width);
    float y2 = min(a.y + a.height, b.y + b.height);

    float intersection =
        max(0.0f, x2 - x1) *
        max(0.0f, y2 - y1);

    float unionArea =
        a.area() + b.area() - intersection;

    if (unionArea <= 0.0f)
        return 0.0f;

    return intersection / unionArea;
}

vector<FaceDetection> weightedNMS(
    vector<FaceDetection> detections)
{
    sort(
        detections.begin(),
        detections.end(),
        [](const FaceDetection& a, const FaceDetection& b)
        {
            return a.score > b.score;
        }
    );

    vector<FaceDetection> result;

    while (!detections.empty())
    {
        FaceDetection top = detections.front();

        float sumWeights = 0.0f;

        float sumX = 0.0f;
        float sumY = 0.0f;
        float sumW = 0.0f;
        float sumH = 0.0f;

        array<Point2f, 6> weightedKeypoints = {};

        vector<FaceDetection> remaining;

        for (const auto& detection : detections)
        {
            if (calculateIoU(top.box, detection.box) >
                NMS_THRESHOLD)
            {
                float weight = detection.score;

                sumWeights += weight;

                sumX += detection.box.x * weight;
                sumY += detection.box.y * weight;
                sumW += detection.box.width * weight;
                sumH += detection.box.height * weight;

                for (int k = 0; k < 6; k++)
                {
                    weightedKeypoints[k] +=
                        detection.keypoints[k] * weight;
                }
            }
            else
            {
                remaining.push_back(detection);
            }
        }

        if (sumWeights > 0.0f)
        {
            top.box = Rect2f(
                sumX / sumWeights,
                sumY / sumWeights,
                sumW / sumWeights,
                sumH / sumWeights
            );

            for (int k = 0; k < 6; k++)
            {
                top.keypoints[k] =
                    weightedKeypoints[k] * (1.0f / sumWeights);
            }
        }

        result.push_back(top);
        detections.swap(remaining);
    }

    return result;
}

// ============================================================
// BLAZEFACE DETECTOR
// ============================================================

class BlazeFaceDetector
{
private:

    Net net;
    vector<Point2f> anchors;

public:

    BlazeFaceDetector(const string& modelPath)
    {
        net = readNetFromONNX(modelPath);

        if (net.empty())
            throw runtime_error("Cannot load BlazeFace ONNX model!");

        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        anchors = generateAnchors();
    }

    vector<FaceDetection> detect(
        const Mat& frame,
        float threshold = DETECTION_THRESHOLD)
    {
        int imageWidth = frame.cols;
        int imageHeight = frame.rows;

        float scale = DETECTOR_SIZE /
            static_cast<float>(max(imageWidth, imageHeight));

        float padX =
            (DETECTOR_SIZE - imageWidth * scale) / 2.0f;

        float padY =
            (DETECTOR_SIZE - imageHeight * scale) / 2.0f;

        Mat letterboxMatrix = (
            Mat_<double>(2, 3) <<
            scale, 0.0, padX,
            0.0, scale, padY
        );

        Mat detectorImage;

        warpAffine(
            frame,
            detectorImage,
            letterboxMatrix,
            Size(DETECTOR_SIZE, DETECTOR_SIZE),
            INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar(0, 0, 0)
        );

        Mat blob = createRGBBlob(
            detectorImage,
            1.0 / 127.5,
            Size(DETECTOR_SIZE, DETECTOR_SIZE),
            Scalar(127.5, 127.5, 127.5)
        );

        net.setInput(blob);

        vector<Mat> outputs;

        net.forward(
            outputs,
            net.getUnconnectedOutLayersNames()
        );

        if (outputs.size() < 2)
            throw runtime_error("Unexpected BlazeFace outputs!");

        Mat regressors;
        Mat logits;

        outputs[0].convertTo(regressors, CV_32F);
        outputs[1].convertTo(logits, CV_32F);

        Mat reg = regressors.reshape(1, NUM_ANCHORS);
        Mat conf = logits.reshape(1, NUM_ANCHORS);

        vector<FaceDetection> candidates;

        for (int i = 0; i < NUM_ANCHORS; i++)
        {
            float score = sigmoid(conf.at<float>(i, 0));

            if (score < threshold)
                continue;

            const Point2f& anchor = anchors[i];
            const float* values = reg.ptr<float>(i);

            float cx = values[0] / DETECTOR_SIZE + anchor.x;
            float cy = values[1] / DETECTOR_SIZE + anchor.y;

            float width = values[2] / DETECTOR_SIZE;
            float height = values[3] / DETECTOR_SIZE;

            auto toImage = [&](float x, float y) -> Point2f
            {
                return Point2f(
                    (x * DETECTOR_SIZE - padX) / scale,
                    (y * DETECTOR_SIZE - padY) / scale
                );
            };

            Point2f center = toImage(cx, cy);

            FaceDetection detection;

            detection.box = Rect2f(
                center.x - width * DETECTOR_SIZE / scale / 2.0f,
                center.y - height * DETECTOR_SIZE / scale / 2.0f,
                width * DETECTOR_SIZE / scale,
                height * DETECTOR_SIZE / scale
            );

            for (int k = 0; k < NUM_FACE_KEYPOINTS; k++)
            {
                float kpX =
                    values[4 + k * 2] / DETECTOR_SIZE + anchor.x;

                float kpY =
                    values[5 + k * 2] / DETECTOR_SIZE + anchor.y;

                detection.keypoints[k] = toImage(kpX, kpY);
            }

            detection.score = score;
            candidates.push_back(detection);
        }

        return weightedNMS(candidates);
    }
};

// ============================================================
// FACE ROI ALIGNMENT
// ============================================================

FaceROI createFaceROI(const FaceDetection& detection)
{
    float side = max(
        detection.box.width,
        detection.box.height
    ) * 1.5f;

    Point2f center(
        detection.box.x + detection.box.width / 2.0f,
        detection.box.y + detection.box.height / 2.0f
    );

    Point2f rightEye = detection.keypoints[0];
    Point2f leftEye = detection.keypoints[1];

    Point2f direction = leftEye - rightEye;

    float angle = atan2(
        direction.y,
        direction.x
    ) * 180.0f / static_cast<float>(CV_PI);

    return { center, side, angle };
}

// ============================================================
// FACE LANDMARK MODEL
// ============================================================

class FaceLandmarker
{
private:

    Net net;
    int inputSize;

public:

    FaceLandmarker(
        const string& modelPath,
        int modelInputSize)
        : inputSize(modelInputSize)
    {
        if (inputSize != 192 && inputSize != 256)
            throw runtime_error("Unsupported landmark input size!");

        cout << "Loading landmark model: "
             << modelPath << endl;

        cout << "Input size: "
             << inputSize << "x" << inputSize << endl;

        net = readNetFromONNX(modelPath);

        if (net.empty())
            throw runtime_error("Cannot load Face Landmark ONNX model!");

        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }

    bool predict(
        const Mat& frame,
        const FaceROI& roi,
        FaceLandmarks& result)
    {
        if (roi.side <= 0.0f)
            return false;

        // Create a rotation-and-scale matrix for the selected model size.
        Mat transform = getRotationMatrix2D(
            roi.center,
            roi.angle,
            inputSize / roi.side
        );

        // Move the ROI center to the center of the network input.
        transform.at<double>(0, 2) +=
            inputSize / 2.0 - roi.center.x;

        transform.at<double>(1, 2) +=
            inputSize / 2.0 - roi.center.y;

        Mat faceCrop;

        warpAffine(
            frame,
            faceCrop,
            transform,
            Size(inputSize, inputSize),
            INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar(0, 0, 0)
        );

        // This preprocessing is a starting assumption.
        // Verify RGB order and normalization for each ONNX model.
        Mat blob = createRGBBlob(
            faceCrop,
            1.0 / 255.0,
            Size(inputSize, inputSize),
            Scalar(0, 0, 0)
        );

        net.setInput(blob);

        vector<Mat> outputs;

        vector<String> outputNames =
            net.getUnconnectedOutLayersNames();

        net.forward(outputs, outputNames);

        if (outputs.empty())
            throw runtime_error("Landmark model returned no outputs!");

        // Print output information for model inspection.
        static bool printedOutputInfo = false;

        if (!printedOutputInfo)
        {
            cout << "\nLandmark model outputs:" << endl;

            for (int i = 0; i < static_cast<int>(outputs.size()); i++)
            {
                cout << "Output " << i
                     << ": dimensions = " << outputs[i].dims
                     << ", total = " << outputs[i].total()
                     << ", shape = ";

                for (int d = 0; d < outputs[i].dims; d++)
                    cout << outputs[i].size[d] << " ";

                cout << endl;
            }

            printedOutputInfo = true;
        }

        // Find the landmark tensor.
        // Expected formats:
        // 468 landmarks * XYZ = 1404 values.
        // 478 landmarks * XYZ = 1434 values.

        int landmarkOutputIndex = -1;
        int presenceOutputIndex = -1;

        Mat landmarkOutput;
        Mat presenceOutput;

        for (int i = 0; i < static_cast<int>(outputs.size()); i++)
        {
            Mat converted;
            outputs[i].convertTo(converted, CV_32F);

            size_t total = converted.total();

            if (total == 468 * 3 || total == 478 * 3)
            {
                landmarkOutputIndex = i;
                landmarkOutput = converted;
                break;
            }
        }

        if (landmarkOutputIndex < 0)
        {
            throw runtime_error(
                "Cannot find a 468/478 XYZ landmark tensor. "
                "Check the ONNX output shapes."
            );
        }

        // Find a separate presence output, if one exists.
        for (int i = 0; i < static_cast<int>(outputs.size()); i++)
        {
            if (i == landmarkOutputIndex)
                continue;

            if (outputs[i].total() <= 4)
            {
                presenceOutputIndex = i;
                outputs[i].convertTo(presenceOutput, CV_32F);
                break;
            }
        }

        int landmarkCount =
            static_cast<int>(landmarkOutput.total() / 3);

        // If there is no separate presence tensor, use the detector result.
        result.presence = 1.0f;

        if (presenceOutputIndex >= 0)
        {
            float presenceLogit = presenceOutput.ptr<float>()[0];
            result.presence = sigmoid(presenceLogit);
        }

        if (result.presence < PRESENCE_THRESHOLD)
            return false;

        // Calculate the inverse transformation.
        Mat inverseTransform;
        invertAffineTransform(transform, inverseTransform);

        Mat flat = landmarkOutput.reshape(1, 1);

        result.points.clear();
        result.points.reserve(landmarkCount);

        // Decode all available landmarks.
        for (int i = 0; i < landmarkCount; i++)
        {
            float x = flat.at<float>(0, i * 3);
            float y = flat.at<float>(0, i * 3 + 1);
            float z = flat.at<float>(0, i * 3 + 2);

            // Assumes X and Y are in input-image pixel coordinates.
            Point2f originalPoint = transformPoint(
                inverseTransform,
                Point2f(x, y)
            );

            // Convert relative depth approximately to image scale.
            z *= roi.side / inputSize;

            result.points.emplace_back(
                originalPoint.x,
                originalPoint.y,
                z
            );
        }

        return true;
    }
};

// ============================================================
// LANDMARK DRAWING
// ============================================================

Point landmarkPoint(
    const vector<Point3f>& landmarks,
    int index)
{
    return Point(
        cvRound(landmarks[index].x),
        cvRound(landmarks[index].y)
    );
}

void drawContour(
    Mat& frame,
    const vector<Point3f>& landmarks,
    const vector<int>& indices,
    const Scalar& color,
    const string& label,
    int thickness = 2)
{
    vector<Point> polygon;

    for (int index : indices)
    {
        if (index < 0 ||
            index >= static_cast<int>(landmarks.size()))
        {
            continue;
        }

        polygon.push_back(landmarkPoint(landmarks, index));
    }

    if (polygon.size() < 3)
        return;

    polylines(
        frame,
        polygon,
        true,
        color,
        thickness,
        LINE_AA
    );

    Rect bounds = boundingRect(polygon);

    putText(
        frame,
        label,
        Point(bounds.x, max(15, bounds.y - 7)),
        FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        LINE_AA
    );
}

void drawIris(
    Mat& frame,
    const vector<Point3f>& landmarks,
    const vector<int>& indices,
    const Scalar& color,
    const string& label)
{
    if (landmarks.size() < 478)
        return;

    Point center = landmarkPoint(landmarks, indices[0]);

    float radius = 0.0f;

    for (int i = 1; i < 5; i++)
    {
        Point p = landmarkPoint(landmarks, indices[i]);

        radius += static_cast<float>(norm(p - center));

        circle(frame, p, 2, color, FILLED, LINE_AA);
    }

    radius /= 4.0f;

    circle(
        frame,
        center,
        max(2, cvRound(radius)),
        color,
        2,
        LINE_AA
    );

    circle(frame, center, 3, color, FILLED);

    putText(
        frame,
        label,
        Point(center.x + 7, center.y - 7),
        FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        LINE_AA
    );
}

// ============================================================
// NOSE DRAWING
// ============================================================

void drawNose(
    Mat& frame,
    const vector<Point3f>& landmarks)
{
    drawContour(
        frame,
        landmarks,
        NOSE,
        Scalar(255, 180, 0),
        "Nose",
        2
    );

    if (landmarks.size() > 1)
    {
        circle(
            frame,
            landmarkPoint(landmarks, 1),
            4,
            Scalar(255, 180, 0),
            FILLED,
            LINE_AA
        );
    }
}

// ============================================================
// EAR DRAWING
// ============================================================

void drawEars(
    Mat& frame,
    const FaceDetection& detection)
{
    Point rightEar(
        cvRound(detection.keypoints[4].x),
        cvRound(detection.keypoints[4].y)
    );

    Point leftEar(
        cvRound(detection.keypoints[5].x),
        cvRound(detection.keypoints[5].y)
    );

    Scalar color(0, 165, 255);

    circle(frame, rightEar, 6, color, 2, LINE_AA);
    circle(frame, leftEar, 6, color, 2, LINE_AA);

    putText(
        frame,
        "Ear",
        Point(rightEar.x + 5, rightEar.y - 8),
        FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        LINE_AA
    );

    putText(
        frame,
        "Ear",
        Point(leftEar.x - 25, leftEar.y - 8),
        FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        LINE_AA
    );
}

// ============================================================
// DRAW ALL FACIAL FEATURES
// ============================================================

void drawFace(
    Mat& frame,
    const FaceDetection& detection,
    const FaceLandmarks& face)
{
    const vector<Point3f>& p = face.points;

    if (p.size() < MIN_LANDMARKS)
        return;

    // Draw the face oval.
    drawContour(
        frame, p, FACE_OVAL,
        Scalar(255, 255, 0),
        "Face contour"
    );

    // Draw the eye contours.
    drawContour(
        frame, p, LEFT_EYE,
        Scalar(0, 255, 0),
        "Left eye"
    );

    drawContour(
        frame, p, RIGHT_EYE,
        Scalar(0, 255, 0),
        "Right eye"
    );

    // Draw the outer and inner lip contours.
    drawContour(
        frame, p, OUTER_LIPS,
        Scalar(255, 0, 255),
        "Mouth"
    );

    drawContour(
        frame, p, INNER_LIPS,
        Scalar(255, 100, 255),
        "Inner lips",
        1
    );

    // Draw the nose.
    drawNose(frame, p);

    // Iris landmarks are available only in the 478-point model.
    if (p.size() >= 478)
    {
        drawIris(
            frame, p, LEFT_IRIS,
            Scalar(0, 255, 255),
            "Left iris"
        );

        drawIris(
            frame, p, RIGHT_IRIS,
            Scalar(0, 255, 255),
            "Right iris"
        );
    }

    // Draw approximate ear reference points.
    drawEars(frame, detection);

    // Draw all available facial landmarks.
    for (int i = 0; i < static_cast<int>(p.size()); i++)
    {
        Point pt = landmarkPoint(p, i);

        circle(
            frame,
            pt,
            1,
            Scalar(255, 255, 255),
            FILLED,
            LINE_AA
        );
    }

    // Display face presence confidence.
    putText(
        frame,
        "Face: " + to_string(face.presence).substr(0, 4),
        Point(
            cvRound(detection.box.x),
            max(20, cvRound(detection.box.y) - 10)
        ),
        FONT_HERSHEY_SIMPLEX,
        0.55,
        Scalar(0, 255, 0),
        2,
        LINE_AA
    );

    // Display the number of landmarks.
    putText(
        frame,
        "Landmarks: " + to_string(p.size()),
        Point(20, 65),
        FONT_HERSHEY_SIMPLEX,
        0.55,
        Scalar(255, 255, 0),
        1,
        LINE_AA
    );
}

// ============================================================
// MAIN APPLICATION
// ============================================================

int main(int argc, char** argv)
{
    try
    {
        // Select the landmark model using a command-line argument.
        // Usage:
        //     FaceMesh.exe 192
        //     FaceMesh.exe 256

        int landmarkSize = DEFAULT_LANDMARK_SIZE;

        if (argc > 1)
        {
            landmarkSize = atoi(argv[1]);
        }

        string landmarkModelPath;

        if (landmarkSize == 192)
        {
            landmarkModelPath = LANDMARK_MODEL_192;
        }
        else if (landmarkSize == 256)
        {
            landmarkModelPath = LANDMARK_MODEL_256;
        }
        else
        {
            throw runtime_error(
                "Invalid model size. Use 192 or 256."
            );
        }

        // Load BlazeFace detector.
        cout << "Loading BlazeFace..." << endl;

        BlazeFaceDetector detector(DETECTOR_MODEL);

        // Load the selected landmark model.
        cout << "Loading Face Landmarker..." << endl;

        FaceLandmarker landmarker(
            landmarkModelPath,
            landmarkSize
        );

        // Open the default webcam.
        VideoCapture camera(0);

        if (!camera.isOpened())
        {
            cerr << "Cannot open webcam!" << endl;
            return -1;
        }

        // Configure camera resolution.
        camera.set(CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(CAP_PROP_FRAME_HEIGHT, 720);

        cout << "Face tracking started." << endl;
        cout << "Selected model input: "
             << landmarkSize << "x" << landmarkSize << endl;

        cout << "Press ESC to exit." << endl;

        Mat frame;

        double fps = 0.0;

        while (true)
        {
            auto startTime = getTickCount();

            // Capture a frame.
            camera >> frame;

            if (frame.empty())
                break;

            // Detect faces using BlazeFace.
            vector<FaceDetection> faces =
                detector.detect(frame);

            // Process each detected face.
            for (const FaceDetection& detection : faces)
            {
                // Draw the face detection bounding box.
                rectangle(
                    frame,
                    detection.box,
                    Scalar(255, 0, 0),
                    2,
                    LINE_AA
                );

                // Calculate the rotated face ROI.
                FaceROI roi = createFaceROI(detection);

                // Run the selected landmark model.
                FaceLandmarks landmarks;

                bool success = landmarker.predict(
                    frame,
                    roi,
                    landmarks
                );

                if (success)
                {
                    // Draw eyes, iris, nose, mouth, contour and ears.
                    drawFace(
                        frame,
                        detection,
                        landmarks
                    );
                }
            }

            // Calculate the current FPS.
            double elapsed =
                (getTickCount() - startTime) / getTickFrequency();

            if (elapsed > 0.0)
                fps = 1.0 / elapsed;

            // Display FPS.
            putText(
                frame,
                "FPS: " + to_string(cvRound(fps)),
                Point(20, 35),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0, 255, 0),
                2,
                LINE_AA
            );

            // Show the processed camera frame.
            imshow(
                "MediaPipe Face Mesh - OpenCV DNN",
                frame
            );

            // ESC exits the application.
            if (waitKey(1) == 27)
                break;
        }

        // Release camera resources.
        camera.release();

        destroyAllWindows();
    }
    catch (const cv::Exception& e)
    {
        cerr << "OpenCV error: " << e.what() << endl;
        return -1;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
