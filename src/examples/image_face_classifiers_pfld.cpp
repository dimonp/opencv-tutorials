
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ============================================================
// Configuration
// ============================================================

// PFLD MobileNetV2 ONNX model.
static const string PFLD_MODEL_PATH =
    "data/pfld_mobilenetv2.onnx";

// YuNet face detector ONNX model.
static const string YUNET_MODEL_PATH =
    "data/yunet_int8.onnx";

// PFLD input image size.
// Most common PFLD MobileNetV2 models use 112x112 input.
static const int INPUT_WIDTH = 112;
static const int INPUT_HEIGHT = 112;

// Standard PFLD model predicts 68 facial landmarks.
static const int NUM_LANDMARKS = 68;

// Face bounding box expansion factor.
// The face ROI is enlarged to include the forehead and chin.
static const float FACE_SCALE = 1.25f;

// YuNet detection parameters.
static const float YUNET_SCORE_THRESHOLD = 0.6f;
static const float YUNET_NMS_THRESHOLD = 0.3f;
static const int YUNET_TOP_K = 5000;

// Landmark visualization settings.
static const int LANDMARK_RADIUS = 2;
static const int LANDMARK_THICKNESS = -1;

// Set to true if PFLD outputs coordinates in [-1, 1].
// Set to false if coordinates are in [0, 1].
static const bool OUTPUT_MINUS_ONE_TO_ONE = false;

// ============================================================
// Facial landmark connections
// ============================================================

// Standard 68-point facial landmark topology.
//
// Landmark groups:
//  0-16  : Jaw
// 17-21  : Right eyebrow
// 22-26  : Left eyebrow
// 27-30  : Nose bridge
// 31-35  : Nose bottom
// 36-41  : Right eye
// 42-47  : Left eye
// 48-59  : Outer mouth
// 60-67  : Inner mouth

static const vector<pair<int, int>> LANDMARK_CONNECTIONS = {

    // Jaw
    {0,1}, {1,2}, {2,3}, {3,4},
    {4,5}, {5,6}, {6,7}, {7,8},
    {8,9}, {9,10}, {10,11}, {11,12},
    {12,13}, {13,14}, {14,15}, {15,16},

    // Right eyebrow
    {17,18}, {18,19}, {19,20}, {20,21},

    // Left eyebrow
    {22,23}, {23,24}, {24,25}, {25,26},

    // Nose bridge
    {27,28}, {28,29}, {29,30},

    // Nose bottom
    {30,31}, {31,32}, {32,33},
    {33,34}, {34,35}, {35,30},

    // Right eye
    {36,37}, {37,38}, {38,39},
    {39,40}, {40,41}, {41,36},

    // Left eye
    {42,43}, {43,44}, {44,45},
    {45,46}, {46,47}, {47,42},

    // Outer mouth
    {48,49}, {49,50}, {50,51},
    {51,52}, {52,53}, {53,54},
    {54,55}, {55,56}, {56,57},
    {57,58}, {58,59}, {59,48},

    // Inner mouth
    {60,61}, {61,62}, {62,63},
    {63,64}, {64,65}, {65,66},
    {66,67}, {67,60}
};

// ============================================================
// PFLD Landmark Detector
// ============================================================

class PFLDDetector
{
private:

    // OpenCV DNN network.
    Net net;

public:

    // --------------------------------------------------------
    // Constructor
    // --------------------------------------------------------

    explicit PFLDDetector(const string& modelPath)
    {
        // Load PFLD ONNX model.
        net = readNetFromONNX(modelPath);

        if (net.empty())
        {
            throw runtime_error(
                "Failed to load PFLD ONNX model: " + modelPath
            );
        }

        // Configure CPU inference.
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        cout << "PFLD model loaded successfully." << endl;
    }

    // --------------------------------------------------------
    // Landmark inference
    // --------------------------------------------------------

    vector<Point2f> detect(const Mat& faceImage)
    {
        vector<Point2f> landmarks;

        if (faceImage.empty())
            return landmarks;

        // ----------------------------------------------------
        // 1. Resize face image to model input dimensions.
        // ----------------------------------------------------

        Mat resizedFace;

        resize(
            faceImage,
            resizedFace,
            Size(INPUT_WIDTH, INPUT_HEIGHT)
        );

        // ----------------------------------------------------
        // 2. Convert BGR image to RGB.
        // ----------------------------------------------------

        Mat rgbFace;

        cvtColor(
            resizedFace,
            rgbFace,
            COLOR_BGR2RGB
        );

        // ----------------------------------------------------
        // 3. Create normalized input blob.
        //
        // Common PFLD preprocessing:
        //
        // normalized = (pixel - 127.5) / 127.5
        //
        // This produces values approximately in [-1, 1].
        //
        // IMPORTANT:
        // Verify preprocessing for your specific ONNX model.
        // ----------------------------------------------------

        Mat blob = blobFromImage(
            rgbFace,
            1.0 / 127.5,
            Size(INPUT_WIDTH, INPUT_HEIGHT),
            Scalar(127.5, 127.5, 127.5),
            false,
            false
        );

        // ----------------------------------------------------
        // 4. Set network input.
        // ----------------------------------------------------

        net.setInput(blob);

        // ----------------------------------------------------
        // 5. Run neural network inference.
        // ----------------------------------------------------

        Mat output = net.forward();

        if (output.empty())
            return landmarks;

        // ----------------------------------------------------
        // 6. Convert network output to float.
        // ----------------------------------------------------

        Mat outputFloat;

        if (output.depth() == CV_32F)
        {
            outputFloat = output;
        }
        else
        {
            output.convertTo(outputFloat, CV_32F);
        }

        // Expected output:
        //
        // [1, 136] or equivalent tensor shape.
        //
        // 136 = 68 landmarks * 2 coordinates.

        if (outputFloat.total() != NUM_LANDMARKS * 2)
        {
            cerr << "Unexpected PFLD output size: "
                 << outputFloat.total()
                 << ", expected: "
                 << NUM_LANDMARKS * 2
                 << endl;

            return landmarks;
        }

        // Flatten output tensor to a single row.
        Mat flatOutput = outputFloat.reshape(1, 1);

        // ----------------------------------------------------
        // 7. Convert normalized coordinates into pixels.
        // ----------------------------------------------------

        for (int i = 0; i < NUM_LANDMARKS; i++)
        {
            float x = flatOutput.at<float>(0, i * 2);
            float y = flatOutput.at<float>(0, i * 2 + 1);

            // Convert [-1, 1] to [0, 1] if necessary.
            if (OUTPUT_MINUS_ONE_TO_ONE)
            {
                x = (x + 1.0f) * 0.5f;
                y = (y + 1.0f) * 0.5f;
            }

            // Convert normalized coordinates to input pixels.
            x *= INPUT_WIDTH;
            y *= INPUT_HEIGHT;

            landmarks.emplace_back(x, y);
        }

        return landmarks;
    }
};

// ============================================================
// YuNet Face Detector
// ============================================================

class YuNetDetector
{
private:

    // OpenCV YuNet face detector.
    Ptr<FaceDetectorYN> detector;

public:

    // --------------------------------------------------------
    // Constructor
    // --------------------------------------------------------

    explicit YuNetDetector(const string& modelPath)
    {
        // Create YuNet detector from ONNX model.
        //
        // Parameters:
        // 1. Model path.
        // 2. Configuration file (empty for ONNX).
        // 3. Input image size.
        // 4. Confidence threshold.
        // 5. NMS threshold.
        // 6. Maximum number of candidates.

        detector = FaceDetectorYN::create(
            modelPath,
            "",
            Size(320, 320),
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
            DNN_BACKEND_OPENCV,
            DNN_TARGET_CPU
        );

        if (detector.empty())
        {
            throw runtime_error(
                "Failed to initialize YuNet detector: " + modelPath
            );
        }

        cout << "YuNet detector loaded successfully." << endl;
    }

    // --------------------------------------------------------
    // Face detection
    // --------------------------------------------------------

    Mat detect(const Mat& frame)
    {
        Mat faces;

        if (frame.empty())
            return faces;

        // Update input size to match the current video frame.
        //
        // YuNet requires the detector input size to be set
        // according to the actual image dimensions.

        detector->setInputSize(frame.size());

        // Run face detection.
        //
        // Output format:
        //
        // Column 0:  x
        // Column 1:  y
        // Column 2:  width
        // Column 3:  height
        //
        // Columns 4-13:
        // Five facial landmarks (x, y pairs).
        //
        // Column 14:
        // Detection confidence.

        detector->detect(frame, faces);

        return faces;
    }
};

// ============================================================
// Face bounding box expansion
// ============================================================

Rect expandFaceBox(
    const Rect& face,
    const Size& imageSize,
    float scale
)
{
    // Calculate face center.
    Point2f center(
        face.x + face.width * 0.5f,
        face.y + face.height * 0.5f
    );

    // Create an enlarged square region.
    float side = max(face.width, face.height) * scale;

    int x = static_cast<int>(center.x - side * 0.5f);
    int y = static_cast<int>(center.y - side * 0.5f);

    int width = static_cast<int>(side);
    int height = static_cast<int>(side);

    Rect expanded(x, y, width, height);

    // Clip the ROI to image boundaries.
    expanded &= Rect(
        0,
        0,
        imageSize.width,
        imageSize.height
    );

    return expanded;
}

// ============================================================
// Map landmarks from model coordinates to frame coordinates
// ============================================================

vector<Point> mapLandmarksToFrame(
    const vector<Point2f>& landmarks,
    const Rect& faceROI
)
{
    vector<Point> result;

    for (const auto& point : landmarks)
    {
        // Scale coordinates from model space to ROI space.
        float x = point.x * faceROI.width / INPUT_WIDTH;
        float y = point.y * faceROI.height / INPUT_HEIGHT;

        // Add ROI offset to obtain full-frame coordinates.
        int frameX = cvRound(faceROI.x + x);
        int frameY = cvRound(faceROI.y + y);

        result.emplace_back(frameX, frameY);
    }

    return result;
}

// ============================================================
// Draw facial landmarks
// ============================================================


void drawLandmarks(cv::Mat& frame,
                   const std::vector<cv::Point>& landmarks)
{
    if (landmarks.size() != 68)
        return;

    // Define colors for each facial feature (BGR format)
    const cv::Scalar FACE_COLOR(0, 0, 255);       // Red
    const cv::Scalar EYEBROW_COLOR(255, 0, 255);  // Purple
    const cv::Scalar NOSE_COLOR(255, 0, 0);       // Blue
    const cv::Scalar EYES_COLOR(0, 255, 0);       // Green
    const cv::Scalar MOUTH_COLOR(0, 165, 255);    // Orange

    // Draw face contour
    for (int i = 0; i <= 16; i++)
        cv::circle(frame, landmarks[i], 2, FACE_COLOR, -1);

    // Draw eyebrows
    for (int i = 17; i <= 26; i++)
        cv::circle(frame, landmarks[i], 2, EYEBROW_COLOR, -1);

    // Draw nose
    for (int i = 27; i <= 35; i++)
        cv::circle(frame, landmarks[i], 2, NOSE_COLOR, -1);

    // Draw eyes
    for (int i = 36; i <= 47; i++)
        cv::circle(frame, landmarks[i], 3, EYES_COLOR, -1);

    // Draw mouth
    for (int i = 48; i <= 67; i++)
        cv::circle(frame, landmarks[i], 3, MOUTH_COLOR, -1);
}

// ============================================================
// Main application
// ============================================================

int main(int argc, char** argv)
{
    try
    {
        // ----------------------------------------------------
        // 1. Read command-line arguments.
        // ----------------------------------------------------

        // Default input is the first webcam.
        //
        // If a filename is provided as an argument,
        // the application will process that video file.

        string videoPath;

        if (argc > 1)
        {
            videoPath = argv[1];
        }

        // ----------------------------------------------------
        // 2. Initialize PFLD detector.
        // ----------------------------------------------------

        PFLDDetector pfld(PFLD_MODEL_PATH);

        // ----------------------------------------------------
        // 3. Initialize YuNet face detector.
        // ----------------------------------------------------

        YuNetDetector yunet(YUNET_MODEL_PATH);

        // ----------------------------------------------------
        // 4. Open video source.
        // ----------------------------------------------------

        VideoCapture cap;

        if (videoPath.empty())
        {
            // Open default webcam.
            cap.open(0);
        }
        else
        {
            // Open video file.
            cap.open(videoPath);
        }

        if (!cap.isOpened())
        {
            throw runtime_error(
                "Cannot open camera or video file."
            );
        }

        // ----------------------------------------------------
        // 5. Initialize video writer.
        // ----------------------------------------------------

        VideoWriter writer;

        bool writerInitialized = false;

        // ----------------------------------------------------
        // 6. Main video processing loop.
        // ----------------------------------------------------

        Mat frame;

        double fps = 0.0;

        while (true)
        {
            // Read next video frame.
            if (!cap.read(frame))
                break;

            if (frame.empty())
                break;

            auto startTime = chrono::steady_clock::now();

            // ------------------------------------------------
            // Detect faces using YuNet.
            // ------------------------------------------------

            Mat faces = yunet.detect(frame);

            // ------------------------------------------------
            // Process all detected faces.
            // ------------------------------------------------

            for (int i = 0; i < faces.rows; i++)
            {
                // --------------------------------------------
                // Read YuNet detection result.
                // --------------------------------------------

                float x = faces.at<float>(i, 0);
                float y = faces.at<float>(i, 1);

                float w = faces.at<float>(i, 2);
                float h = faces.at<float>(i, 3);

                // Detection confidence.
                float confidence = faces.at<float>(i, 14);

                // --------------------------------------------
                // Convert floating-point box to integer Rect.
                // --------------------------------------------

                int left = cvRound(x);
                int top = cvRound(y);

                int width = cvRound(w);
                int height = cvRound(h);

                Rect face(
                    left,
                    top,
                    width,
                    height
                );

                // Clip detected face to image boundaries.
                face &= Rect(
                    0,
                    0,
                    frame.cols,
                    frame.rows
                );

                if (face.empty())
                    continue;

                // --------------------------------------------
                // Expand detected face region.
                // --------------------------------------------

                Rect faceROI = expandFaceBox(
                    face,
                    frame.size(),
                    FACE_SCALE
                );

                if (faceROI.empty())
                    continue;

                // --------------------------------------------
                // Extract face image.
                // --------------------------------------------

                Mat faceImage = frame(faceROI);

                // --------------------------------------------
                // Run PFLD inference.
                // --------------------------------------------

                vector<Point2f> localLandmarks =
                    pfld.detect(faceImage);

                if (localLandmarks.size() != NUM_LANDMARKS)
                    continue;

                // --------------------------------------------
                // Convert landmarks to original frame space.
                // --------------------------------------------

                vector<Point> frameLandmarks =
                    mapLandmarksToFrame(
                        localLandmarks,
                        faceROI
                    );

                // --------------------------------------------
                // Draw facial landmarks.
                // --------------------------------------------

                drawLandmarks(
                    frame,
                    frameLandmarks
                );

                // --------------------------------------------
                // Draw face bounding box.
                // --------------------------------------------

                rectangle(
                    frame,
                    face,
                    Scalar(255, 0, 0),
                    2,
                    LINE_AA
                );

                // --------------------------------------------
                // Draw detection confidence.
                // --------------------------------------------

                string label = std::format(
                    "YuNet: %.2f | PFLD: 68",
                    confidence
                );

                putText(
                    frame,
                    label,
                    Point(
                        face.x,
                        max(20, face.y - 10)
                    ),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(255, 255, 0),
                    1,
                    LINE_AA
                );
            }

            // ------------------------------------------------
            // Calculate FPS.
            // ------------------------------------------------

            auto endTime = chrono::steady_clock::now();

            double elapsed = chrono::duration<double>(
                endTime - startTime
            ).count();

            if (elapsed > 0.0)
            {
                fps = 1.0 / elapsed;
            }

            // ------------------------------------------------
            // Display FPS.
            // ------------------------------------------------

            string fpsText = std::format(
                "FPS: %.1f",
                fps
            );

            putText(
                frame,
                fpsText,
                Point(20, 30),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0, 255, 255),
                2,
                LINE_AA
            );

            // ------------------------------------------------
            // Initialize video writer for video file input.
            // ------------------------------------------------

            if (!writerInitialized && !videoPath.empty())
            {
                double videoFPS = cap.get(CAP_PROP_FPS);

                if (videoFPS <= 0.0)
                    videoFPS = 25.0;

                int fourcc = VideoWriter::fourcc(
                    'm', 'p', '4', 'v'
                );

                writer.open(
                    "output_landmarks.mp4",
                    fourcc,
                    videoFPS,
                    frame.size()
                );

                writerInitialized = writer.isOpened();

                if (!writerInitialized)
                {
                    cerr << "Warning: Cannot initialize video writer."
                         << endl;
                }
            }

            // Save processed frame.
            if (writerInitialized)
            {
                writer.write(frame);
            }

            // ------------------------------------------------
            // Display processed video frame.
            // ------------------------------------------------

            imshow(
                "YuNet + PFLD Facial Landmark Detection",
                frame
            );

            // ------------------------------------------------
            // Keyboard controls.
            //
            // ESC or Q: Exit application.
            // ------------------------------------------------

            int key = waitKey(1);

            if (key == 27 || key == 'q' || key == 'Q')
                break;
        }

        // ----------------------------------------------------
        // 7. Release resources.
        // ----------------------------------------------------

        cap.release();

        if (writer.isOpened())
        {
            writer.release();
        }

        destroyAllWindows();

        cout << "Processing finished." << endl;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
