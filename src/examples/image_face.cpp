#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>

// Define detection modes
enum DetectionMode {
    FACE_ONLY = 0,
    EYES_ONLY = 1,
    SMILE_ONLY = 2,
    FULLBODY_ONLY = 3,
    ALL_FEATURES = 4,
    MODE_COUNT = 5 // Total number of modes
};

int main() {
    // 1. Load Haar Cascade XML classifiers
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade;
    cv::CascadeClassifier smileCascade;
    cv::CascadeClassifier bodyCascade;

    // Load files (make sure these XML files are in your working directory)
    if (!faceCascade.load("data/haarcascade_frontalface_default.xml") ||
        !eyeCascade.load("data/haarcascade_eye.xml") ||
        !smileCascade.load("data/haarcascade_smile.xml") ||
        !bodyCascade.load("data/haarcascade_fullbody.xml")) {
        std::cerr << "Error: Could not load one or more Haar Cascade XML files!" << std::endl;
        std::cerr << "Please ensure you have copied the XML files from opencv/data/haarcascades/" << std::endl;
        return -1;
    }

    // 2. Open the default web camera (ID 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam!" << std::endl;
        return -1;
    }

    cv::Mat frame, gray;
    DetectionMode currentMode = FACE_ONLY;

    std::cout << "=== Controls ===" << std::endl;
    std::cout << "Press [SPACE] to switch detection parts" << std::endl;
    std::cout << "Press [ESC] to exit" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        // 3. Preprocess frame: convert to grayscale and equalize histogram for better detection
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Vectors to store detected bounding boxes
        std::vector<cv::Rect> faces, eyes, smiles, bodies;

        // 4. Detect Full Body (Processed on the whole frame)
        if (currentMode == FULLBODY_ONLY || currentMode == ALL_FEATURES) {
            // Adjust parameters (scaleFactor=1.1, minNeighbors=3) for fullbody detection stability
            bodyCascade.detectMultiScale(gray, bodies, 1.1, 3, 0, cv::Size(40, 70));

            for (const auto& body : bodies) {
                cv::rectangle(frame, body, cv::Scalar(255, 0, 255), 2); // Magenta for Full Body
                cv::putText(frame, "Full Body", cv::Point(body.x, body.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
            }
        }

        // 5. Detect Face components
        if (currentMode == FACE_ONLY || currentMode == ALL_FEATURES || currentMode == EYES_ONLY || currentMode == SMILE_ONLY) {
            faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(30, 30));
        }

        // Draw and process detected parts inside the face
        for (const auto& face : faces) {
            // Draw face rectangle if needed
            if (currentMode == FACE_ONLY || currentMode == ALL_FEATURES) {
                cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2); // Blue for Face
                cv::putText(frame, "Face", cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
            }

            // Define the Region of Interest (ROI) within the face boundaries
            cv::Mat faceROI = gray(face);

            // Detect and draw eyes
            if (currentMode == EYES_ONLY || currentMode == ALL_FEATURES) {
                eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 5, 0, cv::Size(20, 20));
                for (const auto& eye : eyes) {
                    cv::Point center(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                    int radius = cvRound((eye.width + eye.height) * 0.25);
                    cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 2); // Green for Eyes
                }
            }

            // Detect and draw smiles
            if (currentMode == SMILE_ONLY || currentMode == ALL_FEATURES) {
                smileCascade.detectMultiScale(faceROI, smiles, 1.5, 20, 0, cv::Size(25, 25));
                for (const auto& smile : smiles) {
                    cv::Rect smileRect(face.x + smile.x, face.y + smile.y, smile.width, smile.height);
                    cv::rectangle(frame, smileRect, cv::Scalar(0, 0, 255), 2); // Red for Smile
                }
            }
        }

        // 6. Display current mode information on top of the video
        std::string modeText = "Mode: ";
        if (currentMode == FACE_ONLY) modeText += "Faces Only";
        else if (currentMode == EYES_ONLY) modeText += "Eyes Only";
        else if (currentMode == SMILE_ONLY) modeText += "Smiles Only";
        else if (currentMode == FULLBODY_ONLY) modeText += "Full Body Only";
        else if (currentMode == ALL_FEATURES) modeText += "All Features";

        cv::putText(frame, modeText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        // 7. Show the frame
        cv::imshow("Haar Cascade Multi-Detection", frame);

        // 8. Handle keyboard inputs
        int key = cv::waitKey(30);
        if (key == 27) { // ESC key to exit
            break;
        } else if (key == 32) { // Spacebar key to switch modes
            currentMode = static_cast<DetectionMode>((currentMode + 1) % MODE_COUNT);
            std::cout << "Switched to next detection mode." << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
