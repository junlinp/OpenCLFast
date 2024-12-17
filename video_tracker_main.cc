#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return -1;
    }

    // Open video file
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "Error opening video file" << std::endl;
        return -1;
    }

    cv::Mat prev_frame, curr_frame;
    
    // Read first frame
    cap >> prev_frame;
    if (prev_frame.empty()) {
        std::cout << "Error reading video" << std::endl;
        return -1;
    }

    // Create window
    cv::namedWindow("Video Tracking", cv::WINDOW_NORMAL);

    while (true) {
        // Read current frame
        cap >> curr_frame;
        if (curr_frame.empty())
            break;

        // Create combined image
        cv::Mat display;
        // Convert frames to grayscale
        cv::Mat prev_gray, curr_gray;
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        // Detect FAST corners in previous frame
        std::vector<cv::Point2f> prev_corners;
        std::vector<cv::KeyPoint> keypoints;
        int threshold = 20;
        cv::FAST(prev_gray, keypoints, threshold, true);
        
        // Convert keypoints to points for optical flow
        for(const auto& kp : keypoints) {
            prev_corners.push_back(kp.pt);
        }

        // Calculate optical flow
        std::vector<cv::Point2f> curr_corners;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corners, curr_corners, status, err);

        // Draw the tracks
        for(size_t i = 0; i < prev_corners.size(); i++) {
            if(status[i]) {
                // Draw point in prev_frame
                cv::circle(prev_frame, prev_corners[i], 3, cv::Scalar(0, 255, 0), -1);
                
                // Draw point in curr_frame
                cv::circle(curr_frame, curr_corners[i], 3, cv::Scalar(0, 255, 0), -1);
            }
        }
        cv::hconcat(prev_frame, curr_frame, display);

        // Draw line from prev_frame to curr_frame showing motion
        for(size_t i = 0; i < prev_corners.size(); i++) {
            if(status[i]) {
                cv::line(display, prev_corners[i], curr_corners[i] + cv::Point2f(prev_frame.cols, 0), cv::Scalar(0, 255, 0));
            }
        }

        // Show the combined image
        cv::imshow("Video Tracking", display);

        // Update previous frame
        curr_frame.copyTo(prev_frame);

        // Break if 'q' is pressed
        char c = (char)cv::waitKey(25);
        if (c == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
