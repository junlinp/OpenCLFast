#ifndef CPU_FAST_H
#define CPU_FAST_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
typedef unsigned char uchar;
#include <vector>

// Add your function declarations here

bool IsBrighter(uchar p, uchar center, int threshold);
bool IsDarker(uchar p, uchar center, int threshold);
std::vector<uchar> GetCirclePixels(const cv::Mat& img, int row, int col);

// Checks if pixel at (row,col) is a corner using FAST algorithm
bool IsCorner(const cv::Mat& img, int row, int col, int threshold) {
    if (row < 3 || col < 3 || row >= img.rows - 3 || col >= img.cols - 3) {
        return false;
    }

    uchar center = img.at<uchar>(row, col);

    // Get 16 pixels in circle around center point
    std::vector<uchar> circle = {
        img.at<uchar>(row-3, col),    // 0
        img.at<uchar>(row-3, col+1),  // 1
        img.at<uchar>(row-2, col+2),  // 2
        img.at<uchar>(row-1, col+3),  // 3
        img.at<uchar>(row, col+3),    // 4
        img.at<uchar>(row+1, col+3),  // 5
        img.at<uchar>(row+2, col+2),  // 6
        img.at<uchar>(row+3, col+1),  // 7
        img.at<uchar>(row+3, col),    // 8
        img.at<uchar>(row+3, col-1),  // 9
        img.at<uchar>(row+2, col-2),  // 10
        img.at<uchar>(row+1, col-3),  // 11
        img.at<uchar>(row, col-3),    // 12
        img.at<uchar>(row-1, col-3),  // 13
        img.at<uchar>(row-2, col-2),  // 14
        img.at<uchar>(row-3, col-1)   // 15
    };

    // Check for 12 contiguous pixels brighter or darker than center
    // First check for brighter pixels
    int maxCount = 0;
    int currentCount = 0;
    
    // Check twice the length to handle wrap-around cases
    for (int i = 0; i < 32; i++) {
        int idx = i % 16;  // Wrap around to start of circle
        if (IsBrighter(circle[idx], center, threshold)) {
            currentCount++;
            maxCount = std::max(maxCount, currentCount);
        } else {
            currentCount = 0;
        }
    }
    
    if (maxCount >= 9) return true;
    
    // Check for darker pixels similarly
    maxCount = currentCount = 0;
    for (int i = 0; i < 32; i++) {
        int idx = i % 16;
        if (IsDarker(circle[idx], center, threshold)) {
            currentCount++;
            maxCount = std::max(maxCount, currentCount);
        } else {
            currentCount = 0;
        }
    }
    
    return maxCount >= 9;
}

// Helper functions
inline bool IsBrighter(uchar p, uchar center, int threshold) {
    return p >= center + threshold;
}

inline bool IsDarker(uchar p, uchar center, int threshold) {
    return p <= center - threshold;
}

// Main FAST detector function
void DetectFASTCorners(const cv::Mat& input, cv::Mat& output, int threshold = 20) {
    output = cv::Mat::zeros(input.size(), CV_8UC1);
    
    for (int row = 3; row < input.rows - 3; row++) {
        for (int col = 3; col < input.cols - 3; col++) {
            if (IsCorner(input, row, col, threshold)) {
                output.at<uchar>(row, col) = 255;
            }
        }
    }
}


// Detect FAST corners with non-maximum suppression
inline void DetectFASTCornersWithNMS(const cv::Mat& input, cv::Mat& output, int threshold = 20) {
    // First detect corners normally
    cv::Mat corners = cv::Mat::zeros(input.size(), CV_8UC1);
    std::vector<cv::KeyPoint> keypoints;
    
    // For each pixel, calculate corner score if it passes initial FAST test
    for (int row = 3; row < input.rows - 3; row++) {
        for (int col = 3; col < input.cols - 3; col++) {
            if (IsCorner(input, row, col, threshold)) {
                // Calculate corner score as max difference between center and contiguous arc
                uchar center = input.at<uchar>(row, col);
                std::vector<uchar> circle = GetCirclePixels(input, row, col);
                
                int maxScore = 0;
                for (int i = 0; i < 16; i++) {
                    int minVal = 255, maxVal = 0;
                    // Check 9 contiguous pixels
                    for (int j = 0; j < 9; j++) {
                        int idx = (i + j) % 16;
                        minVal = std::min(minVal, (int)circle[idx]);
                        maxVal = std::max(maxVal, (int)circle[idx]);
                    }
                    maxScore = std::max(maxScore, 
                        std::max(minVal - (int)center, (int)center - maxVal));
                }
                
                keypoints.push_back(cv::KeyPoint(col, row, 3, -1, maxScore));
                corners.at<uchar>(row, col) = 255;
            }
        }
    }
    
    // Non-maximum suppression
    output = cv::Mat::zeros(input.size(), CV_8UC1);
    const int radius = 3; // Suppression radius
    
    // Sort keypoints by score
    std::sort(keypoints.begin(), keypoints.end(), 
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });
    
    // Keep track of suppressed points
    std::vector<bool> suppressed(keypoints.size(), false);
    
    // For each keypoint
    for (size_t i = 0; i < keypoints.size(); i++) {
        if (suppressed[i]) continue;
        
        // Mark this point in output
        output.at<uchar>(keypoints[i].pt.y, keypoints[i].pt.x) = 255;
        
        // Suppress weaker neighbors
        for (size_t j = i + 1; j < keypoints.size(); j++) {
            if (!suppressed[j]) {
                float dx = keypoints[i].pt.x - keypoints[j].pt.x;
                float dy = keypoints[i].pt.y - keypoints[j].pt.y;
                if (dx*dx + dy*dy <= radius*radius) {
                    suppressed[j] = true;
                }
            }
        }
    }
}

// Helper function to get circle pixels
std::vector<uchar> GetCirclePixels(const cv::Mat& img, int row, int col) {
    std::vector<uchar> circle(16);
    const int offsets[16][2] = {
        {0,-3}, {1,-3}, {2,-2}, {3,-1}, {3,0}, {3,1}, {2,2}, {1,3},
        {0,3}, {-1,3}, {-2,2}, {-3,1}, {-3,0}, {-3,-1}, {-2,-2}, {-1,-3}
    };
    
    for (int i = 0; i < 16; i++) {
        circle[i] = img.at<uchar>(row + offsets[i][1], col + offsets[i][0]);
    }
    return circle;
}


#endif // CPU_FAST_H
