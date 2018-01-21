#pragma once

//#define SHOW_SEGMENTATION_MASKS  // Show raw segmentation output
//#define CLASSIFY_ALL  // Do not filter out segmentation results

// Segmentation settings
const int FED_H_VAL = 122;
const int FED_H_EPS = 10;
const int FED_S_MIN = 100;
const int EX_H_VAL = 5;
const int EX_H_EPS = 10;
const int EX_S_MIN = 100;
const int SEGMENT_MIN_PIXELS = 150;

// Classification settings
const double FED_W6_MIN = 0.902;
const double FED_W6_MAX = 0.944;
const double FED_M1_MIN = 0.267;
const double FED_M1_MAX = 0.397;
const double FED_M2_MIN = 0.032;
const double FED_M2_MAX = 0.082;
const double FED_M7_MIN = 0.009;
const double FED_M7_MAX = 0.020;
const double EX_W6_MIN = 0.914;
const double EX_W6_MAX = 0.966;
const double EX_M1_MIN = 0.197;
const double EX_M1_MAX = 0.306;
const double EX_M2_MIN = 0.005;
const double EX_M2_MAX = 0.022;
const double EX_M6_MIN = 0.000002;
const double EX_M6_MAX = 0.000036;
const double EX_M7_MIN = 0.007;
const double EX_M7_MAX = 0.019;

// Grouping settings
const int GROUP_MAX_DIST = 8;
const auto GROUP_COLOR = cv::Vec3b(0, 255, 255);
const int GROUP_THICKNESS = 2;

// Results display settings
const auto FED_COLOR = cv::Vec3b(255, 83, 125);
const auto EX_COLOR = cv::Vec3b(34, 126, 255);
const int BOUNDINGBOX_THICKNESS = 1;