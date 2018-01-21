#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <string>

class Segment {
public:
	Segment(cv::Mat_<bool>& image, cv::Mat_<bool>& visited) : image_(image), visited_(visited), segment_(cv::Mat_<bool>(image_.rows, image_.cols, false)),
		cropped_(segment_), counter_(0), boundingBox_(cv::Rect(0, 0, image_.cols, image_.rows)), iC_(0.0), jC_(0.0) {}

	void extract(int i, int j);

	int getPixelsCount();

	cv::Mat_<bool> getCropped();

	cv::Rect& getBoundingBox();

	double getCoeffW3();

	double getCoeffW6();

	double getCoeffM1();

	double getCoeffM2();

	double getCoeffM3();

	double getCoeffM4();

	double getCoeffM5();

	double getCoeffM6();

	double getCoeffM7();

	cv::Point getB1();

	cv::Point getB2();

	cv::Point getB3();

	cv::Point getB4();

	cv::Point getP1();

	cv::Point getP2();

	cv::Point getP3();

	cv::Point getP4();

	std::vector<cv::Point> getPoints();

private:
	bool coordsOk(int i, int j);

	void addPixel(int i, int j);

	void updateBoundingBox();

	int perimeter();

	double m(int p, int q, double ic = 0.0, double jc = 0.0);

	int area();

	void updateMassCenter();

	bool isEdge(int i, int j);

	cv::Mat_<bool> image_;
	cv::Mat_<bool> visited_;
	cv::Mat_<bool> segment_, cropped_;
	int counter_;
	cv::Rect boundingBox_;
	double m00_, iC_, jC_;
};
