#pragma once

#include "Segment.h"

class Image {
public:
	Image(const char * name) : image_(cv::imread(name)), name_(name) {
		std::cout << "File " << name_ << " loaded." << std::endl;
	};

	void process();

private:
	cv::Mat segmentate(cv::Mat & I, const int hVal, const int hEps, const int sMin);

	cv::Mat segmentateFed(cv::Mat & I);

	cv::Mat segmentateEx(cv::Mat & I);

	cv::Mat maskToImage(cv::Mat& I);

	void erode1(cv::Mat& I);

	void dilate1(cv::Mat& I);

	void erode2(cv::Mat& I);

	void dilate2(cv::Mat& I);

	cv::Vec3b pixelBGRToHSV(const cv::Vec3b& I);

	void BGRToHSV(cv::Mat& O);

	cv::Vec3b mult(const cv::Mat_<cv::Vec3b>& I, const cv::Mat_<float>& F, int _i, int _j);

	cv::Mat conv(const cv::Mat& I, const cv::Mat& M);

	std::vector<Segment> getSegments(const cv::Mat& I);

	void renderResults(std::vector<Segment> &segs, const std::string& segType, const cv::Vec3b& bbColor);

	void matchSegs(std::vector<Segment> &feds, std::vector<Segment> &exs);

	cv::Mat image_;
	std::string name_;
};