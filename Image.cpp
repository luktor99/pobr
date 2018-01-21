#include "Image.h"
#include "Settings.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

void Image::process()
{
	// Convert to HSV colorspace
	std::cout << "Converting the input image to HSV colorspace..." << std::endl;
	cv::Mat imagehsv;
	BGRToHSV(imagehsv);

	// Segmentation
	std::cout << "Starting segmentation..." << std::endl;
	auto maskFed = segmentateFed(imagehsv);
	auto maskEx = segmentateEx(imagehsv);

	// Perform closing on the images
	std::cout << "Closing the masks..." << std::endl;
	dilate2(maskFed);
	erode1(maskFed);
	dilate2(maskEx);
	erode1(maskEx);

	// Display segmentation masks
#ifdef SHOW_SEGMENTATION_MASKS
	cv::imshow(name_ + ": maskFed", maskToImage(maskFed));
	cv::imshow(name_ + ": maskEx", maskToImage(maskEx));
#endif

	// Retrieve individual segments
	std::cout << "Clustering the segmented pixels..." << std::endl;
	auto segsFed = getSegments(maskFed);
	auto segsEx = getSegments(maskEx);
	std::cout << "-> Found " << segsFed.size() << " possible Fed segments." << std::endl;
	std::cout << "-> Found " << segsEx.size() << " possible Ex segments." << std::endl;

	// Classify the segments
	std::cout << "Starting classification..." << std::endl;
#ifndef CLASSIFY_ALL
	segsFed.erase(std::remove_if(segsFed.begin(), segsFed.end(), [](Segment &s) {
		auto W6 = s.getCoeffW6();
		auto M1 = s.getCoeffM1();
		auto M2 = s.getCoeffM2();
		auto M7 = s.getCoeffM7();
		return
			(W6 < FED_W6_MIN || W6 > FED_W6_MAX) ||
			(M1 < FED_M1_MIN || M1 > FED_M1_MAX) ||
			(M2 < FED_M2_MIN || M2 > FED_M2_MAX) ||
			(M7 < FED_M7_MIN || M7 > FED_M7_MAX);
	}), segsFed.end());
	segsEx.erase(std::remove_if(segsEx.begin(), segsEx.end(), [](Segment &s) {
		auto W6 = s.getCoeffW6();
		auto M1 = s.getCoeffM1();
		auto M2 = s.getCoeffM2();
		auto M6 = s.getCoeffM6();
		auto M7 = s.getCoeffM7();
		return
			(W6 < EX_W6_MIN || W6 > EX_W6_MAX) ||
			(M1 < EX_M1_MIN || M1 > EX_M1_MAX) ||
			(M2 < EX_M2_MIN || M2 > EX_M2_MAX) ||
			(M6 < EX_M6_MIN || M6 > EX_M6_MAX) ||
			(M7 < EX_M7_MIN || M7 > EX_M7_MAX);
	}), segsEx.end());
#endif
	std::cout << "-> Found " << segsFed.size() << " Fed segments." << std::endl;
	std::cout << "-> Found " << segsEx.size() << " Ex segments." << std::endl;

	// Group matching segments
	std::cout << "Grouping matching segments..." << std::endl;
	matchSegs(segsEx, segsFed);

	// Render results
	std::cout << "Rendering results..." << std::endl;
	renderResults(segsFed, "Fed", FED_COLOR);
	renderResults(segsEx, "Ex", EX_COLOR);

	// Display results
	cv::imshow(name_ + ": results", image_);

	cv::waitKey(-1);
}

cv::Mat Image::segmentate(cv::Mat & I, const int hVal, const int hEps, const int sMin) {
	cv::Mat_<cv::Vec3b> I_ = I;
	cv::Mat_<bool> O_(I.rows, I.cols);

	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			auto const &h = I_(i, j)[0];
			auto const &s = I_(i, j)[1];

			if (h > hVal - hEps &&
				h < hVal + hEps &&
				s > sMin)
				O_(i, j) = true;
			else
				O_(i, j) = false;
		}
	}

	return O_;
}

cv::Mat Image::segmentateFed(cv::Mat & I) {
	return segmentate(I, FED_H_VAL, FED_H_EPS, FED_S_MIN);
}

cv::Mat Image::segmentateEx(cv::Mat & I) {
	return segmentate(I, EX_H_VAL, EX_H_EPS, EX_S_MIN);
}

cv::Mat Image::maskToImage(cv::Mat & I) {
	cv::Mat_<bool> I_ = I;
	cv::Mat_<cv::Vec3b> O_(I.rows, I.cols);

	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			if (I_(i, j))
				O_(i, j) = cv::Vec3b(255, 255, 255);
			else
				O_(i, j) = cv::Vec3b(0, 0, 0);
		}
	}

	return O_;
}

void Image::erode1(cv::Mat & I) {
	cv::Mat_<bool> I_ = I.clone();
	cv::Mat_<bool> O_ = I;

	for (int i = 1; i < I.rows - 1; ++i) {
		for (int j = 1; j < I.cols - 1; ++j) {
			if (I_(i - 1, j - 1) && I_(i - 1, j) && I_(i - 1, j + 1) &&
				I_(i, j - 1) && I_(i, j + 1) &&
				I_(i + 1, j - 1) && I_(i + 1, j) && I_(i + 1, j + 1))
				O_(i, j) = true;
			else
				O_(i, j) = false;
		}
	}
}

void Image::dilate1(cv::Mat & I) {
	cv::Mat_<bool> I_ = I.clone();
	cv::Mat_<bool> O_ = I;

	for (int i = 1; i < I.rows - 1; ++i) {
		for (int j = 1; j < I.cols - 1; ++j) {
			if (I_(i - 1, j - 1) || I_(i - 1, j) || I_(i - 1, j + 1) ||
				I_(i, j - 1) || I_(i, j + 1) ||
				I_(i + 1, j - 1) || I_(i + 1, j) || I_(i + 1, j + 1))
				O_(i, j) = true;
			else
				O_(i, j) = false;
		}
	}
}

void Image::erode2(cv::Mat & I) {
	cv::Mat_<bool> I_ = I.clone();
	cv::Mat_<bool> O_ = I;

	for (int i = 2; i < I.rows - 2; ++i) {
		for (int j = 2; j < I.cols - 2; ++j) {
			if (I_(i - 2, j - 2) && I_(i - 2, j - 1) && I_(i - 2, j) && I_(i - 2, j + 1) && I_(i - 2, j + 2) &&
				I_(i - 1, j - 2) && I_(i - 1, j - 1) && I_(i - 1, j) && I_(i - 1, j + 1) && I_(i - 1, j + 2) &&
				I_(i, j - 2) && I_(i, j - 1) && I_(i, j + 1) && I_(i, j + 2) &&
				I_(i + 1, j - 2) && I_(i + 1, j - 1) && I_(i + 1, j) && I_(i + 1, j + 1) && I_(i + 1, j + 2) &&
				I_(i + 2, j - 2) && I_(i + 2, j - 1) && I_(i + 2, j) && I_(i + 2, j + 1) && I_(i + 2, j + 2))
				O_(i, j) = true;
			else
				O_(i, j) = false;
		}
	}
}

void Image::dilate2(cv::Mat & I) {
	cv::Mat_<bool> I_ = I.clone();
	cv::Mat_<bool> O_ = I;

	for (int i = 2; i < I.rows - 2; ++i) {
		for (int j = 2; j < I.cols - 2; ++j) {
			if (I_(i - 2, j - 2) || I_(i - 2, j - 1) || I_(i - 2, j) || I_(i - 2, j + 1) || I_(i - 2, j + 2) ||
				I_(i - 1, j - 2) || I_(i - 1, j - 1) || I_(i - 1, j) || I_(i - 1, j + 1) || I_(i - 1, j + 2) ||
				I_(i, j - 2) || I_(i, j - 1) || I_(i, j + 1) || I_(i, j + 2) ||
				I_(i + 1, j - 2) || I_(i + 1, j - 1) || I_(i + 1, j) || I_(i + 1, j + 1) || I_(i + 1, j + 2) ||
				I_(i + 2, j - 2) || I_(i + 2, j - 1) || I_(i + 2, j) || I_(i + 2, j + 1) || I_(i + 2, j + 2))
				O_(i, j) = true;
			else
				O_(i, j) = false;
		}
	}
}

cv::Vec3b Image::pixelBGRToHSV(const cv::Vec3b & I) {
	double h = 0.0;
	uchar s;
	uchar v = std::max(std::max(I[2], I[1]), I[0]);
	uchar delta = v - std::min(std::min(I[2], I[1]), I[0]);

	if (v == 0)
		s = 0;
	else
		s = delta * 255 / v;

	if (s != 0) {
		if (I[2] == v)
			h = (I[1] - I[0]) / static_cast<double>(delta);
		else if (I[1] == v)
			h = 2.0 + (I[0] - I[2]) / static_cast<double>(delta);
		else if (I[0] == v)
			h = 4.0 + (I[2] - I[1]) / static_cast<double>(delta);

		h *= 179.0 / 6.0;

		if (h < 0.0)
			h += 179.0;
	}

	return cv::Vec3b{ static_cast<uchar>(std::round(h)), static_cast<uchar>(std::round(s)), v };
}

void Image::BGRToHSV(cv::Mat & O) {
	cv::Mat_<cv::Vec3b> O_ = image_.clone();
	for (int i = 0; i < image_.rows; ++i) {
		for (int j = 0; j < image_.cols; ++j) {
			O_(i, j) = pixelBGRToHSV(O_(i, j));
		}
	}

	O = O_;
}

cv::Vec3b Image::mult(const cv::Mat_<cv::Vec3b>& I, const cv::Mat_<float>& F, int _i, int _j) {
	_i -= F.rows / 2;
	_j -= F.cols / 2;
	cv::Vec3b out{ 0, 0, 0 };
	float R = 0.0f, G = 0.0f, B = 0.0f;

	for (int i = 0; i < F.rows; ++i)
		for (int j = 0; j < F.cols; ++j) {
			R += I(_i + i, _j + j)[0] * F(i, j);
			G += I(_i + i, _j + j)[1] * F(i, j);
			B += I(_i + i, _j + j)[2] * F(i, j);
		}

	if (R>255.0) R = 255.0;
	if (G>255.0) G = 255.0;
	if (B>255.0) B = 255.0;
	if (R<0.0) R = 0.0;
	if (G<0.0) G = 0.0;
	if (B<0.0) B = 0.0;

	return cv::Vec3b{ (uchar)R, (uchar)G, (uchar)B };
}

cv::Mat Image::conv(const cv::Mat & I, const cv::Mat & M) {
	cv::Mat_<cv::Vec3b> O = I;
	for (int i = M.cols / 2; i < I.rows - M.rows / 2; ++i)
		for (int j = M.cols / 2; j < I.cols - M.cols / 2; ++j) {
			O(i, j) = mult(I, M, i, j);
		}

	return O;
}

std::vector<Segment> Image::getSegments(const cv::Mat & I) {
	cv::Mat_<bool> I_ = I;
	cv::Mat_<bool> visited(I.rows, I.cols, false);
	std::vector<Segment> segments;

	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			if (!visited(i, j)) {
				if (I_(i, j)) {
					// New segment detected - extract it
					Segment seg(I_, visited);
					seg.extract(i, j);

					if (seg.getPixelsCount() > SEGMENT_MIN_PIXELS) {
						segments.push_back(std::move(seg));
					}
				}
			}
		}
	}

	return segments;
}

void Image::renderResults(std::vector<Segment>& segs, const std::string & segType, const cv::Vec3b & bbColor) {
	int id = 1;
	for (auto &s : segs) {
		auto bb = s.getBoundingBox();
		auto w3 = s.getCoeffW3();
		auto w6 = s.getCoeffW6();
		auto m1 = s.getCoeffM1();
		auto m2 = s.getCoeffM2();
		auto m3 = s.getCoeffM3();
		auto m4 = s.getCoeffM4();
		auto m6 = s.getCoeffM6();
		auto m7 = s.getCoeffM7();

		auto text = "Segment " + segType + " #" + std::to_string(id) + ":" +
			"\n\tW3: " + std::to_string(w3) +
			"\n\tW6: " + std::to_string(w6) +
			"\n\tM1: " + std::to_string(m1) +
			"\n\tM2: " + std::to_string(m2) +
			"\n\tM3: " + std::to_string(m3) +
			"\n\tM4: " + std::to_string(m4) +
			"\n\tM6: " + std::to_string(m6) +
			"\n\tM7: " + std::to_string(m7) + "\n";
		std::cout << text << std::flush;

		cv::rectangle(image_, bb, bbColor, BOUNDINGBOX_THICKNESS);
		cv::putText(image_, "#" + std::to_string(id), bb.tl() - cv::Point(0, BOUNDINGBOX_THICKNESS), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(0));

		++id;
	}
}

void Image::matchSegs(std::vector<Segment>& feds, std::vector<Segment>& exs) {
	for (auto &f : feds) {
		auto fPs = f.getPoints();
		for (auto &e : exs) {
			auto ePs = e.getPoints();

			int matches = 0;
			for (auto &fP : fPs) {
				for (auto &eP : ePs) {
					int dX = fP.x - eP.x;
					int dY = fP.y - eP.y;
					if (dX*dX + dY * dY < GROUP_MAX_DIST*GROUP_MAX_DIST)
						++matches;
				}
			}

			if (matches >= 2) {
				// Segments matched
				// Draw a bounding box
				auto fTL = f.getB1();
				auto eTL = e.getB1();
				auto fBR = f.getB3();
				auto eBR = e.getB3();

				auto rTL = cv::Point(std::min(fTL.x, eTL.x), std::min(fTL.y, eTL.y));
				auto rBR = cv::Point(std::max(fBR.x, eBR.x), std::max(fBR.y, eBR.y));

				cv::Rect r(rTL, rBR);
				cv::rectangle(image_, r, GROUP_COLOR, GROUP_THICKNESS);	
			}
		}
	}
}
