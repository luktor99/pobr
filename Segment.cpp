#include "Segment.h"
#include "Settings.h"

void Segment::extract(int i, int j) {
	struct Coord { int i, j; };
	std::queue<Coord> queue;
	queue.push({ i, j });

	while (!queue.empty()) {
		auto pixel = queue.front();
		queue.pop();
		auto pi = pixel.i;
		auto pj = pixel.j;

		if (!visited_(pi, pj)) {
			visited_(pi, pj) = true;

			if (image_(pi, pj)) {
				addPixel(pi, pj);

				// Try extracting pixel's neighbours
				if (coordsOk(pi, pj - 1))
					queue.push({ pi, pj - 1 });
				if (coordsOk(pi, pj + 1))
					queue.push({ pi, pj + 1 });
				if (coordsOk(pi - 1, pj))
					queue.push({ pi - 1, pj });
				if (coordsOk(pi + 1, pj))
					queue.push({ pi + 1, pj });
			}
		}
	}

	updateBoundingBox();
	updateMassCenter();
}

int Segment::getPixelsCount() {
	return counter_;
}

cv::Mat_<bool> Segment::getCropped() {
	return cropped_;
}

cv::Rect & Segment::getBoundingBox() {
	return boundingBox_;
}

double Segment::getCoeffW3() {
	double coeff = ((double)perimeter()) / (2.0 * sqrt(CV_PI * area())) - 1.0;
	return coeff;
}

double Segment::getCoeffW6() {
	int n = 0;
	double sumd = 0.0;
	double sumd2 = 0.0;

	for (int i = 0; i < cropped_.rows; ++i) {
		for (int j = 0; j < cropped_.cols; ++j) {
			if (!cropped_(i, j))
				continue;
			if (isEdge(i, j)) {
				double dI = i - iC_;
				double dJ = j - jC_;
				double d2 = dI * dI + dJ * dJ;
				double d = std::sqrt(d2);

				sumd += d;
				sumd2 += d2;
				++n;
			}
		}
	}

	return std::sqrt((sumd * sumd) / (n * sumd2 - 1.0));
}

double Segment::getCoeffM1() {
	return (m(2, 0, iC_, jC_) + m(0, 2, iC_, jC_)) / pow(m00_, 2.0);
}

double Segment::getCoeffM2() {
	return (pow(m(2, 0, iC_, jC_) - m(0, 2, iC_, jC_), 2.0) + 4.0*pow(m(1, 1, iC_, jC_), 2.0)) / pow(m00_, 4.0);
}

double Segment::getCoeffM3() {
	return (pow(m(3, 0, iC_, jC_) - 3.0*m(1, 2, iC_, jC_), 2.0) + pow(3.0*m(2, 1, iC_, jC_) - m(0, 3, iC_, jC_), 2.0)) / pow(m00_, 5.0);
}

double Segment::getCoeffM4() {
	return (pow(m(3, 0, iC_, jC_) + m(1, 2, iC_, jC_), 2.0) + pow(m(2, 1, iC_, jC_) - m(0, 3, iC_, jC_), 2.0)) / pow(m00_, 5.0);
}

double Segment::getCoeffM5() {
	return ((m(3, 0, iC_, jC_) - 3.0*m(1, 2, iC_, jC_)) * (m(3, 0, iC_, jC_) + m(1, 2, iC_, jC_)) *
		(pow(m(3, 0, iC_, jC_) + m(1, 2, iC_, jC_), 2.0) - 3.0*pow(m(2, 1, iC_, jC_) + m(0, 3, iC_, jC_), 2.0)) +
		(3.0*m(2, 1, iC_, jC_) - m(0, 3, iC_, jC_)) * (m(2, 1, iC_, jC_) + m(0, 3, iC_, jC_)) *
		(3.0*pow(m(3, 0, iC_, jC_) + m(1, 2, iC_, jC_), 2.0) - pow(m(2, 1, iC_, jC_) + m(0, 3, iC_, jC_), 2.0))) / pow(m00_, 10.0);
}

double Segment::getCoeffM6() {
	return ((m(2, 0, iC_, jC_) - m(0, 2, iC_, jC_)) * (pow(m(3, 0, iC_, jC_) + m(1, 2, iC_, jC_), 2.0) - pow(m(2, 1, iC_, jC_) + m(0, 3, iC_, jC_), 2.0)) +
		4.0*m(1, 1, iC_, jC_) * (m(3, 0, iC_, jC_) + m(1, 2, iC_, jC_)) * (m(2, 1, iC_, jC_) + m(0, 3, iC_, jC_))) / pow(m00_, 7.0);
}

double Segment::getCoeffM7() {
	return (m(2, 0, iC_, jC_)*m(0, 2, iC_, jC_) - pow(m(1, 1, iC_, jC_), 2.0)) / pow(m00_, 4.0);
}

cv::Point Segment::getB1() {
	return boundingBox_.tl();
}

cv::Point Segment::getB2() {
	return boundingBox_.tl() + cv::Point(0, boundingBox_.height);
}

cv::Point Segment::getB3() {
	return boundingBox_.br();
}

cv::Point Segment::getB4() {
	return boundingBox_.tl() + cv::Point(boundingBox_.width, 0);
}

cv::Point Segment::getP1() {
	for (int i = 0; i < cropped_.rows; ++i)
		if (cropped_(i, 0))
			return boundingBox_.tl() + cv::Point(0, i);

	return getB1();
}

cv::Point Segment::getP2() {
	for (int j = 0; j < cropped_.cols; ++j)
		if (cropped_(cropped_.rows - 1, j))
			return boundingBox_.tl() + cv::Point(j, cropped_.rows - 1);

	return getB2();
}

cv::Point Segment::getP3() {
	for (int i = cropped_.rows - 1; i >= 0; --i)
		if (cropped_(i, cropped_.cols - 1))
			return boundingBox_.tl() + cv::Point(cropped_.cols - 1, i);

	return getB3();
}

cv::Point Segment::getP4() {
	for (int j = cropped_.cols - 1; j >= 0; --j)
		if (cropped_(0, j))
			return boundingBox_.tl() + cv::Point(j, 0);

	return getB4();
}

std::vector<cv::Point> Segment::getPoints() {
	return { getB1(), getB2(), getB3(), getB4(), getP1(), getP2(), getP3(), getP4() };
}

bool Segment::coordsOk(int i, int j) {
	return (i >= 0) && (i < image_.rows) && (j >= 0) && (j < image_.cols);
}

void Segment::addPixel(int i, int j) {
	segment_(i, j) = true;
	++counter_;
}

void Segment::updateBoundingBox() {
	int i_min = segment_.rows;
	int j_min = segment_.cols;
	int i_max = 0;
	int j_max = 0;

	for (int i = 0; i < segment_.rows; ++i)
		for (int j = 0; j < segment_.cols; ++j) {
			auto pixel = segment_(i, j);
			if (pixel) {
				if (i < i_min)
					i_min = i;
				if (j < j_min)
					j_min = j;
				if (i > i_max)
					i_max = i;
				if (j > j_max)
					j_max = j;
			}

		}

	boundingBox_ = cv::Rect{ j_min, i_min, j_max - j_min, i_max - i_min };
	cropped_ = segment_(boundingBox_);
}

int Segment::perimeter() {
	int perimeter = 0;
	for (int i = 0; i < cropped_.rows; ++i)
		for (int j = 0; j < cropped_.cols; ++j) {
			if (!cropped_(i, j))
				continue;
			if (isEdge(i, j))
				++perimeter;
		}

	return perimeter;
}

double Segment::m(int p, int q, double ic, double jc) {
	double momentum = 0;
	for (int i = 0; i < cropped_.rows; ++i)
		for (int j = 0; j < cropped_.cols; ++j) {
			if (cropped_(i, j))
				momentum += pow(i - ic, p)*pow(j - jc, q);
		}

	return momentum;
}

int Segment::area() {
	return static_cast<int>(m(0, 0));
}

void Segment::updateMassCenter() {
	m00_ = m(0, 0);
	iC_ = m(1, 0) / m00_;
	jC_ = m(0, 1) / m00_;
}

bool Segment::isEdge(int i, int j) {
	bool isEdge = i == 0 || j == 0 || i == cropped_.rows - 1 || j == cropped_.cols - 1;
	if (!isEdge) {
		isEdge = !cropped_(i + 1, j) || !cropped_(i - 1, j) || !cropped_(i, j + 1) || !cropped_(i, j - 1) ||
			!cropped_(i - 1, j - 1) || !cropped_(i - 1, j + 1) || !cropped_(i + 1, j - 1) || !cropped_(i + 1, j + 1);
	}

	return isEdge;
}
