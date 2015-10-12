#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define SMOOTHER_SIZE 25
#define NUM_FIRST_FRAMES 50
#define FACE_MARGINS 10
#define CROP_RATE 0.75
#define PI 3.14159265359
#define BORDER_EXPAND 100
#define IS_INITIAL_FRAME (firstFrameCounter > 0)
#define CLASSIFIER "lbpcascade_frontalface.xml"
#define CAMERA_RESOLUTION Size(320, 240)

using namespace cv;

class VideoCorrect
{
public:
	VideoCorrect(void);
	~VideoCorrect(void);
	void correctImage(Mat &inputFrame, Mat &outputFrame, bool developerMode);
	int Cb;
	int Cr;
	int replaceFace;

private:
	float rotationBuffer[SMOOTHER_SIZE];
	float sizeBuffer[SMOOTHER_SIZE];
	int firstFrameCounter;
	int bufferCounter;
	double smoothAngle;
	double smoothSize;
	double angle;
	Size prevSize;
	Rect lastFace;
	Rect face;
	Rect roi;
	Mat img; //The current frame that being processed
	Mat bestImg; //Best pose
	Mat head;
	Mat ycbcr;
	Mat bw; //Thresholded image
	Mat G; //Gaussian mask
	Moments m;
	Point center;
	CascadeClassifier face_cascade;

	/* Private Functions */
	Rect detectFaces(Mat& img);
	void rotateImage(Mat& source, double angle);
	Point rotatePoint(Point p, Point origin, double angle);
	Rect getROI(Mat& img, Point center);
	void overlayImage(Mat& img, Mat& bestImg, Point center, double width);
};

