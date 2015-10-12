#include "VideoCorrect.h"


VideoCorrect::VideoCorrect(void)
{
	this->Cb = 30;
	this->Cr = 20;
	this->firstFrameCounter = NUM_FIRST_FRAMES;
	this->bufferCounter = 0;
	this->prevSize = Size(0,0);
	this->smoothAngle = 0;
	this->smoothSize = 0;
	this->replaceFace = 1;

	//Initialize smoothing buffers
	for(int i = 0; i < SMOOTHER_SIZE; i++){
		this->rotationBuffer[i] = 0;
		this->sizeBuffer[i] = 0;
	}

	//Load cascade classifier training file
	face_cascade.load(CLASSIFIER);

	//Load Gaussian Kernel
	G = imread("gaussianMask.png");
	cvtColor(G, G, CV_RGB2GRAY);
	G.convertTo(G, CV_64F);
	normalize(G, G, 0, 1, cv::NORM_MINMAX);
}

VideoCorrect::~VideoCorrect(void)
{
}

void VideoCorrect::correctImage(Mat& inputFrame, Mat& outputFrame, bool developerMode){
	
	resize(inputFrame, inputFrame, CAMERA_RESOLUTION);
	inputFrame.copyTo(img);

	//Convert to YCbCr color space
	cvtColor(img, ycbcr, CV_BGR2YCrCb);

	//Skin color thresholding
	inRange(ycbcr, Scalar(0, 150 - Cr, 100 - Cb), Scalar(255, 150 + Cr, 100 + Cb), bw);

	if(IS_INITIAL_FRAME){
		face = detectFaces(img);
		if(face.x != 0){
			lastFace = face;
		}
		else{
			outputFrame = img;
			return;
		}
		prevSize = Size(face.width/2, face.height/2);
		head = Mat::zeros(bw.rows, bw.cols, bw.type());
		ellipse(head, Point(face.x + face.width/2, face.y + face.height/2), prevSize, 0, 0, 360, Scalar(255,255,255,0), -1, 8, 0);
		if(face.x > 0 && face.y > 0 && face.width > 0 && face.height > 0 
			&& (face.x + face.width) < img.cols && (face.y + face.height) < img.rows){
			img(face).copyTo(bestImg);
		}
		putText(img, "Give your best pose!", Point(face.x, face.y), CV_FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255,0), 1, CV_AA);
	}

	firstFrameCounter--;

	if(face.x == 0) //missing face prevention
		face = lastFace;

	//Mask the background out
	bw &= head;

	//Compute more accurate image moments after background removal
	m = moments(bw, true);
	angle = (atan((2*m.nu11)/(m.nu20-m.nu02))/2)*180/PI;
	center = Point(m.m10/m.m00,m.m01/m.m00);

	//Smooth rotation (running average)
	bufferCounter++;
	rotationBuffer[ bufferCounter % SMOOTHER_SIZE ] = angle;
	smoothAngle += (angle - rotationBuffer[(bufferCounter + 1) % SMOOTHER_SIZE]) / SMOOTHER_SIZE;

	//Expand borders
	copyMakeBorder( img, img, BORDER_EXPAND, BORDER_EXPAND, BORDER_EXPAND, BORDER_EXPAND, 
					BORDER_REPLICATE, Scalar(255,255,255,0));

	if(!IS_INITIAL_FRAME){
		//Rotate the image to correct the leaning angle
		rotateImage(img, smoothAngle);
	
		//After rotation detect faces
		face = detectFaces(img);
		if(face.x != 0)
			lastFace = face;

		//Create background mask around the face
		head = Mat::zeros(bw.rows, bw.cols, bw.type());
		ellipse(head, Point(face.x - BORDER_EXPAND + face.width/2, face.y -BORDER_EXPAND + face.height/2),
					  prevSize, 0, 0, 360, Scalar(255,255,255,0), -1, 8, 0);

		//Draw a rectangle around the face
		//rectangle(img, face, Scalar(255,255,255,0), 1, 8, 0);

		//Overlay the ideal pose
		if(replaceFace && center.x > 0 && center.y > 0){
			center = Point(face.x + face.width/2, face.y + face.width/2);
			overlayImage(img, bestImg, center, smoothSize);
		}

	} else{
		face.x += BORDER_EXPAND; //position alignment after border expansion (not necessary if we detect the face after expansion)
		face.y += BORDER_EXPAND;
	}
	
	//Smooth ideal image size (running average)
	sizeBuffer[ bufferCounter % SMOOTHER_SIZE ] = face.width;
	smoothSize += (face.width - sizeBuffer[(bufferCounter + 1) % SMOOTHER_SIZE]) / SMOOTHER_SIZE;

	//Get ROI
	center = Point(face.x + face.width/2, face.y + face.width/2);
	roi = getROI(img, center);
	if(roi.x > 0 && roi.y > 0 && roi.width > 0 && roi.height > 0 
		&& (roi.x + roi.width) < img.cols && (roi.y + roi.height) < img.rows){
		img = img(roi);
	}

	//Resize the final image
	resize(img, img, CAMERA_RESOLUTION);

	if(developerMode){

		Mat developerScreen(img.rows, 
							img.cols + 
							inputFrame.cols +
							bw.cols, CV_8UC3);

		Mat left(developerScreen, Rect(0, 0, img.size().width, img.size().height));
		img.copyTo(left);

		Mat center(developerScreen, Rect(img.cols, 0, inputFrame.cols, inputFrame.rows));
		inputFrame.copyTo(center);

		cvtColor(bw, bw, CV_GRAY2BGR);
		Mat right(developerScreen, Rect(img.size().width + inputFrame.size().width, 0, bw.size().width, bw.size().height));
		bw.copyTo(right);

		Mat rightmost(developerScreen, Rect(img.size().width + inputFrame.size().width + bw.size().width - bestImg.size().width, 0,
											bestImg.size().width, bestImg.size().height));
		bestImg.copyTo(rightmost);

		outputFrame = developerScreen;
	}
	else{
		outputFrame = img;
	}
}

Rect VideoCorrect::getROI(Mat& img, Point center){

	double roiWidth = ( img.size().width - 2*BORDER_EXPAND) * CROP_RATE;
	double roiHeight = ( img.size().height - 2*BORDER_EXPAND) * CROP_RATE;

	//boundary check
	if(center.x + roiWidth/2 > img.size().width){
		center.x = img.size().width - roiWidth/2;
	}
	if(center.y + roiHeight/2 > img.size().height){
		center.y = img.size().height - roiHeight/2;
	}
	
	return Rect(center.x - roiWidth/2, center.y - roiHeight/2, roiWidth, roiHeight);
}

void VideoCorrect::rotateImage(Mat& source, double angle)
{
    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, source, rot_mat, source.size(), 1, 0, Scalar(255,255,255));
}

Rect VideoCorrect::detectFaces(Mat& img)
{   
	vector<Rect> faces;
	face_cascade.detectMultiScale(img, faces, 1.1, 3, 3, Size(50, 50), Size(250, 250) );

	int maxArea = 0;
	Rect maxFace;

	for( vector<Rect>::const_iterator face = faces.begin(); face != faces.end(); face++ )
    {
		if(face->area() > maxArea){
			maxArea = face->area();
			maxFace = *face;
		}
    }

	//Expand the face area if any face found
	if(maxFace.x != 0){
		maxFace.width += 2*FACE_MARGINS;
		maxFace.height += 2*FACE_MARGINS;
		maxFace.x -= FACE_MARGINS;
		maxFace.y -= FACE_MARGINS;
	}

	return maxFace;
}

void VideoCorrect::overlayImage(Mat& img, Mat& bestImg, Point center, double width)
{  
	//scale best image
	Mat resizedBestImg;
	double fx = width/(bestImg.cols);
	if(fx > 0.25 && fx < 5)
		resize(bestImg, resizedBestImg, Size(fx * bestImg.cols, fx * bestImg.rows), 0, 0, 1);
	
	//Gaussian kernel
	resize(G, G, Size(resizedBestImg.cols, resizedBestImg.rows), 0, 0, 1);
	
	//face center 
	int x = center.x - resizedBestImg.cols/2;
	int y = center.y - resizedBestImg.rows/2;

	double alpha; //blending parameter

	for(int i = 0; i < resizedBestImg.cols; i++){
		for(int j = 0; j < resizedBestImg.rows; j++){
			for(int c = 0; c < resizedBestImg.channels(); c++){
				if(i >= img.cols || j >= img.rows || (i+y) < 0 || (j+x) < 0) //overflow prevention for the first image
					break;

				alpha = G.at<double>(i,j);

				img.data[img.step[0]*(i+y) + img.step[1]*(j+x) + c] = 
					(1 - alpha) * (img.data[img.step[0]*(i+y) + img.step[1]*(j+x) + c]) +
					alpha * resizedBestImg.data[resizedBestImg.step[0]*i + resizedBestImg.step[1]*j + c];
			}
		}
	}
}