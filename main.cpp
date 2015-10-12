#include <stdio.h>

#include "VideoCorrect.h"
#define APPNAME "AutoPoser v0.1"

using namespace cv;

int main()
{
    Mat img;
    VideoCapture cap(0);
	VideoCorrect vc;

	namedWindow( APPNAME, 1 );

	createTrackbar( "Cb Range", APPNAME, &vc.Cb, 100, NULL );
	createTrackbar( "Cr Range", APPNAME, &vc.Cr, 100, NULL );
	createTrackbar( "Face Swap", APPNAME, &vc.replaceFace, 1, NULL );

    while (cap.read(img))
    {
		Mat correctedImg;

		vc.correctImage(img, correctedImg, true);

		imshow(APPNAME, correctedImg);

		waitKey(1);
    }

    return 0;
}