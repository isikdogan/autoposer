This is a program that automatically corrects images of the participants in video conferences. 
It corrects your posture if you’re leaning and puts a smile on your face.

The program consists of three main stages: initialization, posture correction, and ideal image superimposition. In the initialization step, the initial location of the face is detected by using cascade classifiers and the ideal pose of the user is captured. The user is assumed to be sitting up straight and giving an ideal pose. For posture correction, the input image is first thresholded in YCbCr color space for skin color detection. Then, the skin-colored artifacts in the background are masked out using an elliptical mask around the face that detected in the previous frame. Finally, the main orientation of the head is calculated by using image moments, and the image is rotated to the opposite direction to zero out the leaning angle. In the ideal image superimposition step, the ideal face is aligned with the current face and superimposed on the current frame. For a seamless blending, I use a circular mask with smoothed edges as an alpha mask.

This was a class project for EE 371R Image and Video Processing Class, Spring 2013.

For more information check out this blog post: [AutoPoser: automatic image correction for video conferences](http://www.isikdogan.com/blog/autoposer-automatic-image-correction-for-video-conferences.html) 
