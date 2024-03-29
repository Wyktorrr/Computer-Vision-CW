#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

ifstream fin("dataGTNoEntrySigns/dataNoEntry0gt.txt");
ofstream fout("Subtask3/noentry0results.txt");

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;

/*Global vectors of detected frontal faces and actual ground truths*/
vector<Rect> signs;
vector<Rect> gtsigns;
vector<Rect> boxes;
vector<Rect> finalBoxes;

const int maxRadius = 150;

vector<double> roLocations;
vector<double> thetaLocations;
vector<Point> roThetaLoc;

typedef struct LineParameters {
	//double m, c, theta, ro;
    Point pointStart, pointFinish;
}LineParameters;

typedef struct CircleParameters {
	int x, y, radius;
}CircleParameters;

//vector<CircleParameters> circles;

int*** malloc3dArray(int dim1, int dim2, int dim3)
{
	int i, j, k;
	int*** array = (int***)malloc(dim1 * sizeof(int**));
	for (i = 0; i < dim1; i++) {
		array[i] = (int**)malloc(dim2 * sizeof(int*));
		for (j = 0; j < dim2; j++) {
			array[i][j] = (int*)malloc(dim3 * sizeof(int));
		}
	}
	return array;
}

void free3d(int ***arr, int y, int x, int r)  {
    for (int i = 0; i < y; i++)  {
        for (int j = 0; j < x; j++)  {
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

void computeDXDY(Mat input, Mat& dx, Mat& dy)
{
    //Set kernel values and find kernels' radius dimensions to perform convolution
	Mat kernelDx = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat kernelDy = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	int kernelRadiusX = (kernelDx.size[0] - 1) / 2;
	int kernelRadiusY = (kernelDx.size[1] - 1) / 2;
	
	// Create a padded version of the input or there will be border effects
	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);
	// Convolution
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			double sumDx = 0.0;
			double sumDy = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
					// Retrieve filter values
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernelDxVal = kernelDx.at<double>(kernelx, kernely);
					double kernelDyVal = kernelDy.at<double>(kernelx, kernely);
					// Perform the convolution
					sumDx += imageval * kernelDxVal;
					sumDy += imageval * kernelDyVal;
				}
			}
			// Store vertical and horizontal lines respectively from the original image
			dx.at<double>(i, j) = (double)sumDx;
			dy.at<double>(i, j) = (double)sumDy;
		}
	}
}

// Create the magnitude image to display edges from the original image
void computeGradientMagnitude(Mat image, Mat dx, Mat dy, Mat& gradientMagnitude) {
	// gradientMagnitude.create(image.rows, image.cols, CV_64F);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			gradientMagnitude.at<double>(y, x) = sqrt(dx.at<double>(y, x) * dx.at<double>(y, x) + dy.at<double>(y, x) * dy.at<double>(y, x));
		}
	}
	normalize(gradientMagnitude, gradientMagnitude, 0, 255, NORM_MINMAX);
}

// Retrieve the direction of the edges. Needed for the creation of the Hough spaces
void computeGradientDirection(Mat image, Mat dx, Mat dy, Mat& gradientDirection) {
	// gradientDirection.create(image.rows, image.cols, CV_64F);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			gradientDirection.at<double>(y, x) = atan2(dy.at<double>(y, x), dx.at<double>(y, x)); 
		}
	}
}

// Allocate number of rows and columns for the images, perform sobel, magnitude and direction
//which are all used for the hough spaces.
void sobel(Mat image, Mat& dx, Mat& dy, Mat& gradientMagnitude, Mat& gradientDirection) {
	dx.create(image.rows, image.cols, CV_64F);
	dy.create(image.rows, image.cols, CV_64F);
	gradientMagnitude.create(image.rows, image.cols, CV_64F);
	gradientDirection.create(image.rows, image.cols, CV_64F);
	computeDXDY(image, dx, dy);
	computeGradientMagnitude(image, dx, dy, gradientMagnitude);
	computeGradientDirection(image, dx, dy, gradientDirection);
}

//Thresholding the values of an image - used to store only strong values in the Hough Spaces
void imageTH(Mat image, Mat& resultTH, double thValue) {
	resultTH.create(image.size(), CV_64F);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			double value = image.at<double>(y, x);
			if (value >= thValue) {
				resultTH.at<double>(y, x) = 255.0;
			}
			else if (value < thValue) {
				resultTH.at<double>(y, x) = 0.0;
			}
		}
	}
}

bool validXY(int x, int y, Mat imag) {
    return (x >= 0 && x < imag.cols) && (y >= 0 && y < imag.rows);
}

int countVotesForEachRadius(int*** acc, int r, int rows, int cols) {
	int votes = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			votes += acc[y][x][r];
		}
	}
	return votes;
}

void createHoughSpaceCircles(Mat gradientDirection, Mat gradientMagnitude, int***& accumulator) {
	for (int y = 0; y < gradientDirection.rows; y++) {
		for (int x = 0; x < gradientDirection.cols; x++) {
			if (gradientMagnitude.at<double>(y, x) > 0) {
				for (int r = 0; r < maxRadius; r++) {
					unsigned int power = 0;
					while(power <= 3) {
						int xcenter = (int)(x + pow(-1, power) * r * cos(gradientDirection.at<double>(y, x)));
						int ycenter = (int)(y + pow(-1, power) * r * sin(gradientDirection.at<double>(y, x)));
						if (validXY(xcenter, ycenter, gradientDirection)) {
	 							accumulator[ycenter][xcenter][r]++;
	 					}
						power++; 
					}
				}
			}
		}
	}
}

void retainBigVotes(int***& accumulator, Mat gradientDirection, unsigned int votesNumber) {
	for (int r = 0; r < maxRadius; r++) {
		for (int y = 0; y < gradientDirection.rows; y++) {
			for (int x = 0; x < gradientDirection.cols; x++) {
				if (accumulator[y][x][r] > votesNumber) {
					accumulator[y][x][r] = 255;
				}
				else {
					accumulator[y][x][r] = 0;
				}
			}
		}
	}
}

//Used when there is guarantee that all the circles have the same radius. 
void getMostSuitableRadius(int*** accumulator, Mat gradientDirection, int& corectRadius) {
	//corectRadius = -1;
	int maxSum = -1;
	for (int r = 0; r < maxRadius; r++) {
		int sum = countVotesForEachRadius(accumulator, r, gradientDirection.rows, gradientDirection.cols);
		if (sum > maxSum) {
			maxSum = sum;
			corectRadius = r;
		}
	}
}

void getImageHoughSpace(Mat& accumulatorImage, Mat gradientDirection, int*** accumulator) {
	accumulatorImage.create(gradientDirection.rows, gradientDirection.cols, CV_64F);
	for (int y = 0; y < gradientDirection.rows; y++) {
		for (int x = 0; x < gradientDirection.cols; x++) {
			for (int r = 0; r < maxRadius; r++) {
				accumulatorImage.at<double>(y, x) += accumulator[y][x][r];
			}
		}
	}
}

//Only keep one value from the "anthill" of pixels that the hough space creates to extract circles accurately.
void extractNoOfCirclesAndTheirCenteres(Mat gradientDirection, Mat accImag, int& circlesCount, vector<Point>& circlesCenters, int minDist) {
	
	circlesCount = 0;
	
	for (int y = 0; y < gradientDirection.rows; y++) {
		for (int x = 0; x < gradientDirection.cols; x++) {
			if (accImag.at<double>(y, x) == 255) {
				
				Point p;
				
				p.x = x;
				p.y = y;

				bool flag = true;
				for (int i = 0; i < circlesCenters.size(); i++) {
					if (abs(p.x - circlesCenters[i].x) < minDist && abs(p.y - circlesCenters[i].y) < minDist) { 
						flag = false;
						break;
					}
				}
				if (flag) {
					circlesCount++;
					circlesCenters.push_back(p);
				}
			}
		}
	}
}

void getCirclesParamsAndDraw(vector<Point> circlesCenters, int*** accumulator, 
								Mat& image, CircleParameters circleparams, vector<CircleParameters>& circles) {
	int count = 0;
	
	for (int i = 0; i < circlesCenters.size(); i++) {
		int maxi = -1;
		int suitableRadius = -1;
		Point p = circlesCenters[i];
		for (int r = 0; r < maxRadius; r++) {
			if (accumulator[p.y][p.x][r] > maxi) {
				maxi = accumulator[p.y][p.x][r];
				suitableRadius = r;
			}
		}
		
		circle(image, p, suitableRadius, Scalar(255, 0, 0), 3, CV_AA);

		for (int r = 0; r < maxRadius; r++) {
			if (accumulator[p.y][p.x][r] == maxi && suitableRadius == r) {
				count++;
				circleparams.x = p.x;
				circleparams.y = p.y;
				circleparams.radius = r;
				circles.push_back(circleparams);
			}
		}
	}
}

bool checkLineDim(int ro, int diag) {
	return ((ro > 0) && (ro < diag));
}

void clearNeighbours(Mat& image, int minDist, int& linesnumber, vector<Point>& pixels) {

	linesnumber = 0;
	
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (image.at<double>(y, x) == 255) {
				
				Point p;
				
				p.x = x;
				p.y = y;

				bool flag = true;
				for (int j = 0; j < pixels.size(); j++) {
					if (abs(p.x - pixels[j].x) < minDist && abs(p.y - pixels[j].y < minDist)) {
						flag = false;
						break;
					}
					
				}
		
				if (flag) {
					linesnumber++;
					pixels.push_back(p);
				}
			}
		}
	}
}


void getHoughLinesSpace(Mat magnitudeImage, Mat gradDirection, Mat image, vector<double>& thetaLocations, 
							vector<double>& roLocations, Mat& houghSpaceLines, unsigned int deltaTheta) {

	const double ro = sqrt((magnitudeImage.rows * magnitudeImage.rows) + (magnitudeImage.cols * magnitudeImage.cols));
	cout << "ro: " << ro << "\n";
	houghSpaceLines.create(cvRound(ro), 360, CV_64F);
	houghSpaceLines = Scalar(0, 0, 0);

	for (int x = 0; x < magnitudeImage.rows; x++) {
		for (int y = 0; y < magnitudeImage.cols; y++) {
			//Check if there is an edge at the given position
			if (magnitudeImage.at<double>(x, y) == 255) {
				//retrieve angle value in radians from direction matrix for this edge
				double thetaRad = gradDirection.at<double>(x, y);
				//convert value of theta into degrees
				int thetaDgr = (thetaRad * 180) / M_PI + 180;  //180 needs to be added to keep the values in range 0 - 360
				//cout << "thetaDgr: " << thetaDgr << "\n";
				//for (int i = thetaDgr - deltaTheta; i <= thetaDgr + deltaTheta; i++) {
					for (int i = 0; i < 360; i++) {	
					//I have also tested my system with allowing some room for error in the gradient direction can cause some 
					//very small or very big values for theta to lie outside the range, thus 
					//some form of normalization is needed
					//double normDgr = (i + 360) % 360;
					double normDgr = i;
					//in order to write the equation of the line in polar coordinates, we still need to work with radians
					//double thetaVal = (normDgr - 180) * M_PI / 180;
					double thetaVal = normDgr * M_PI / 180.0;
					//equation of the line in polar coordinates
					float currentRo = (y * cos(thetaVal)) + (x * sin(thetaVal));
					//cout << "currentRo: " << currentRo << "\n";
					if (checkLineDim(currentRo, ro)) {
						houghSpaceLines.at<double>(currentRo, normDgr)++; //here I am gathering "votes" for the coordinates of the peacks
						//cout << "hough space: " << houghSpaceLines.at<double>(ro, normDgr) << " ";
					}

				}
				
			}

		}	
	}

	//normalize the image
	//normalize(houghSpaceLines, houghSpaceLines, 0, 255, NORM_MINMAX);


	//imageTH(houghSpaceLines, houghSpaceLines, houghSpaceLines.cols/2);
	//imageTH(houghSpaceLines, houghSpaceLines, houghSpaceLines.cols/2 - 20);
	imageTH(houghSpaceLines, houghSpaceLines, houghSpaceLines.cols/2);

	//Pursue by iterating through the thresholded image and store the ro-theta locations 
	// for (int y = 0; y < houghSpaceLines.rows; y++) {
	// 	for (int x = 0; x < houghSpaceLines.cols; x++) {
	// 		if (houghSpaceLines.at<double>(y, x) == 255) {
	// 			roLocations.push_back(y);
	// 			cout << "ro coord: " << y << "\n";
	// 			thetaLocations.push_back(x);
	// 			cout << "theta coord: " << x << "\n";
	// 		}
	// 	}
	// }

	imwrite("houghSpaceLines0.jpg", houghSpaceLines);
}

void extractLinesFromPeacks(Mat& originalImage, Mat houghSpaceLines, vector<Point>& roThetaLocations) {

	vector<LineParameters> lineCoordinates;

	 int linesno;
	//vector<Point> roThetaLocations;
	 clearNeighbours(houghSpaceLines, 20, linesno, roThetaLocations);

	const int xStart = 0, xEnd = originalImage.cols, xHeight = originalImage.rows;
	int yStart, yEnd;

	//cout << roThetaLocations.size();

	for (int i = 0; i < roThetaLocations.size(); i++) {

		//Point point1, point2;
		double theta = roThetaLocations[i].x;
		theta = theta * M_PI / 180.0;
		double rho = roThetaLocations[i].y;
     	double m, c;
		//double radians = theta * (M_PI/ 180);

		//cout << radians;

		m = - cos(theta) / sin(theta);
		//cout << "m :" << m << "\n";
		c = rho / sin(theta);
		//cout << "c: " << c << "\n";

		if (theta == 0) {

			Point point1(rho, rho);
			Point point2(1, xHeight);
			line(originalImage, point1, point2, (0, 0, 255), 2);
		} else {
			
			yStart = m * xStart + c;
			//cout << "YSTART: " << yStart << "\n";
			yEnd = m * xEnd + c;
			//cout << "YEND: " << yStart << "\n";
			Point p1(xStart, yStart);
			Point p2(xEnd, yEnd);
		
			line(originalImage, p1, p2, (0, 0, 255), 2);
		}
	}
}


void hough(Mat gradientDirection, Mat gradientMagnitude, int***& accumulator, Mat& image) {
	Mat accumulatorImage;

	createHoughSpaceCircles(gradientDirection, gradientMagnitude, accumulator);
	
	retainBigVotes(accumulator, gradientDirection, 20);
	
	int corectRadius;
	getMostSuitableRadius(accumulator, gradientDirection, corectRadius);

	cout << "BEST RADIUS: " << corectRadius << endl;

	getImageHoughSpace(accumulatorImage, gradientDirection, accumulator);

	normalize(accumulatorImage, accumulatorImage, 0, 255, NORM_MINMAX);
	
	imageTH(accumulatorImage, accumulatorImage, 100);

    imwrite("Subtask3/houghCircles0.jpg", accumulatorImage);
	
	// IDEA: Mark neighbours as visited
	int circlesCount;
	vector<Point> circlesCenters;
	vector<CircleParameters> cp;
	CircleParameters circleParamsStructure;

	extractNoOfCirclesAndTheirCenteres(gradientDirection, accumulatorImage, circlesCount, circlesCenters, 55);
	cout << "The number of circles found in the image is: " << circlesCount << "\n";

	getCirclesParamsAndDraw(circlesCenters, accumulator, image, circleParamsStructure, cp);

	imwrite("houghTHCircles.jpg", accumulatorImage);
	//imwrite("corectRadius.jpg", corectRadius);
	//circles.clear();

	Mat copyImage;
	copyImage = image.clone();
	Rect boxAroundCenter;
	Mat croppedBox;

	// vector<Rect> boxes;
	vector<Mat> frames;

	
	for (int i = 0; i < cp.size(); i++) {
		Point center(cp[i].x, cp[i].y);
		int radius = cp[i].radius;

		Rect boxAroundCenter(center.x - radius, center.y - radius, 2 * radius, 2 * radius);

		//rectangle(image, Point(boxAroundCenter.x, boxAroundCenter.y), Point(boxAroundCenter.x + boxAroundCenter.width, boxAroundCenter.y + boxAroundCenter.height), Scalar( 255, 255, 255 ), 2);

		boxes.push_back(boxAroundCenter);
		 
	}


	// for (int i = 0; i < boxes.size(); i++) {
	// 	croppedBox = copyImage(boxes[i]);
	// 	frames.push_back(croppedBox);
	// 	imwrite("testCropped.jpg", croppedBox);
	// }
	
	//Mat dxCropped, dyCropped, gradientMagnitudeCropped, gradientDirectionCropped;

	// for (int i = 0; i < frames.size(); i++) {
	// 	Canny(frames[i], frames[i], 50, 200, 3);
	// 	vector<Vec2f> lines;
	// 	 HoughLines(frames[i], lines, 30, CV_PI/180, 100, 0, 0);
	// 	 cout << "linii din crop" << lines.size();
	// 	 lines.clear();
	// 	 imwrite("evidenceCrop.jpg", frames[i]);

    /** my implementation of depictiong lines in cropped images **/
		// sobel(frames[i], dxCropped, dyCropped, gradientMagnitudeCropped, gradientDirectionCropped);
		// imageTH(gradientMagnitudeCropped, gradientMagnitudeCropped, 100);
		// imwrite("croppedMagnitude.jpg", gradientMagnitudeCropped);
		// imwrite("croppedDirection.jpg", gradientDirectionCropped);
		// Mat houghSpaceLinesCropped;
		// vector<Point> roThetaLocations;
		// int linesno = 0;
		// vector<Point> numberoflinesandcoord;
		// getHoughLinesSpace(gradientMagnitudeCropped, gradientDirectionCropped, frames[i], thetaLocations, roLocations, houghSpaceLinesCropped, 5);
		// imwrite("houghLinesCropped.jpg", houghSpaceLinesCropped);
		// //clearNeighbours(houghSpaceLinesCropped, 20, linesno, numberoflinesandcoord);
		// extractLinesFromPeacks(frames[i], houghSpaceLinesCropped, roThetaLocations);
		// cout << "------" << roThetaLocations.size();
		// cout << "--------" << numberoflinesandcoord.size() << " " << linesno;
		// imwrite("evidenceCropped.jpg", frames[i]);

		// numberoflinesandcoord.clear();
	//}
	
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    int x, y, width, height, bottomrightx, bottomrighty;

	//std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, signs, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << signs.size() << std::endl;

       // 4. Draw box around faces found
	// for( int i = 0; i < signs.size(); i++ )
	// {
	// 	rectangle(frame, Point(signs[i].x, signs[i].y), Point(signs[i].x + signs[i].width, signs[i].y + signs[i].height), Scalar( 0, 255, 0 ), 2);
	// }
		//5. Draw red box around ground truth. Read the ground truth coordinates from txts.
		// Each line contains the top left coordinates of the ground truth box
		// followed by the width and height of the box.
    while(fin >> x >> y >> width >> height) {
		bottomrightx = x + width;
		bottomrighty = y + height;
		rectangle(frame, Point(x, y), Point(bottomrightx, bottomrighty), Scalar( 0, 0, 255 ), 2);
		gtsigns.push_back(Rect(x, y, width, height));
	}

    fin.close();
}

//Perform Intersection Over Union
float computeIOU( Rect predictedBox, Rect groundTruthBox ) {
	float groundTruthBoxArea, predictedBoxArea, result;

	int intersectionArea, unionArea, xtopleftgt, ytopleftgt, xbottomrightgt, ybottomrightgt,
			xtopleftpred, ytopleftpred, xbottomrightpred, ybottomrightpred;

	xtopleftgt = groundTruthBox.x;
	ytopleftgt = groundTruthBox.y;
	xbottomrightgt = xtopleftgt + groundTruthBox.width;
	ybottomrightgt = ytopleftgt + groundTruthBox.height;

	xtopleftpred = predictedBox.x;
	ytopleftpred = predictedBox.y;
	xbottomrightpred = xtopleftpred + predictedBox.width;
	ybottomrightpred = ytopleftpred + predictedBox.height;


	//Settle assertions and checks regarding overlapping boxes based on the dimensions received 
	//before computing iou.
	if ( (xtopleftgt > xbottomrightgt) || (ytopleftgt > ybottomrightgt) ) 
	    assert("Ground Truth Bounding Box is not correct");

	if ( (xtopleftpred > xbottomrightpred) || (ytopleftpred > ybottomrightpred) )	
		assert("Predicted Bounding Box is not correct");

	//If the ground truth box and predcited box do not overlap then iou=0
	if (xbottomrightgt < xtopleftpred)
		return 0;

	if (ybottomrightgt < ytopleftpred) 
	    return 0;

	if (xtopleftgt > xbottomrightpred)
	    return 0;

	if (ytopleftgt > ybottomrightpred)
		return 0;


	intersectionArea = (groundTruthBox & predictedBox).area();
	//cout << "intersection: "<< intersectionArea << "\n";
	unionArea = (groundTruthBox | predictedBox).area();
	//cout << "union: " << unionArea << "\n";
	
	result = intersectionArea / (float)unionArea;

    //cout << "result: "<<result;
	return result;
}

void keepRectangles(Rect circleBox, Rect violaDetections) {
    
    float result = computeIOU(circleBox, violaDetections);
    if (result >= 0.5) {
        finalBoxes.push_back(circleBox);
    }
}

//Compute TPR and F1-score based on IOU 
void getImageResults(vector<Rect> gtFaces, vector<Rect> signs, float iouThreshold) {
	int truePositives = 0, falsePositives, falseNegatives, detectedSigns, trueNoOfSigns;
	float iou, TPR, accuracy, precision, recall, F1Score;

	trueNoOfSigns = gtsigns.size();
	detectedSigns = signs.size();

	//cout << trueNoOfFaces << " " << detectedFaces << "\n";

	for (int i = 0; i < detectedSigns; i++) {
		for (int j = 0; j < trueNoOfSigns; j++) {
             iou = computeIOU(signs[i], gtsigns[j]);

			 //cout << "iou: " << iou << " ";

			 if (iou > iouThreshold) {
				 truePositives++;
			 }
		}
	}

	cout << "detectedSigns: " << detectedSigns << "trueNoOfSigns: " << trueNoOfSigns << "\n";
	cout << "truePositives: " << truePositives << "\n";

	falsePositives = detectedSigns - truePositives;
	
	falseNegatives = trueNoOfSigns - truePositives;

	cout << "falsePositives: " << falsePositives << "\n";
	cout << "falseNegatives: " << falseNegatives << "\n";

	if (trueNoOfSigns > 0) {
		TPR = (float)truePositives / (float)trueNoOfSigns;
	} else if (trueNoOfSigns == 0) {
		TPR = 0;
	}

	accuracy = (float)truePositives / (float)detectedSigns;
	cout << "accuracy: " << accuracy << "\n";
	precision = (float)truePositives / (float)(truePositives + falsePositives);
	cout << "precision: " << precision << "\n";
	recall = (float)truePositives / (float)(truePositives + falseNegatives);
	cout << "recall: " << recall << "\n";

	F1Score = (float)(2 * precision * recall) / (float)(precision + recall);

	//cout << F1Score << " " << TPR;

	fout << fixed << showpoint;
	fout << setprecision(2);
	fout <<  "TPR: " << TPR << " " << "F1Score: " << F1Score << "\n";

	fout.close();
}	

int main(int argc, char** argv) {
	int*** ipppArr;
	int dim1 = 1001, dim2 = 1001, dim3 = 500; 
	int i, j, k;
	ipppArr = malloc3dArray(dim1, dim2, dim3);
	for (i = 0; i < dim1; ++i)
		for (j = 0; j < dim2; ++j)
			for (k = 0; k < dim3; ++k)
				ipppArr[i][j][k] = 0;
	
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
	Mat gray_image;
	cvtColor(frame, gray_image, COLOR_BGR2GRAY);
	Mat dx;
	Mat dy;
	Mat gradientMagnitude;
	Mat gradientMagnitudeTH;
	Mat gradientDirection;
	// medianBlur(gray_image, gray_image, 5);
	GaussianBlur(gray_image, gray_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
	imwrite("gaussian_blur.png", gray_image);

	sobel(gray_image, dx, dy, gradientMagnitude, gradientDirection);
	// gradImageTH(gradientMagnitude, gradientMagnitudeTH);
	imwrite("magnitude_NO_TH.jpg", gradientMagnitude);
	imageTH(gradientMagnitude, gradientMagnitude, 70);
	
	imwrite("dx.jpg", dx);
	imwrite("dy.jpg", dy);
	imwrite("Subtask3/magnitude0.jpg", gradientMagnitude);
	imwrite("direction.jpg", gradientDirection);
	hough(gradientDirection, gradientMagnitude, ipppArr, frame);
    
   // cout << "-------------" << boxes.size() << "\n";
    imwrite("CircleDetections/detected0.png", frame);
	
       
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	detectAndDisplay( frame );

    cout << "boxs size:" << boxes.size() << "\n";
    cout << "signs zise:" << signs.size() << "\n";
    for (int i = 0; i < boxes.size(); i++) {
        for (int j = 0; j < signs.size(); j++) {
                keepRectangles(boxes[i], signs[j]);
                cout << "------" << "\n";
        }
    }

    cout << "aici";
    cout << finalBoxes.size() << "\n";

    // imwrite("CircleDetections/detected.png", frame);

	getImageResults(gtsigns, finalBoxes, 0.5);

    for (int i = 0; i < finalBoxes.size(); i++) {
        rectangle(frame, Point(finalBoxes[i].x, finalBoxes[i].y), Point(finalBoxes[i].x + finalBoxes[i].width, finalBoxes[i].y + finalBoxes[i].height), Scalar( 0, 255, 0 ), 2);
    }

	// 5. Save Result Image showing both detected and ground truth boxes
	imwrite( "Subtask3/detectedSignsNoEntry0.jpg", frame );


    Mat houghSpaceLines;
	getHoughLinesSpace(gradientMagnitude, gradientDirection, gray_image, 
						thetaLocations, roLocations, houghSpaceLines, 5);
	//cout << thetaLocations.size();
	imwrite("Subtask3/houghSpaceLines0.jpg", houghSpaceLines);

	extractLinesFromPeacks(frame, houghSpaceLines, roThetaLoc);
	//cout << "LINII: " << roThetaLoc.size();
	//imwrite("evidence.jpg", frame);


	return 0;
}