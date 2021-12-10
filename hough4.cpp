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

using namespace cv;
using namespace std;

ifstream fin("dataGTNoEntrySigns/dataNoEntry0gt.txt");
ofstream fout("imageResultsFaces/noentry2facesresults.txt");

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;

/*Global vectors of detected frontal faces and actual ground truths*/
vector<Rect> noentrysign;
vector<Rect> gtnoentrysign;

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

//TODO: Talk about the "anthill"
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
	
	// double rho = 0.0;
	// double radians = 0.0;
	// double directionTheta = 0.0;
	// double directionVal = 0.0;
	// int angleRange = 1;

	// vector<double> rhoValues;
	// vector<double> thetaValues;

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
					//allowing some room for error in the gradient direction can cause some 
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
	imageTH(houghSpaceLines, houghSpaceLines, houghSpaceLines.cols/7);

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

	imwrite("houghSpaceLines.jpg", houghSpaceLines);
}

// int getLineEq(int x, float m, float c) {
// 	return (m * x + c);
// }

void extractLinesFromPeacks(Mat& originalImage, Mat houghSpaceLines, vector<Point>& roThetaLocations) {

	vector<LineParameters> lineCoordinates;
    // vector<Vec2f> parallelLines;
    // bool visited[roThetaLocations.size()];

	 int linesno;
	//vector<Point> roThetaLocations;
	 clearNeighbours(houghSpaceLines, 20, linesno, roThetaLocations);

    // for (int i = 0; i < roThetaLocations.size(); i++) {
    //     visited[i] = false;
    // }

    // for (int i = 0; i < roThetaLocations.size(); i++) {
    //     int rho = roThetaLocations[i].x;
    //     int theta = roThetaLocations[i].y;

    //     // if (!visited[i]) {
    //     //     parallelLines.push_back((rho, theta));
    //     //     visited[i] = true;
    //     // }

    //     for (int s = 0; s < roThetaLocations.size(); s++) {
    //         if (i != s && abs(theta - roThetaLocations[s].y) < 5) {
    //             if (!visited[i]) {
	// 				cout << "eeeeee";
    //                 parallelLines.push_back((roThetaLocations[s].x, roThetaLocations[s].y));
    //                 visited[i] = true;
    //             }
    //         }
    //     }
    // }

	// cout << parallelLines.size() <<"-----";
	// cout << "LINES NO: " << linesno << "\n";

	// for (int i = 0; i < roThetaLocations.size(); i++) {
	// 	visited[i] = false;
	// 	for (int j = 0; j < roThetaLocations.size(); j++) {
	// 		if (abs(roThetaLocations[i].y - roThetaLocations[j].y) < 3 && !visited[j]) {
	// 			parallelLines.push_back((roThetaLocations[j].x, roThetaLocations[j].y));
	// 			visited[j] = true;
	// 		}
	// 	}
	// }


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
	
	retainBigVotes(accumulator, gradientDirection, 30);
	
	int corectRadius;
	getMostSuitableRadius(accumulator, gradientDirection, corectRadius);

	cout << "BEST RADIUS: " << corectRadius << endl;

	getImageHoughSpace(accumulatorImage, gradientDirection, accumulator);

	normalize(accumulatorImage, accumulatorImage, 0, 255, NORM_MINMAX);
	
	//imwrite("hough.jpg", accumulatorImage);
	
	imageTH(accumulatorImage, accumulatorImage, 100);
	
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

	vector<Rect> boxes;
	vector<Mat> frames;

	
	for (int i = 0; i < cp.size(); i++) {
		Point center(cp[i].x, cp[i].y);
		int radius = cp[i].radius;

		Rect boxAroundCenter(center.x - radius, center.y - radius, 2 * radius, 2 * radius);

		rectangle(image, Point(boxAroundCenter.x, boxAroundCenter.y), Point(boxAroundCenter.x + boxAroundCenter.width, boxAroundCenter.y + boxAroundCenter.height), Scalar( 255, 255, 255 ), 2);

		boxes.push_back(boxAroundCenter);
		 
	}


	for (int i = 0; i < boxes.size(); i++) {
		croppedBox = copyImage(boxes[i]);
		frames.push_back(croppedBox);
		imwrite("testCropped.jpg", croppedBox);
	}
	
	Mat dxCropped, dyCropped, gradientMagnitudeCropped, gradientDirectionCropped;

	for (int i = 0; i < frames.size(); i++) {
		Canny(frames[i], frames[i], 50, 200, 3);
		vector<Vec2f> lines;
		 HoughLines(frames[i], lines, 30, CV_PI/180, 100, 0, 0);
		 cout << "linii din crop" << lines.size();
		 lines.clear();
		 imwrite("evidenceCrop.jpg", frames[i]);
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
	}
	
}


/** @function detectAndDisplay */
/*void detectAndDisplay( Mat frame )
{
    int x, y, width, height, bottomrightx, bottomrighty;

	//std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, noentrysign, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << noentrysign.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < noentrysign.size(); i++ )
	{
		rectangle(frame, Point(noentrysign[i].x, noentrysign[i].y), Point(noentrysign[i].x + noentrysign[i].width, noentrysign[i].y + noentrysign[i].height), Scalar( 0, 255, 0 ), 2);
	}
		//5. Draw red box around ground truth. Read the ground truth coordinates from txts.
		// Each line contains the top left coordinates of the ground truth box
		// followed by the width and height of the box.
    while(fin >> x >> y >> width >> height) {
		bottomrightx = x + width;
		bottomrighty = y + height;
		rectangle(frame, Point(x, y), Point(bottomrightx, bottomrighty), Scalar( 0, 0, 255 ), 2);
		gtnoentrysign.push_back(Rect(x, y, width, height));
	}

    fin.close();
}*/



int sumRadius[445][545];
int main(int argc, char** argv) {
	int*** ipppArr;
	//int dim1 = 342, dim2 = 442, dim3 = 70; //coins1
	// int dim1 = 519, dim2 = 493, dim3 = 101; //coins2
	// int dim1 = 319, dim2 = 361, dim3 = 101; //coins3
	 int dim1 = 1001, dim2 = 1001, dim3 = 500; // NoEntry0.bmp
	int i, j, k;
	ipppArr = malloc3dArray(dim1, dim2, dim3);
	for (i = 0; i < dim1; ++i)
		for (j = 0; j < dim2; ++j)
			for (k = 0; k < dim3; ++k)
				ipppArr[i][j][k] = 0;
	// if (argc != 2) {
	// 	cout << "Expecting a image file to be passed to program" << endl;
	// 	return -1;
	// }
	Mat image;
    image = imread("No_entry/NoEntry0.bmp", 1);
	//image = imread("No_entry/NoEntry0.bmp");
	//image = imread("no_entry.jpg", 1);

	// image = imread(argv[1], 1);
	// if (image.empty()) {
	// 	cout << "Not a valid image file!" << endl;
	// 	return -1;
	// }
	Mat gray_image;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);
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
	//threshold(gradientMagnitude, gradientMagnitude, 70, 255, THRESH_BINARY);
	imwrite("dx.jpg", dx);
	imwrite("dy.jpg", dy);
	imwrite("magnitude.jpg", gradientMagnitude);
	imwrite("direction.jpg", gradientDirection);
	hough(gradientDirection, gradientMagnitude, ipppArr, image);
	imwrite("CircleDetections/detected.png", image);

	Mat houghSpaceLines;
	getHoughLinesSpace(gradientMagnitude, gradientDirection, gray_image, 
						thetaLocations, roLocations, houghSpaceLines, 5);
	//cout << thetaLocations.size();
	imwrite("houghSpaceLines.jpg", houghSpaceLines);

	extractLinesFromPeacks(image, houghSpaceLines, roThetaLoc);
	cout << "LINII: " << roThetaLoc.size();
	imwrite("evidence.jpg", image);

	return 0;
}