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

using namespace cv;
using namespace std;

const int maxRadius = 150;

vector<double> roLocations;
vector<double> thetaLocations;

typedef struct LineParameters {
	double m, c, theta, ro;
    Point pointStart, pointFinish;
}LineParameters;

typedef struct CircleParameters {
	int x, y, radius;
}CircleParameters;

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
								Mat& image, CircleParameters circleparams, vector<CircleParameters>& cp) {
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
			if (accumulator[p.y][p.x][r] == maxi) {
				circleparams.x = p.x;
				circleparams.y = p.y;
				circleparams.radius = r;
				cp.push_back(circleparams);
			}
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
	
	//imwrite("hough.jpg", accumulatorImage);
	
	imageTH(accumulatorImage, accumulatorImage, 100);
	
	// IDEA: Mark neighbours as visited
	int circlesCount;
	vector<Point> circlesCenters;
	vector<CircleParameters> cp;
	CircleParameters circleParamsStructure;

	extractNoOfCirclesAndTheirCenteres(gradientDirection, accumulatorImage, circlesCount, circlesCenters, 30);
	cout << "The number of circles found in the image is: " << circlesCount << "\n";

	getCirclesParamsAndDraw(circlesCenters, accumulator, image, circleParamsStructure, cp);
	
	imwrite("houghTHCircles.jpg", accumulatorImage);
	//imwrite("corectRadius.jpg", corectRadius);
}

bool checkLineDim(int ro, int diag) {
	return ((ro > 0) && (ro < diag));
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
				for (int i = thetaDgr - deltaTheta; i <= thetaDgr + deltaTheta; i++) {
					//allowing some room for error in the gradient direction can cause some 
					//very small or very big values for theta to lie outside the range, thus 
					//some form of normalization is needed
					int normDgr = (i + 360) % 360;
					//in order to write the equation of the line in polar coordinates, we still need to work with radians
					double thetaVal = (normDgr - 180) * M_PI / 180;
					//equation of the line in polar coordinates
					float currentRo = (x * cos(thetaVal)) + (y * sin(thetaVal));
					//cout << "currentRo: " << currentRo << "\n";
					if (checkLineDim(currentRo, ro)) {
						houghSpaceLines.at<double>(currentRo, normDgr)++; //here I am gathering "votes" for the coordinates of the peacks
						//cout << "hough space: " << houghSpaceLines.at<double>(ro, normDgr) << " ";
					}

				}
				
			}

		}	
	}

	imageTH(houghSpaceLines, houghSpaceLines, 40);

	//normalize the image
	normalize(houghSpaceLines, houghSpaceLines, 0, 255, NORM_MINMAX);

	//Pursue by iterating through the thresholded image and store the ro-theta locations 
	for (int y = 0; y < houghSpaceLines.rows; y++) {
		for (int x = 0; x < houghSpaceLines.cols; x++) {
			if (houghSpaceLines.at<double>(y, x) == 255) {
				roLocations.push_back(y);
				//cout << "ro coord: " << x << "\n";
				thetaLocations.push_back(x);
				//cout << "theta coord: " << y << "\n";
			}
		}
	}

	imwrite("houghSpaceLines.jpg", houghSpaceLines);


	// Mat displayhoughSpace(ro, 360, CV_64F);

    // for (int x = 0; x < ro; x++) {
    //     for (int y = 0; y < 360; y++) {
    //         displayhoughSpace.at<double>(x,y) += houghSpaceLines.at<double>(x, y);  
    //     }
    // }

    // Mat houghSpaceNomalised(ro, 360, CV_64F);
    // normalize(displayhoughSpace, houghSpaceNomalised, 0, 255, NORM_MINMAX);

    // imwrite( "houghLineOuput.jpg", houghSpaceNomalised );

	//normalize the image
	// normalize(houghSpaceLines, houghSpaceLines, 0, 255, NORM_MINMAX);

	//We are keen on finding the very peacks constructed in "houghSpaceLines", the ro-theta coordinates of the lines
	//thus, thresholding is desirable.
	//imageTH(houghSpaceLines, houghSpaceLines, 50);

	//Pursue by iterating through the thresholded image and store the ro-theta locations 
	// for (int y = 0; y < houghSpaceLines.rows; y++) {
	// 	for (int x = 0; x < houghSpaceLines.cols; x++) {
	// 		if (houghSpaceLines.at<double>(y, x) == 255) {
	// 			roLocations.push_back(y);
	// 			cout << "theta coord: " << x;
	// 			thetaLocations.push_back(x);
	// 			cout << "ro coord: " << y;
	// 		}
	// 	}
	// }
	//imwrite("houghSpaceLines.jpg", houghSpaceLines);
}

int getLineEq(int x, float m, float c) {
	return (m * x + c);
}

// imshow(im); %// Show the image
// hold on; %// Hold so we can draw lines
// numLines = numel(rho); %// or numel(theta);

// %// These are constant and never change
// x0 = 1;
// xend = size(im,2); %// Get the width of the image

// %// For each rho,theta pair...
// for idx = 1 : numLines
//     r = rho(idx); th = theta(idx); %// Get rho and theta

//     %// if a vertical line, then draw a vertical line centered at x = r
//     if (th == 0)
//         line([r r], [1 size(im,1)], 'Color', 'blue');
//     else
//         %// Compute starting y coordinate
//         y0 = (-cosd(th)/sind(th))*x0 + (r / sind(th)); %// Note theta in degrees to respect your convention

//         %// Compute ending y coordinate
//         yend = (-cosd(th)/sind(th))*xend + (r / sind(th));

//         %// Draw the line
//         line([x0 xend], [y0 yend], 'Color', 'blue');
//    end
// end

void extractLinesFromPeacks(Mat& originalImage, vector<double> thetaLocations,
							vector<double> roLocations, Mat houghSpaceLines) {
	
	//originalImage.create(houghSpaceLines.rows, houghSpaceLines.cols, CV_64F);

	vector<LineParameters> lineCoordinates;

	//ine(originalImage, Point(5, 5), Point(100, 100), (0, 0, 255), 5);

	//const int xStart = 0, xEnd = originalImage.cols, xHeight = originalImage.rows;
	const int xStart = 0, xEnd = houghSpaceLines.cols, xHeight = houghSpaceLines.rows;
	int yStart, yEnd;

	//cout << thetaLocations.size();
	//cout << roLocations.size();

	for (int i = 0; i < roLocations.size(); i++) {

		//cout << "eeeeeeeeee";
		//Point point1, point2;
		double theta = thetaLocations[i];
		double rho = roLocations[i];
     	double m, c;
		//double radians = theta * (M_PI/ 180);

		//cout << radians;

		m = - cos(theta) / sin(theta);
		cout << "m :" << m << "\n";
		c = rho / sin(theta);
		cout << "c: " << c << "\n";

		if (theta == 0) {
			//cout << "Aici1";
			Point point1(rho, rho);
			Point point2(1, xHeight);
			line(originalImage, point1, point2, (0, 0, 255), 2);
		} else {
			//cout << "Aici2";
			yStart = m * xStart + c;
			cout << "YSTART: " << yStart << "\n";
			yEnd = m * xEnd + c;
			cout << "YEND: " << yStart << "\n";
			Point p1(xStart, xEnd);
			Point p2(yStart, yEnd);
			line(originalImage, p1, p2, (0, 0, 255), 2);
		}
	}


	// for (int ro = 0; ro < houghSpaceLines.rows; ro++) {
	// 	for (int theta = 0; theta < houghSpaceLines.cols; theta++) {
	// 		if (houghSpaceLines.at<double>(ro, theta) == 255) {
	// 			/*Equations documented from OpenCV official documentation*/

	// 			double a = cos(theta), b = sin(theta);
    // 			double x0 = a*ro, y0 = b*ro;
    // 			Point pt1(cvRound(x0 + 1000*(-b)),
    //           	cvRound(y0 + 1000*(a)));
    // 			Point pt2(cvRound(x0 - 1000*(-b)),
    //           	cvRound(y0 - 1000*(a)));
    // 			line( evidenceLines, pt1, pt2, Scalar(0,0,255), 3, 8 );
	// 			// float thetaRad = (theta - 180) * M_PI / 180;

				// float m = - cos(thetaRad) / sin(thetaRad);
                // float c = ro / sin(thetaRad);
				
				// Point pStart(200, getLineEq(200, m, c));
				// Point pFinish(300, getLineEq(300, m, c));

				// line(evidenceLines, pStart, pFinish, Scalar(0, 0, 255), 2);
			//}
		//}
	//}
	//imwrite("arataLinii.jpg", evidenceLines);
}

// void hough(Mat gradImag, int*** &Accumulator) {
// int Radius;
// cv::Mat paddedGradImage;
// cv::copyMakeBorder(gradImag, paddedGradImage,
// 55, 55, 55, 55, cv::BORDER_CONSTANT);
// //cout << paddedGradImage.rows << " " << paddedGradImage.cols;
// for (int y = 0; y < paddedGradImage.rows; y++) {
// for (int x = 0; x < paddedGradImage.cols; x++) {
// if (paddedGradImage.at<uchar>(y, x) == 255) {
// for (int m = y - 50; m <= y + 50; m++) {
// for (int n = x - 50; n <= x + 50; n++) {
// Radius = (int)sqrt((y - m) * (y - m) + (x - n) * (x - n));
// Accumulator[m][n][Radius]++;
// }
// }
// }
// }
// }
// }


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

	extractLinesFromPeacks(image, thetaLocations, roLocations, houghSpaceLines);
	imwrite("evidence.jpg", image);

	return 0;
}