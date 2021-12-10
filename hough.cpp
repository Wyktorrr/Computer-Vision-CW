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

vector<double> roLocations;
vector<double> thetaLocations;

typedef struct LineParameters {
	double m, c, theta, ro;
    Point pointStart, pointFinish;
}LineParameters;

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



// TODO: Make it prettier
void computeDXDY(Mat input, Mat& dx, Mat& dy)
{
	// dx.create(input.rows, input.cols, CV_64F);
	// dy.create(input.rows, input.cols, CV_64F);
	Mat kernelDx = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat kernelDy = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	int kernelRadiusX = (kernelDx.size[0] - 1) / 2;
	int kernelRadiusY = (kernelDx.size[1] - 1) / 2;
	// SET KERNEL VALUES
	// for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
	// for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
	// kernel.at<double>(m+ kernelRadiusX, n+ kernelRadiusY) = (double) 1.0/(size*size);
	// }
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
					// Get the right indices to use
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernelDxVal = kernelDx.at<double>(kernelx, kernely);
					double kernelDyVal = kernelDy.at<double>(kernelx, kernely);
					// Do the multiplication
					sumDx += imageval * kernelDxVal;
					sumDy += imageval * kernelDyVal;
				}
			}
			// Set the output value as the sum of the convolution
			dx.at<double>(i, j) = (double)sumDx;
			dy.at<double>(i, j) = (double)sumDy;
		}
	}
}




// TODO: Mat image not needed (i.e. only rows and cols)
void computeGradientMagnitude(Mat image, Mat dx, Mat dy, Mat& gradientMagnitude) {
	// gradientMagnitude.create(image.rows, image.cols, CV_64F);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			gradientMagnitude.at<double>(y, x) = sqrt(dx.at<double>(y, x) * dx.at<double>(y, x) + dy.at<double>(y, x) * dy.at<double>(y, x));
		}
	}
	normalize(gradientMagnitude, gradientMagnitude, 0, 255, NORM_MINMAX);
}




// TODO: Cover edge cases! + Mat image not needed (i.e. only rows and cols)
void computeGradientDirection(Mat image, Mat dx, Mat dy, Mat& gradientDirection) {
	// gradientDirection.create(image.rows, image.cols, CV_64F);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			gradientDirection.at<double>(y, x) = atan2(dy.at<double>(y, x), dx.at<double>(y, x)); // What if dx = 0 or dy = 0?
		}
	}
	// TODO normalize for image (to look similar to lab solutions)
	//normalize(gradientDirection, gradientDirection, 0, 255, NORM_MINMAX);
}



// TODO: Add comments
void sobel(Mat image, Mat& dx, Mat& dy, Mat& gradientMagnitude, Mat& gradientDirection) {
	dx.create(image.rows, image.cols, CV_64F);
	dy.create(image.rows, image.cols, CV_64F);
	gradientMagnitude.create(image.rows, image.cols, CV_64F);
	gradientDirection.create(image.rows, image.cols, CV_64F);
	computeDXDY(image, dx, dy);
	computeGradientMagnitude(image, dx, dy, gradientMagnitude);
	computeGradientDirection(image, dx, dy, gradientDirection);
}

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

bool validX(int x, int cols) {
	return (x >= 0 && x < cols);
}
bool validY(int y, int rows) {
	return (y >= 0 && y < rows);
}
int sumMatrix(int*** acc, int r, int rows, int cols) {
	int sum = 0;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			sum += acc[y][x][r];
		}
	}
	return sum;
}
void hough(Mat gradientDirection, Mat gradientMagnitude, int***& accumulator, Mat& image) {
	for (int y = 0; y < gradientDirection.rows; y++) {
		for (int x = 0; x < gradientDirection.cols; x++) {
			if (gradientMagnitude.at<double>(y, x) > 0) {
				for (int r = 0; r < 115; r++) {
					for (int i = -1; i <= 1; i = i + 2) {
						for (int j = -1; j <= 1; j = j + 2) {
							int x0 = (int)(x + i * r * cos(gradientDirection.at<double>(y, x)));
							int y0 = (int)(y + j * r * sin(gradientDirection.at<double>(y, x)));
							if ((x0 > 0 && x0 < gradientDirection.cols) && (y0 > 0 && y0 < gradientDirection.rows)) {
								// output.at<int>(x0,y0,r) += 1;
								accumulator[y0][x0][r] += 1;
							}
						}
					}
				}
			}
		}
	}
    //cout << "aaaaaaaaaaaaaaaaa";
	Mat accumulatorTH;
	Mat r10;
	r10.create(gradientDirection.rows, gradientDirection.cols, CV_64F);
	accumulatorTH.create(gradientDirection.rows, gradientDirection.cols, CV_64F);
	for (int r = 0; r < 115; r++) {
		for (int y = 0; y < gradientDirection.rows; y++) {
			for (int x = 0; x < gradientDirection.cols; x++) {
				if (accumulator[y][x][r] > 20) { // 10
					accumulator[y][x][r] = 255;
				}
				else {
					accumulator[y][x][r] = 0;
				}
			}
		}
	}
	int bestRadius = -1;
	int maxSum = -1;
	for (int r = 0; r < 115; r++) {
		int sum = sumMatrix(accumulator, r, gradientDirection.rows, gradientDirection.cols);
		if (sum > maxSum) {
			maxSum = sum;
			bestRadius = r;
		}
	}
	cout << "BEST R = " << bestRadius << endl;
	for (int y = 0; y < gradientDirection.rows; y++) {
		for (int x = 0; x < gradientDirection.cols; x++) {
			for (int r = 0; r < 115; r++) {
				if (r == 94) {
					r10.at<double>(y, x) = accumulator[y][x][94];
				}
				accumulatorTH.at<double>(y, x) += accumulator[y][x][r];
			}
		}
		//cout << "AICIII";
	}
	normalize(accumulatorTH, accumulatorTH, 0, 255, NORM_MINMAX);
	imwrite("hough.jpg", accumulatorTH);
	imageTH(accumulatorTH, accumulatorTH, 100);
	//threshold(accumulatorTH, accumulatorTH, 40, 255, THRESH_BINARY); // 100
	// IDEA: Mark neighbours as visited
	int circlesCount = 0;
	vector<Point> circles;
	vector<int> radiuses;
	for (int y = 0; y < gradientDirection.rows; y++) {
		for (int x = 0; x < gradientDirection.cols; x++) {
			if (accumulatorTH.at<double>(y, x) == 255) {
				// cout << x << " " << y << endl;
				Point p;
				Scalar color(255, 255, 255);
				p.x = x;
				p.y = y;
				bool ok = true;
				for (int i = 0; i < circles.size(); i++) {
					if (abs(p.x - circles[i].x) < 55 && abs(p.y - circles[i].y) < 55) { // change it
						ok = false;
						break;
					}
				}
				if (ok) {
					circlesCount++;
					// circle( image, p, bestRadius, Scalar(255,0,0), 3, LINE_AA);
					circles.push_back(p);
				}
				// circle(accumulatorTH, p, 5, color, 0);
			}
		}
	}
	for (int i = 0; i < circles.size(); i++) {
		int maxi = -1;
		int bestR = -1;
		Point p = circles[i];
		for (int r = 0; r < 115; r++) {
			if (accumulator[p.y][p.x][r] > maxi) {
				maxi = accumulator[p.y][p.x][r];
				bestR = r;
			}
		}
		radiuses.push_back(bestR);
		circle(image, p, bestR, Scalar(255, 0, 0), 3, CV_AA);
	}
	cout << "Number of circles: " << circlesCount << endl;
	imwrite("houghTH.jpg", accumulatorTH);
	imwrite("r10.jpg", r10);
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

	imageTH(houghSpaceLines, houghSpaceLines, 30);

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

	//imwrite("houghSpaceLines.jpg", houghSpaceLines);


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
	// return 0;
	// vector<Vec3f> circles;
	// HoughCircles(gray_image, circles, HOUGH_GRADIENT, 1,
	// gray_image.rows/16, // change this value to detect circles with different distances to each other
	// 150, 20, 1, 50 // change the last two parameters
	// // (min_radius & max_radius) to detect larger circles
	// );
	// for( size_t i = 0; i < circles.size(); i++ )
	// {
	// Vec3i c = circles[i];
	// Point center = Point(c[0], c[1]);
	// // circle center
	// circle( image, center, 1, Scalar(0,100,100), 3, LINE_AA);
	// // circle outline
	// int radius = c[2];
	// circle( image, center, radius, Scalar(255,0,255), 3, LINE_AA);
	// }
	// imwrite("detected circle.png", image);
	// return 0;
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
	imwrite("detected.png", image);

	Mat houghSpaceLines;
	getHoughLinesSpace(gradientMagnitude, gradientDirection, gray_image, 
						thetaLocations, roLocations, houghSpaceLines, 5);
	//cout << thetaLocations.size();
	imwrite("houghSpaceLines.jpg", houghSpaceLines);

	extractLinesFromPeacks(image, thetaLocations, roLocations, houghSpaceLines);
	imwrite("evidence.jpg", image);
	// imwrite("th_magnitude.jpg", gradientMagnitudeTH);
	// image.release();
	// gray_image.release();
	// dx.release();
	// dy.release();
	// gradientMagnitude.release();
	// gradientDirection.release();
	// Mat blurred_greyImage;
	// for (int i = 0; i < 4; i++) {
	// GaussianBlur(gray_image, blurred_greyImage, Size(5, 5), 0);
	// gray_image = blurred_greyImage;
	// }
	// imwrite("blurred.jpg", blurred_greyImage);
	// Mat evidenceVote;
	// evidenceVote.create(445, 545, gray_image.type());
	// Mat resultx;
	// Mat resulty;
	// Mat gradImage;
	// Mat gradImageth;
	// Mat directionGrad;
	// Mat directionGradth;
	// Mat sobel;
	// Mat sobelx;
	// Mat sobely;
	//sobel.create(blurred_greyImage.rows, blurred_greyImage.cols, blurred_greyImage.type());
	//sobel.create(445, 545, blurred_greyImage.type());
	// sobel.create(blurred_greyImage.size(), blurred_greyImage.type());
	// sobelx.create(blurred_greyImage.size(), CV_64F);
	// sobely.create(blurred_greyImage.size(), CV_64F);
	// Sobel(blurred_greyImage, sobelx, CV_64F, 1, 0);
	// Sobel(blurred_greyImage, sobely, CV_64F, 0, 1);
	// imwrite("sobelx_opencv.jpg", sobelx);
	// imwrite("sobely_opencv.jpg", sobely);
	// SobelY(blurred_greyImage, 3, resulty);
	// SobelX(blurred_greyImage, 3, resultx);
	// Sobel(blurred_greyImage, sobel, CV_32F, 1, 1);
	//for (int y = 0; y < sobel.rows; y++) {
	// //cout << "merge";
	// for (int x = 0; x < sobel.cols; x++) {
	// uchar value = sobel.at<uchar>(y, x);
	// if (value >= 50) {
	// sobel.at<uchar>(y, x) = 255;
	// }
	// else if (value < 50) {
	// sobel.at<uchar>(y, x) = 0;
	// }
	// }
	//}
	//
	//
	// imwrite("opencvsobel.jpg", sobel);
	// gradDirection(resultx, resulty, directionGrad);
	// imwrite("directionGrad.jpg", directionGrad);
	// gradientImage(resultx, resulty, gradImage);
	// gradImageTH(gradImage, gradImageth);
	//hough(gradImageth, ipppArr);
	//
	//
	//for (i = 0; i < dim1; ++i)
	// for (j = 0; j < dim2; ++j)
	// for (k = 0; k < dim3; ++k) {
	// sumRadius[i][j] += ipppArr[i][j][k];
	// //printf("[%d]", ipppArr[i][j][k]);
	// }
	//for (i = 0; i < dim1; ++i) {
	// for (j = 0; j < dim2; ++j) {
	// //printf("[%d]", sumRadius[i][j]);
	// if (sumRadius[i][j] > 1500) {
	// evidenceVote.at<uchar>(i, j) = 255;
	// }
	// }
	//}
	//
	//
	// imwrite("evidence.jpg", evidenceVote);
	// gradImageTH(directionGrad, directionGradth);
	// imwrite("resulty.jpg", resulty);
	// imwrite("resultx.jpg", resultx);
	// imwrite("gradImage.jpg", gradImage);
	// imwrite("gradImageth.jpg", gradImageth);
	// imwrite("directionGradth.jpg", directionGradth);
	//
	//
	return 0;
}