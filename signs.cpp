/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - signs.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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
#include <iomanip>

using namespace std;
using namespace cv;

ifstream fin("dataGTNoEntrySigns/dataNoEntry13gt.txt");
ofstream fout("imageResultsSignsVJ/signsResults13.txt");

/** Function Headers */
void detectAndDisplay( Mat frame );
float computeIOU( Rect Box1, Rect Box2 );
void getImageResults( vector<Rect> gtBoxes, vector<Rect> predBoxes, float threshold );

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;

/*Global vectors of detected frontal faces and actual ground truths*/
vector<Rect> signs;
vector<Rect> gtsigns;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Compute TPR and F1-score for the given image
	getImageResults(gtsigns, signs, 0.5);

	// 5. Save Result Image showing both detected and ground truth boxes
	imwrite( "detectedSigns/detectedSignsNoEntry13.jpg", frame );

	return 0;
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
	for( int i = 0; i < signs.size(); i++ )
	{
		rectangle(frame, Point(signs[i].x, signs[i].y), Point(signs[i].x + signs[i].width, signs[i].y + signs[i].height), Scalar( 0, 255, 0 ), 2);
	}
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