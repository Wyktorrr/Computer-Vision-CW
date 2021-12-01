/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
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
#include <algorithm>

using namespace std;
using namespace cv;

ifstream fin("noentry1gt.txt");

/** Function Headers */
void detectAndDisplay( Mat frame );
float computeIOU( Rect Box1, Rect Box2 );
void getImageResults( vector<Rect> gtBoxes, vector<Rect> predBoxes, float threshold );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

vector<Rect> faces;
vector<Rect> gtFaces;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	getImageResults(gtFaces, faces, 0.5);

	//cout << "faces detected: " << faces.size() << " " <<"ground truth faces: "<< gtFaces.size() << "\n";

	//IOU(faces.size(), gtFaces.size());

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    int x, y, width, height, bottomrightx, bottomrighty;
	string line;
	unsigned int truenooffaces = 0;

	//std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
		//5. Draw red box around ground truth
    while((fin >> x >> y >> width >> height) && getline(fin, line)) {
		//cout << x << y << width << height << "\n";
		truenooffaces++;
		bottomrightx = x + width;
		bottomrighty = y + height;
		rectangle(frame, Point(x, y), Point(bottomrightx, bottomrighty), Scalar( 0, 0, 255 ), 2);
		gtFaces.push_back(Rect(x, y, width, height));
	}

     cout << truenooffaces << "\n";
	 cout << gtFaces.size() << "\n";

    fin.close();
}

/*void IOU(unsigned int facesDetected, unsigned int trueNoOfFaces) {
	unsigned int truePositives = 0, falsePositives = 0;
	Rect intersection;
	float overlappingArea, overlappingPercentage;
	
	//Compare bounding boxes detected by the boosting algorithm with the ground truths
	for (int i = 0; i < facesDetected; i++) {
		for (int j = 0; j < trueNoOfFaces; j++) {
			//Compute the intersection 
			intersection = faces[i] & gtFaces[j];
			overlappingArea = intersection.area();
			cout << "intersection: " << overlappingArea << "\n";

			//If the bounding boxes overlap, compute the percentage of the intersection
			if (overlappingArea > 0) {
				overlappingPercentage = (overlappingArea / gtFaces[j].area()) * 100;
				cout << "gt faces area: " << gtFaces[j].area() << "\n";
				cout << "overlappingPercentage: " << overlappingPercentage << "\n";
                //If the the intersection area reaches a certain threshold, it means we have true positive
				if (overlappingPercentage >= 60) {
					truePositives++;
				}

				if (j == (gtFaces.size() - 1)) {
					falsePositives++;
				}
			} 
			else {
				if (j == (gtFaces.size() - 1)) {
					falsePositives++;
				}
			}
		}
	} 
	cout << "true positives: "<< truePositives << " " << "false positives: " << falsePositives << "\n";
}*/

float computeIOU( Rect groundTruthBox, Rect predictedBox ) {

	/*int xtopleftgt, ytopleftgt, xbottomrightgt, ybottomrightgt, 
	    xtopleftpred, ytopleftpred, xbottomrightpred, ybottomrightpred,
		xtopleft, ytopleft, xbottomright, ybottomright;*/

	float groundTruthBoxArea, predictedBoxArea, result;

	int intersectionArea, unionArea;

	intersectionArea = (groundTruthBox & predictedBox).area();
	cout << "intersevtie: "<< intersectionArea << "\n";
	unionArea = (groundTruthBox | predictedBox).area();
	cout << "uniune: " << unionArea << "\n";
	

	if (intersectionArea > 0) {
		result = intersectionArea / (float)unionArea;
	} else {
		result = 0;
	}

    //cout << "result: "<<result;
	return result;


	/*xtopleftgt = groundTruthBox.x;
	ytopleftgt = groundTruthBox.y;
	xbottomrightgt = xtopleftgt + groundTruthBox.width;
	ybottomrightgt = ytopleftgt + groundTruthBox.height;

	xtopleftpred = predictedBox.x;
	ytopleftpred = predictedBox.y;
	xbottomrightpred = xtopleftpred + predictedBox.width;
	ybottomrightpred = ytopleftpred + predictedBox.height;

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

	groundTruthBoxArea = (xbottomrightgt - xtopleftgt + 1) * (ybottomrightgt - ytopleftgt + 1);
	predictedBoxArea = (xbottomrightpred - xtopleftpred + 1) * (ybottomrightpred - ytopleftpred + 1);

	xtopleft = max(xtopleftgt, xtopleftpred);
	ytopleft = max(ytopleftgt, ytopleftpred);
	xbottomright = min(xbottomrightgt, xbottomrightpred);
	ybottomright = min(ybottomrightgt, ybottomrightpred);

	intersectionArea = (xbottomright - xtopleft + 1) * (ybottomright - ytopleft  + 1);
	cout << "intersevtie: "<< intersectionArea << "\n";
    
    unionArea = (groundTruthBoxArea + predictedBoxArea - intersectionArea);
	cout << "uniune: " << unionArea << "\n";

	result = intersectionArea / unionArea;
   
    return result;*/
}

void getImageResults(vector<Rect> gtFaces, vector<Rect> faces, float iouThreshold) {
	int truePositives = 0, falsePositives, falseNegatives, detectedFaces, trueNoOfFaces;
	float iou, TPR, accuracy, precision, recall, F1Score;

	trueNoOfFaces = gtFaces.size();
	detectedFaces = faces.size();

	//cout << trueNoOfFaces << " " << detectedFaces << "\n";

	if ((detectedFaces == 0) || (trueNoOfFaces == 0)) {
		truePositives = 0; 
		falsePositives = 0;
		falseNegatives = 0;
	}

	for (int i = 0; i < detectedFaces; i++) {
		for (int j = 0; j < trueNoOfFaces; j++) {
             iou = computeIOU(faces[i], gtFaces[j]);

			 cout << "iou: " << iou << " ";

			 if (iou > iouThreshold) {
				 truePositives++;
			 }
		}
	}

	cout << "detectedFaces: " << detectedFaces << "trueNoOfFaces: " << trueNoOfFaces << "\n";
	cout << "truePositives: " << truePositives << "\n";

	falsePositives = detectedFaces - trueNoOfFaces;
	
	falseNegatives = trueNoOfFaces - truePositives;

	cout << "falsePositives: " << falsePositives << "\n";
	cout << "falseNegatives: " << falseNegatives << "\n";

	if (trueNoOfFaces > 0) {
		TPR = (float)truePositives / (float)trueNoOfFaces;
	} else if (trueNoOfFaces == 0) {
		TPR = 0;
	}

	accuracy = (float)truePositives / (float)detectedFaces;
	cout << "accuracy: " << accuracy << "\n";
	precision = (float)truePositives / (float)(truePositives + falsePositives);
	cout << "precision: " << precision << "\n";
	recall = (float)truePositives / (float)(truePositives + falseNegatives);
	cout << "recall: " << recall << "\n";

	F1Score = (float)(2 * precision * recall) / (float)(precision + recall);

	cout <<  "TPR: " << TPR << " " << "F1Score: " << F1Score << "\n";
}	

/*void getImageResults(vector<Rect> gtFaces, vector<Rect> faces, int iouThreshold) {
	unsigned int truePositives = 0, falsePositives = 0, falseNegatives = 0, detectedFaces, trueNoOfFaces;
	float iou;

	trueNoOfFaces = gtFaces.size();
	detectedFaces = faces.size();

	if ((detectedFaces == 0) || (trueNoOfFaces == 0)) {
		truePositives = 0; 
		falsePositives = 0;
		falseNegatives = 0;
	}

	vector<int> groundTruthThreshholdIndexes;
	vector<int> predictedThreshholdIndeces;
	vector<float> ious;
	vector<float> iousSorted;

	for (int i = 0; i < detectedFaces; i++) {
		for (int j = 0; j < trueNoOfFaces; j++) {
             iou = computeIOU(faces[i], gtFaces[j]);

			 if (iou > iouThreshold) {
				 groundTruthThreshholdIndexes.push_back(j); 
				 predictedThreshholdIndeces.push_back(i);
				 ious.push_back(iou);
			 }
		}
	} 
	sort(ious.begin(), ious.end());
	
	vector <float> gtMatchIndex;
	vector <float> predMatchIndex;

	for (int i = 0; i < ious.size(); i++) {
		int gtIndex = groundTruthThreshholdIndexes[i];
		int predIndex = predictedThreshholdIndeces[i];

        // If the boxes are unmatched, add them to matches
		if (!binary_search(gtMatchIndex.begin(), gtMatchIndex.end(), gtIndex) &&
		     !binary_search(predMatchIndex.begin(), predMatchIndex.end(), predMatchIndex)) {
			groundTruthThreshholdIndexes.push_back(gtIndex);
			predictedThreshholdIndeces.push_back(predIndex);
		}
	}

	truePositives = groundTruthThreshholdIndexes.size();
	falsePositives = faces.size() - predMatchIndex.size();
	falseNegatives = gtFaces.size() - gtMatchIndex.size();

	cout << truePositives << " " << falsePositives << " " << falseNegatives << "\n";
}*/
