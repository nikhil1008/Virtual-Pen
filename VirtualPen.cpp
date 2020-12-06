#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <unordered_map>

using namespace cv;
using namespace std;

//
//Mat removeGlare(Mat img) {
//	cv::cvtColor(img, img, CV_BGR2YUV);
//	std::vector<cv::Mat> channels;
//	cv::split(img, channels);
//	cv::equalizeHist(channels[0], channels[0]);
//	cv::merge(channels, img);
//	cv::cvtColor(img, img, CV_YUV2BGR);
//
//}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}



void floodfill(int i, int j, Mat & img , int levels) {
	int level = levels;
	for (int level = 1; level <= levels; level++) {
		for (int k = max(0, i - level); k <= min(i + level, img.rows); k++) {
			img.at<uchar>(k, max(j - level, 0)) = 255;
			img.at<uchar>(k, min(j + level, img.cols)) = 255;
		}
		for (int k = max(0, j - level); k <= min(j + level, img.cols); k++) {
			img.at<uchar>(max(i - level, 0), k) = 255;
			img.at<uchar>(min(i + level, img.rows), k) = 255;
		}

	}
}

pair<int, int> find_centroid(Mat& img) {
	Mat tmp;
	int n = connectedComponents(img, tmp);
	unordered_map<int, int> counts;
	unordered_map<int, pair<int, int>> centroidX; 
	unordered_map<int, pair<int, int>> centroidY;

	for (int i = 0; i < tmp.rows; i++) {
		for (int j = 0; j < tmp.cols; j++) {
			int label = tmp.at<int>(i, j);
			if (counts.find(label) != counts.end()) {
				counts[label] += 1;
				centroidX[label].first += i;
				centroidX[label].second += 1;
				centroidY[label].first += j;
				centroidY[label].second += 1;
			}
			else {
				counts[label] = 1;
				centroidX[label] = make_pair(0, 0);
				centroidY[label] = make_pair(0, 0);
			}
		}
	}

	int max_label = -1;
	int max_count = -1;

	auto it = counts.begin();
	while (it != counts.end()) {
		if (it->first != 0 and it->second > max_count) {
			max_count = it->second;
			max_label = it->first;
		}
		it++;
	}


	int cent_x = (centroidX[max_label].second >0 ) ? centroidX[max_label].first / centroidX[max_label].second : -1;
	int cent_y = (centroidY[max_label].second >0)  ? centroidY[max_label].first / centroidY[max_label].second : -1;

	cout << "\nmax_label = " << max_label << "\n";

	//int cent_x = 0;
	//int cent_y = 0;
	return make_pair(cent_x, cent_y); 
}


//int main(int argc, char* argv[])
//{
//    Mat img = imread("D:/expt/tmp1.jpg", IMREAD_COLOR);
//    resize(img, img, Size(256, 256));
//
//    Mat rgbChannel[3];
//    split(img, rgbChannel);
//    Mat tmp = rgbChannel[0];
//    Mat tmp2 = rgbChannel[2];
//
//    Mat out;
//    string window_name = "My Camera Feed";
//    namedWindow("before"); //create a window called "My Camera Feed"
//    namedWindow("after"); //create a window called "My Camera Feed"
//	namedWindow("after2"); //create a window called "My Camera Feed"
//
//    threshold(tmp-tmp2, out, 50, 255, 0);
//
//	auto pair_ = find_centroid(out);
//	int cent_x = pair_.first ;
//	int cent_y = pair_.second;
//	cout << "cent_x = " << cent_x << "  cent_y =  " << cent_y << " \n";
//    
//	Mat cumulative = Mat::zeros(Size(out.rows, out.cols), CV_8UC1);
//
//	floodfill(cent_x, cent_y,  cumulative, 3);
//
//
//	string ty = type2str(out.type());
//	printf("Matrix: %s %dx%d \n", ty.c_str(), out.cols, out.rows);
//
//	imshow("before", out);
//    imshow("after", img);
//	imshow("after2", cumulative);
//
//    //cout << "M = " << endl << " " << img << endl << endl;
//
//    //imshow("Display window", img);
//    int k = waitKey(0); // Wait for a keystroke in the window
//    
//    return 0;
//
//}



int main(int argc, char* argv[])
{
	//Open the default video camera
	VideoCapture cap(0);

	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video camera" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;

	string window_name = "My Camera Feed";
	namedWindow(window_name); //create a window called "My Camera Feed"
	namedWindow("before"); //create a window called "My Camera Feed"
	namedWindow("after"); //create a window called "My Camera Feed"

	Mat background;
	bool bSuccess = cap.read(background);
	Mat rgbChannel[3];
	split(background, rgbChannel);
	Mat tmp =  rgbChannel[0];
	Mat tmp2 = rgbChannel[2];
	threshold(tmp - tmp2, background, 50, 255, 0);
	
	Mat cumulative = Mat::zeros(Size(background.rows, background.cols), CV_8UC1);

	while (true)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video 

		//Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			cin.get(); //Wait for any key press
			break;
		}

		Mat rgbChannel[3];
		split(frame, rgbChannel);
		Mat tmp = rgbChannel[0];
		Mat tmp2 = rgbChannel[2];
		Mat out;
		threshold(tmp - tmp2, out, 50, 255, 0);

		auto pair_ = find_centroid(out);
		int cent_x = pair_.first;
		int cent_y = pair_.second;
		cout << "cent_x = " << cent_x << "  cent_y =  " << cent_y << " \n";
		if (cent_x != -1 and cent_y != -1 ) floodfill(cent_x, cent_y, cumulative, 3);

		//show the frame in the created window
		imshow(window_name, out);
		imshow("before", frame);
		imshow("after", cumulative);

		background = out;

		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop 
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}

	return 0;

}

