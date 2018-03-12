#define _CRT_SECURE_NO_WARNINGS
#include "BPnet.h"
#include<opencv2/opencv.hpp>



using namespace std;
using namespace cv;




int main(){
	int num = 0;
	
	Mat Pic;
	String FileSpecificAddr;
	Pic = imread("D://1.png",0);
	cout << Pic.size();
	imshow("low", Pic);
	waitKey(0);
	for (int i = 0; i < 128 * 128; i++)
	{
		cout << (double)(uchar)Pic.data[i];
	}
	getchar();
	return 0;
}