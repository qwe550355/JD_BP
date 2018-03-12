#define _CRT_SECURE_NO_WARNINGS
#include "BPnet.h"
#include<opencv2/opencv.hpp>



using namespace std;
using namespace cv;

int getArrayLen(vector<double>* array)
{
	return (sizeof(array) / sizeof(array[0]));
}

String Speci_tostring(int p)
{
	String result = to_string(p);
	if (p < 10)
		result = "0" + result;
	return result;
}

vector<vector<double>> readMat(int number, int PicSize)
{
	vector<vector<double>> Data;
	vector<double> tempData;
	vector<Mat> Pic;
	for (int i = 1; i < 11; i++)
	{
		for (int j = 1; j <= number; j++)
		{
			String FileSpecificAddr;
			if (j < 100)
				FileSpecificAddr = "D:\\jinda\\JDBP2\\JDBP\\EnglishFnt\\English\\Fnt\\Sample0" + Speci_tostring(i) + "\\img0" + Speci_tostring(i) + "-000" + Speci_tostring(j) + ".png";
			if (j >= 100)
				FileSpecificAddr = "D:\\jinda\\JDBP2\\JDBP\\EnglishFnt\\English\\Fnt\\Sample0" + Speci_tostring(i) + "\\img0" + Speci_tostring(i) + "-00" + Speci_tostring(j) + ".png";
			//std::cout << FileSpecificAddr;
			Mat Temp = imread(FileSpecificAddr,0);
			double TempDataStore = 0;
			for (int k = 0; k < PicSize*PicSize; k++)
			{
				TempDataStore = (uchar)Temp.data[k];
				if (TempDataStore < 190)
					TempDataStore = 1;
				else
					TempDataStore = 0;
				tempData.push_back(TempDataStore);
			}
			Data.push_back(tempData);
			tempData.clear();
		}
	}
	return Data;
}

int main()
{	
	bool errorFlag = 0;
	bool readFlag = 0;
	bool storeFlag = 0;
	int NeuralType = 0;
	int LongNum;
	BpNet testNet;
	cout << "是否读取原有的神经网络" << endl;
	cin >> readFlag;
	if (readFlag)
		testNet.readNeural();
	else
	{
		cout << "请选择神经网络运算类型  0： XOR 1：数字识别" << endl;
		cin >> NeuralType;
		vector<double>* samplein = NULL;
		vector<double>* sampleout = NULL;
		vector<vector<double>> sampleInit;
		sample *sampleInOut = NULL; //让Sample有一个可变大小
		if (NeuralType == 0)
		{
			// 学习样本
			LongNum = 4;
			samplein = new vector<double>[LongNum];
			sampleout = new vector<double>[LongNum];
			sampleInOut = new sample[LongNum];
			samplein[0].push_back(0); samplein[0].push_back(0); sampleout[0].push_back(0);
			samplein[1].push_back(0); samplein[1].push_back(1); sampleout[1].push_back(1);
			samplein[2].push_back(1); samplein[2].push_back(0); sampleout[2].push_back(1);
			samplein[3].push_back(1); samplein[3].push_back(1); sampleout[3].push_back(0);
			for (int i = 0; i < LongNum ; i++)
			{
				sampleInOut[i].in = samplein[i];
				sampleInOut[i].out = sampleout[i];
			}
			vector<sample> sampleGroup(sampleInOut, sampleInOut + LongNum);
			testNet.training(sampleGroup, 0.001);
		}
		if (NeuralType == 1)
		{
			// 学习样本
			cout << "请输入想以多少张图作为输入（每个数字个n张）" << endl;
			cin >> LongNum;

			sampleInit = readMat(LongNum,128);
			cout << "是否继承原有的神经网络" << endl;
			cin >> testNet.inherit;
			samplein = new vector<double>[LongNum*10];
			sampleout = new vector<double>[LongNum*10];
			sampleInOut = new sample[LongNum*10];
			
			//for (int i = 0; i<sampleInit.size(); i++)
			//{
			//	for (int j = 0; j < sampleInit[i].size(); j++)
			//		samplein[i].push_back(sampleInit[i][j]);
			//}
			for (double i = 0; i < 10; i++)
			{		
				for (int j = 0; j < LongNum; j++)
				{
					sampleout[(int)i*LongNum+j].push_back(i);
				}
			}
			for (int i = 0; i < LongNum * 10; i++)
			{
				sampleInOut[i].in = sampleInit[i];
				sampleInOut[i].out = sampleout[i];
			}
			vector<sample> sampleGroup(sampleInOut, sampleInOut + LongNum * 10);
			testNet.training(sampleGroup, 0.2);
		}
		cout << "是否将误差数据储存" << endl;
		cin >> errorFlag;
		if (errorFlag)
		{
			FILE* stream0;
			if ((stream0 = fopen("error.txt", "w+")) == NULL)
			{
				cout << "创建文件失败!";
				exit(1);
			}
			for (int i = 0; i < testNet.errorStatic.size();i++)
			{
				fprintf(stream0, "%f,", testNet.errorStatic[i]);
			}
			fclose(stream0);
		}

		delete [] sampleInOut;  //防止溢出
		delete [] samplein;     //防止溢出
		delete [] sampleout;    //防止溢出

		cout << "是否储存训练值" << endl;
		cin >> storeFlag;
		if (storeFlag) 
			testNet.writeNeural();
	}
	
	// 测试数据
	vector<vector<double>> sampleOutit;
	vector<double> testin[4];
	vector<double> testout[4];
	sample testInOut[4];
	if(NeuralType == 0)
	{
		testin[0].push_back(0.1);   testin[0].push_back(0.2);
		testin[1].push_back(0.15);  testin[1].push_back(0.9);
		testin[2].push_back(1.1);   testin[2].push_back(0.01);
		testin[3].push_back(0.88);  testin[3].push_back(1.03);
		for (int i = 0; i < 4; i++)
			testInOut[i].in = testin[i];
	}
	else
	{
		String FileSpecificAddr;
		for (int i = 1; i <= 4; i++)
		{
			vector<double> temp;
			FileSpecificAddr = "D:\\jinda\\JDBP2\\JDBP\\EnglishFnt\\English\\Fnt\\Sample0" + Speci_tostring(i) + "\\img0" + Speci_tostring(i) + "-00296.png";
			//std::cout << FileSpecificAddr;
			Mat Temp = imread(FileSpecificAddr, 0);
			double tempdataStore = 0;
			for (int k = 0; k < 128 * 128; k++)
			{
				tempdataStore = (uchar)Temp.data[k];
				if (tempdataStore < 190)
					tempdataStore = 1;
				else
					tempdataStore = 0;
				temp.push_back(tempdataStore);
			}
			sampleOutit.push_back(temp);
			temp.clear();
		}

		for (int i = 0; i < 4; i++)
			testInOut[i].in = sampleOutit[i];
	}

	vector<sample> testGroup(testInOut, testInOut + 4);

	// 预测测试数据，并输出结果
	testNet.predict(testGroup);
	for (int i = 0; i < testGroup.size(); i++)
	{
		if (NeuralType == 0)
		{
			for (int j = 0; j < testGroup[i].in.size(); j++) 
				cout << testGroup[i].in[j] << "\t";
			cout << "-- prediction :";
		}
		for (int j = 0; j < testGroup[i].out.size(); j++) 
			cout << testGroup[i].out[j] << "\t";
		cout << endl;
	}
	system("pause");
	return 0;
}

