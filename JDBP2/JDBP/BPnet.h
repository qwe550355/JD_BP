#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define innode 128*128     //���������������޸�
#define hidenode 129      //���������
#define hidelayer 3   //��������
#define outnode 1       //��������


// --- -1~1 ����������� --- 
inline double get_11Random()    // -1 ~ 1
{
	return ((2.0*(double)rand() / RAND_MAX) - 1);
}

// --- sigmoid ���� --- 
inline double sigmoid(double x)
{
	double ans = 1 / (1 + exp(-x));
	return ans;
}

inline double relu(double x)
{
	double ans;
	if (x <= 0)
		ans = 0;
	else
		ans = x;
	return ans;
}

inline double difR(double x)
{
	double ans;
	if (x < 0)
		ans = 0;
	else
		ans = 1;
	return ans;
}

// --- �����ڵ㡣�������·�����--- 
// 1.value:     �̶�����ֵ�� 
// 2.weight:    ��Ե�һ��������ÿ���ڵ㶼��Ȩֵ�� 
// 3.wDeltaSum: ��Ե�һ��������ÿ���ڵ�Ȩֵ��deltaֵ�ۻ�
typedef struct inputNode
{
	double value;
	vector<double> weight, wDeltaSum, oldWD;
}inputNode;

// --- �����ڵ㡣����������ֵ��--- 
// 1.value:     �ڵ㵱ǰֵ�� 
// 2.delta:     ����ȷ���ֵ֮���deltaֵ�� 
// 3.rightout:  ��ȷ���ֵ
// 4.bias:      ƫ����
// 5.bDeltaSum: bias��deltaֵ���ۻ���ÿ���ڵ�һ��
typedef struct outputNode   // �����ڵ�
{
	double value, delta, rightout, bias, bDeltaSum, oldBD;
}outputNode;

// --- ������ڵ㡣����������ֵ��--- 
// 1.value:     �ڵ㵱ǰֵ�� 
// 2.delta:     BP�Ƶ�����deltaֵ��
// 3.bias:      ƫ����
// 4.bDeltaSum: bias��deltaֵ���ۻ���ÿ���ڵ�һ��
// 5.weight:    �����һ�㣨������/����㣩ÿ���ڵ㶼��Ȩֵ�� 
// 6.wDeltaSum�� weight��deltaֵ���ۻ��������һ�㣨������/����㣩ÿ���ڵ���Ի���
typedef struct hiddenNode   // ������ڵ�
{
	double value, delta, bias, bDeltaSum, oldBD;
	vector<double> weight, wDeltaSum, oldWD;
}hiddenNode;

// --- �������� --- 
typedef struct sample
{
	vector<double> in, out;
}sample;


// --- BP������ --- 
class BpNet
{
public:
	BpNet();    //���캯��
	void forwardPropagationEpoc();  // ��������ǰ�򴫲�
	void backPropagationEpoc();     // �����������򴫲�
	void updateParaEpoc();          //���²���
	void AIadjust();      //����Ӧѧϰ��
	void protectPara();

	void training(static vector<sample> sampleGroup, double threshold);// ���� weight, bias
	void predict(vector<sample>& testGroup);                          // ������Ԥ��

	void setInput(static vector<double> sampleIn);     // ����ѧϰ��������
	void setOutput(static vector<double> sampleOut);    // ����ѧϰ�������

	void readNeural();
	void writeNeural();
	
	

public:
	vector<double> errorStatic;
	double error;
	inputNode* inputLayer[innode];                      // ����㣨��һ�㣩
	outputNode* outputLayer[outnode];                   // ����㣨��һ�㣩
	hiddenNode* hiddenLayer[hidelayer][hidenode];       // �����㣨�����ж�㣩
	bool inherit = 0;

private:
	int sampleNum = 0;             //��¼��������
	double learningrate = 0.05;   //ѧϰ����
	double lastError = 0.f;
	double alpher = 0.95;
	int repeatTime = 0;

};
