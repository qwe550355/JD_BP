#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define innode 128*128     //输入结点数，可以修改
#define hidenode 129      //隐含结点数
#define hidelayer 3   //隐含层数
#define outnode 1       //输出结点数


// --- -1~1 随机数产生器 --- 
inline double get_11Random()    // -1 ~ 1
{
	return ((2.0*(double)rand() / RAND_MAX) - 1);
}

// --- sigmoid 函数 --- 
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

// --- 输入层节点。包含以下分量：--- 
// 1.value:     固定输入值； 
// 2.weight:    面对第一层隐含层每个节点都有权值； 
// 3.wDeltaSum: 面对第一层隐含层每个节点权值的delta值累积
typedef struct inputNode
{
	double value;
	vector<double> weight, wDeltaSum, oldWD;
}inputNode;

// --- 输出层节点。包含以下数值：--- 
// 1.value:     节点当前值； 
// 2.delta:     与正确输出值之间的delta值； 
// 3.rightout:  正确输出值
// 4.bias:      偏移量
// 5.bDeltaSum: bias的delta值的累积，每个节点一个
typedef struct outputNode   // 输出层节点
{
	double value, delta, rightout, bias, bDeltaSum, oldBD;
}outputNode;

// --- 隐含层节点。包含以下数值：--- 
// 1.value:     节点当前值； 
// 2.delta:     BP推导出的delta值；
// 3.bias:      偏移量
// 4.bDeltaSum: bias的delta值的累积，每个节点一个
// 5.weight:    面对下一层（隐含层/输出层）每个节点都有权值； 
// 6.wDeltaSum： weight的delta值的累积，面对下一层（隐含层/输出层）每个节点各自积累
typedef struct hiddenNode   // 隐含层节点
{
	double value, delta, bias, bDeltaSum, oldBD;
	vector<double> weight, wDeltaSum, oldWD;
}hiddenNode;

// --- 单个样本 --- 
typedef struct sample
{
	vector<double> in, out;
}sample;


// --- BP神经网络 --- 
class BpNet
{
public:
	BpNet();    //构造函数
	void forwardPropagationEpoc();  // 单个样本前向传播
	void backPropagationEpoc();     // 单个样本后向传播
	void updateParaEpoc();          //更新参数
	void AIadjust();      //自适应学习率
	void protectPara();

	void training(static vector<sample> sampleGroup, double threshold);// 更新 weight, bias
	void predict(vector<sample>& testGroup);                          // 神经网络预测

	void setInput(static vector<double> sampleIn);     // 设置学习样本输入
	void setOutput(static vector<double> sampleOut);    // 设置学习样本输出

	void readNeural();
	void writeNeural();
	
	

public:
	vector<double> errorStatic;
	double error;
	inputNode* inputLayer[innode];                      // 输入层（仅一层）
	outputNode* outputLayer[outnode];                   // 输出层（仅一层）
	hiddenNode* hiddenLayer[hidelayer][hidenode];       // 隐含层（可能有多层）
	bool inherit = 0;

private:
	int sampleNum = 0;             //记录样本个数
	double learningrate = 0.05;   //学习速率
	double lastError = 0.f;
	double alpher = 0.95;
	int repeatTime = 0;

};
