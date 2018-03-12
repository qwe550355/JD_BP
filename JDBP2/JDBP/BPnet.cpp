#define _CRT_SECURE_NO_WARNINGS
#include "BPnet.h"

using namespace std;

BpNet::BpNet()
{
	srand((unsigned)time(NULL));        // 随机数种子    
	error = 100.f;                      // error初始值，极大值即可
	errorStatic.clear();
	// 初始化输入层
	for (int i = 0; i < innode; i++)
	{
		inputLayer[i] = new inputNode();
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->weight.push_back(get_11Random());
			inputLayer[i]->wDeltaSum.push_back(0.f);
			inputLayer[i]->oldWD.push_back(0.f);
		}
	}

	// 初始化隐藏层
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				hiddenLayer[i][j]->oldBD = 0.f;
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->weight.push_back(get_11Random());
					hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
					hiddenLayer[i][j]->oldWD.push_back(0.f);
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				hiddenLayer[i][j]->oldBD = 0.f;
				for (int k = 0; k < hidenode; k++) 
				{ 
					hiddenLayer[i][j]->weight.push_back(get_11Random());
					hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
					hiddenLayer[i][j]->oldWD.push_back(0.f);
				}
			}
		}
	}

	// 初始化输出层
	for (int i = 0; i < outnode; i++)
	{
		outputLayer[i] = new outputNode();
		outputLayer[i]->bias = get_11Random();
		outputLayer[i]->oldBD = 0.f;
	}
	if (inherit)
		readNeural();
}

void BpNet::forwardPropagationEpoc()
{
	// forward propagation on hidden layer
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < innode; k++)
				{
					sum += inputLayer[k]->value * inputLayer[k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = relu(sum);
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < hidenode; k++)
				{
					sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = relu(sum);
			}
		}
	}

	// forward propagation on output layer
	for (int i = 0; i < outnode; i++)
	{
		double sum = 0.f;
		for (int j = 0; j < hidenode; j++)
		{
			sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
		}
		sum += outputLayer[i]->bias;
		outputLayer[i]->value = relu(sum);
	}
}

void BpNet::backPropagationEpoc()
{
	// backward propagation on output layer
	// -- compute delta
	for (int i = 0; i < outnode; i++)
	{
		double tmpe = fabs(outputLayer[i]->value - outputLayer[i]->rightout);
		error += tmpe * tmpe / 2;

		outputLayer[i]->delta
			= -(outputLayer[i]->value - outputLayer[i]->rightout)*difR(outputLayer[i]->value);
	}
	// backward propagation on hidden layer
	// -- compute delta
	for (int i = hidelayer - 1; i >= 0; i--)    // 反向计算
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k<outnode; k++)
				{ 
					sum += outputLayer[k]->delta * hiddenLayer[i][j]->weight[k]; 
				}
				hiddenLayer[i][j]->delta = sum * difR(hiddenLayer[i][j]->value);
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k<hidenode; k++)
				{ 
					sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k];
				}
				hiddenLayer[i][j]->delta = sum *difR(hiddenLayer[i][j]->value);
			}
		}
	}

	// backward propagation on input layer
	// -- update weight delta sum
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->wDeltaSum[j] += inputLayer[i]->value * hiddenLayer[0][j]->delta;
		}
	}

	// backward propagation on hidden layer
	// -- update weight delta sum & bias delta sum
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta;
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < hidenode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i + 1][k]->delta;
				}
			}
		}
	}

	// backward propagation on output layer
	// -- update bias delta sum
	for (int i = 0; i < outnode; i++)
		outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
}

void BpNet::updateParaEpoc()
{
	// backward propagation on input layer
	// -- update weight
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{

			inputLayer[i]->weight[j] +=
				(1 - alpher) * learningrate * inputLayer[i]->wDeltaSum[j] + inputLayer[i]->oldWD[j] * alpher;
			inputLayer[i]->oldWD[j] =
				(1 - alpher) * learningrate * inputLayer[i]->wDeltaSum[j] + inputLayer[i]->oldWD[j] * alpher;
		}
	}

	// backward propagation on hidden layer
	// -- update weight & bias
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				// bias
				hiddenLayer[i][j]->bias += 
					(1-alpher) * learningrate * hiddenLayer[i][j]->bDeltaSum + alpher * hiddenLayer[i][j]->oldBD;
				hiddenLayer[i][j]->oldBD = 
					(1 - alpher) * learningrate * hiddenLayer[i][j]->bDeltaSum + alpher * hiddenLayer[i][j]->oldBD;

				// weight
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->weight[k] +=
						(1 - alpher) * learningrate * hiddenLayer[i][j]->wDeltaSum[k] + hiddenLayer[i][j]->oldWD[k] * alpher;
					hiddenLayer[i][j]->oldWD[k] =
						(1 - alpher) * learningrate * hiddenLayer[i][j]->wDeltaSum[k] + hiddenLayer[i][j]->oldWD[k] * alpher;
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				// bias
				hiddenLayer[i][j]->bias +=
					(1 - alpher) * learningrate * hiddenLayer[i][j]->bDeltaSum + alpher * hiddenLayer[i][j]->oldBD;
				hiddenLayer[i][j]->oldBD = 
					(1 - alpher) * learningrate * hiddenLayer[i][j]->bDeltaSum + alpher * hiddenLayer[i][j]->oldBD;

				// weight
				for (int k = 0; k < hidenode; k++)
				{
					hiddenLayer[i][j]->weight[k] +=
						(1 - alpher) * learningrate * hiddenLayer[i][j]->wDeltaSum[k] + hiddenLayer[i][j]->oldWD[k] * alpher;

					hiddenLayer[i][j]->oldWD[k] =
						(1 - alpher) * learningrate * hiddenLayer[i][j]->wDeltaSum[k] + hiddenLayer[i][j]->oldWD[k] * alpher;
				}
			}
		}
	}

	// backward propagation on output layer
	// -- update bias
	for (int i = 0; i < outnode; i++)
	{
		outputLayer[i]->bias +=
			(1 - alpher) * learningrate * outputLayer[i]->bDeltaSum + alpher * outputLayer[i]->oldBD;
		outputLayer[i]->oldBD =
			(1 - alpher) * learningrate * outputLayer[i]->bDeltaSum + alpher * outputLayer[i]->oldBD;
	}
}

void BpNet::training(static vector<sample> sampleGroup, double threshold)
{
	int TrainTime=0;
	sampleNum = sampleGroup.size();
	long int times = 0;
	while (error > threshold && TrainTime < 1800)
		//for (int curTrainingTime = 0; curTrainingTime < trainingTime; curTrainingTime++)
	{
		error = 0.f;
		times++;
		// initialize delta sum
		for (int i = 0; i < innode; i++)
		{
			inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);
		}		
		for (int i = 0; i < hidelayer; i++){
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
				hiddenLayer[i][j]->bDeltaSum = 0.f;
			}
		}
		for (int i = 0; i < outnode; i++)
		{
			outputLayer[i]->bDeltaSum = 0.f;
		}
		//initalize finished

		for (int iter = 0; iter < sampleNum; iter++)
		{
			setInput(sampleGroup[iter].in);
			setOutput(sampleGroup[iter].out);

			forwardPropagationEpoc();
			backPropagationEpoc();
		}
		AIadjust();
		updateParaEpoc();
		protectPara();
		TrainTime++;
		cout << "第" << times << "次  " << "training error: " << error << "   LearningRate: " << learningrate << "  重复次数" << repeatTime<<endl;
		lastError = error;
		errorStatic.push_back(error);
	}
}

void BpNet::predict(vector<sample>& testGroup)
{
	int testNum = testGroup.size();

	for (int iter = 0; iter < testNum; iter++)
	{
		testGroup[iter].out.clear();
		setInput(testGroup[iter].in);

		// forward propagation on hidden layer
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == 0)
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < innode; k++)
					{
						sum += inputLayer[k]->value * inputLayer[k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = relu(sum);
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < hidenode; k++)
					{
						sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = relu(sum);
				}
			}
		}

		// forward propagation on output layer
		for (int i = 0; i < outnode; i++)
		{
			double sum = 0.f;
			for (int j = 0; j < hidenode; j++)
			{
				sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
			}
			sum += outputLayer[i]->bias;
			outputLayer[i]->value = relu(sum);
			testGroup[iter].out.push_back(outputLayer[i]->value);
		}
	}
}

void BpNet::setInput(static vector<double> sampleIn)
{
	for (int i = 0; i < innode; i++) 
		inputLayer[i]->value = sampleIn[i];
}

void BpNet::setOutput(static vector<double> sampleOut)
{
	for (int i = 0; i < outnode; i++) 
		outputLayer[i]->rightout = sampleOut[i];
}

void BpNet::readNeural()
{
	FILE *stream0; //用来读取权值
	FILE *stream1; //用来读取偏移
	float w;
	if (inherit)
	{
		if ((stream0 = fopen("winit.txt", "r")) == NULL)
		{
			cout << "打开文件失败!";
			exit(1);
		}
		for (int i = 0; i < innode; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				fscanf(stream0, "%f", &w);
				inputLayer[i]->weight[j] = w;
			}
		}
		for (int i = 0; i < hidelayer - 1; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				for (int k = 0; k < hidenode; k++)
				{
					fscanf(stream0, "%f", &w);
					hiddenLayer[i][j]->weight[k] = w;
				}
			}
		}
		for (int i = 0; i < hidenode; i++)
		{
			for (int j = 0; j < outnode; j++)
			{
				fscanf(stream0, "%f", &w);
				hiddenLayer[hidelayer - 1][i]->weight[j] = w;
			}
		}
		fclose(stream0);
		//各点偏差读取
		if ((stream1 = fopen("binit.txt", "r")) == NULL)
		{
			cout << "创建文件失败!";
			exit(1);
		}
		for (int i = 0; i < hidelayer; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				fscanf(stream1, "%f", &w);
				hiddenLayer[i][j]->bias = w;
			}
		}
		for (int i = 0; i < outnode; i++)
		{
			fscanf(stream1, "%f", &w);
			outputLayer[i]->bias = w;
		}
		fclose(stream1);
	}
	else
	{
		if ((stream0 = fopen("w.txt", "r")) == NULL)
		{
			cout << "打开文件失败!";
			exit(1);
		}
		for (int i = 0; i < innode; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				fscanf(stream0, "%f", &w);
				inputLayer[i]->weight[j] = w;
			}
		}
		for (int i = 0; i < hidelayer - 1; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				for (int k = 0; k < hidenode; k++)
				{
					fscanf(stream0, "%f", &w);
					hiddenLayer[i][j]->weight[k] = w;
				}
			}
		}
		for (int i = 0; i < hidenode; i++)
		{
			for (int j = 0; j < outnode; j++)
			{
				fscanf(stream0, "%f", &w);
				hiddenLayer[hidelayer - 1][i]->weight[j] = w;
			}
		}
		fclose(stream0);
		//各点偏差读取
		if ((stream1 = fopen("b.txt", "r")) == NULL)
		{
			cout << "创建文件失败!";
			exit(1);
		}
		for (int i = 0; i < hidelayer; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				fscanf(stream1, "%f", &w);
				hiddenLayer[i][j]->bias = w;
			}
		}
		for (int i = 0; i < outnode; i++)
		{
			fscanf(stream1, "%f", &w);
			outputLayer[i]->bias = w;
		}
		fclose(stream1);
	}
	
}

void BpNet::writeNeural()
{
	FILE *stream0;//用来输出权值
	FILE *stream1;//用来输出偏移

	//各点权值写入  
	if ((stream0 = fopen("w.txt", "w+")) == NULL)
	{
		cout << "创建文件失败!";
		exit(1);
	}
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{
			fprintf(stream0, "%f\n", inputLayer[i]->weight[j]);
		}
	}
	for (int i = 0; i < hidelayer - 1; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{
			for (int k = 0; k < hidenode; k++)
				fprintf(stream0, "%f\n", hiddenLayer[i][j]->weight[k]);
		}
	}
	for (int i = 0; i < hidenode; i++)
	{
		for (int j = 0; j < outnode; j++)
		{
			fprintf(stream0, "%f\n", hiddenLayer[hidelayer - 1][i]->weight[j]);
		}
	}
	fclose(stream0);
	//各点偏差写入
	if ((stream1 = fopen("b.txt", "w+")) == NULL)
	{
		cout << "创建文件失败!";
		exit(1);
	}
	for (int i = 0; i < hidelayer; i++)
	{
		for (int j = 0; j < hidenode; j++)
			fprintf(stream1, "%f\n", hiddenLayer[i][j]->bias);
	}
	for (int i = 0; i < outnode; i++)
		fprintf(stream1, "%f\n", outputLayer[i]->bias);
	fclose(stream1);
}

void BpNet::AIadjust()
{
	if (lastError != 0.f)
	{
		if (1.02 * lastError >= error)
		{
			if (learningrate <= 0.9)
				learningrate *= 1.05;
		/*	alpher = 0.95;*/
		}
		else if (1.04 * lastError < error)
		{
			double m = 1.0 * lastError / error;
			if (learningrate >= 0.01)
			{
				if (m >= 0.4)
					learningrate *= 0.7;
				else
					learningrate *= (m * exp(1 - m));
			}
			//if (1.04 * m < 1.0)
			//	alpher = 0.f;
		}
	}
	
}
//暂时over
void BpNet::protectPara()
{
	if (lastError == error)
		repeatTime++;
	else
		repeatTime = 0;
	if (repeatTime == 6)
	{
		repeatTime = 0;
		srand((unsigned)time(NULL));
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == hidelayer - 1)
			{
				for (int j = 0; j < hidenode; j++)
				{
					hiddenLayer[i][j]->bias = get_11Random();
					hiddenLayer[i][j]->oldBD = 0.f;
					hiddenLayer[i][j]->value = 0.f;
					for (int k = 0; k < outnode; k++)
					{
						hiddenLayer[i][j]->weight[k] = (get_11Random());
						hiddenLayer[i][j]->wDeltaSum[k] = (0.f);
						hiddenLayer[i][j]->oldWD[k] = (0.f);
					}
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					hiddenLayer[i][j]->bias = get_11Random();
					hiddenLayer[i][j]->oldBD = 0.f;
					hiddenLayer[i][j]->value = 0.f;
					for (int k = 0; k < hidenode; k++)
					{
						hiddenLayer[i][j]->weight[k] = (get_11Random());
						hiddenLayer[i][j]->wDeltaSum[k] = (0.f);
						hiddenLayer[i][j]->oldWD[k] = (0.f);
					}
				}
			}
		}

		// 初始化输出层
		for (int i = 0; i < outnode; i++)
		{
			outputLayer[i]->bias = get_11Random();
			outputLayer[i]->oldBD = 0.f;
			outputLayer[i]->value = 0.f;
		}
	}
}