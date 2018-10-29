#include <iostream>
#include <iomanip>
#include <string>
#include <ctype.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op) \
{ \
	TickMeter tm; \
	tm.start(); \
	op; \
	tm.stop(); \
	cout << tm.getTimeSec() << " sec" << endl; \
}

static Ptr<cv::superres::DenseOpticalFlowExt> createOptFlow(const string& name, bool useGpu)
{
	if (name == "farneback")
	{
		if (useGpu)
			return cv::superres::createOptFlow_Farneback_CUDA();
		else
			return cv::superres::createOptFlow_Farneback();
	}
	/*else if (name == "simple")
	return createOptFlow_Simple();*/
	else if (name == "tvl1")
	{
		if (useGpu)
			return cv::superres::createOptFlow_DualTVL1_CUDA();
		else
			return cv::superres::createOptFlow_DualTVL1();
	}
	else if (name == "brox")
		return cv::superres::createOptFlow_Brox_CUDA();
	else if (name == "pyrlk")
		return cv::superres::createOptFlow_PyrLK_CUDA();
	else
		cerr << "Incorrect Optical Flow algorithm - " << name << endl;

	return Ptr<cv::superres::DenseOpticalFlowExt>();
}

int main(int argc, const char* argv[])
{
	/*
		"{ v video      |           | Input video (mandatory)}"
		"{ o output     |           | Output video }"
		"{ s scale      | 4         | Scale factor }"
		"{ i iterations | 180       | Iteration count }"
		"{ t temporal   | 4         | Radius of the temporal search area }"
		"{ f flow       | farneback | Optical flow algorithm (farneback, tvl1, brox, pyrlk) }"
		"{ g gpu        | false     | CPU as default device, cuda for CUDA }"
		"{ h help       | false     | Print help message }"
		);*/

	const string inputVideoName = "bike.avi";

	const int scale = 2;  ///超分辨率放大倍数 2*2
	const int iterations = 180;
	const int temporalAreaRadius = 4;  ///时间搜索区域的半径
	const string optFlow = "brox";  ////(farneback, tvl1, brox, pyrlk)
	string gpuOption = "gpu";

	std::transform(gpuOption.begin(), gpuOption.end(), gpuOption.begin(), ::tolower);

	bool useCuda = gpuOption.compare("cuda") == 1;  ///使用cuda 
	Ptr<SuperResolution> superRes;

	if (useCuda)
		superRes = createSuperResolution_BTVL1_CUDA();
	else
		superRes = createSuperResolution_BTVL1();

	Ptr<cv::superres::DenseOpticalFlowExt> of = createOptFlow(optFlow, useCuda);

	if (of.empty())
		return EXIT_FAILURE;
	superRes->setOpticalFlow(of);

	superRes->setScale(scale);
	superRes->setIterations(iterations);
	superRes->setTemporalAreaRadius(temporalAreaRadius);

	Ptr<FrameSource> frameSource;
	if (useCuda)
	{
		
		try
		{
			frameSource = createFrameSource_Video_CUDA(inputVideoName);
			Mat frame;
			frameSource->nextFrame(frame);
		}
		catch (const cv::Exception&)
		{
			frameSource.release();
		}
	}
	if (!frameSource)
		frameSource = createFrameSource_Video(inputVideoName);

	
	{
		Mat frame;
		frameSource->nextFrame(frame);
		cout << "Input           : " << inputVideoName << " " << frame.size() << endl;
		cout << "Scale factor    : " << scale << endl;
		cout << "Iterations      : " << iterations << endl;
		cout << "Temporal radius : " << temporalAreaRadius << endl;
		cout << "Optical Flow    : " << optFlow << endl;
		cout << "Mode            : " << (useCuda ? "CUDA" : "CPU") << endl;

		imshow("src", frame);
	}

	superRes->setInput(frameSource);

	
	

	for (int i = 0;; ++i)
	{
		cout << '[' << setw(3) << i << "] : " << flush;
		Mat result;

		MEASURE_TIME(superRes->nextFrame(result));

		if (result.empty())
			break;

		imshow("超分辨率重构", result);

		if (waitKey(1000) > 0)
			break;

	
	}

	return 0;
}