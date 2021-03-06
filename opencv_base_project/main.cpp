#include"detect.h"

void test_video(Detector &detector)
{
	cv::VideoCapture vc("demo.mp4");
	cv::Mat img;
	while (true)
	{
		vc >> img;
		if (!img.data) {
			break;
		}

		std::vector<Anchor> result = detector.Detect(img, img.size());
		char strs[64];
		for (int i = 0; i < result.size(); i++) {
			cv::rectangle(img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(255, 255, 128), 1, 8, 0);
			cv::circle(img, result[i].pts[0], 2, cv::Scalar(255, 255, 0), -1);
			cv::circle(img, result[i].pts[1], 2, cv::Scalar(255, 255, 0), -1);
			cv::circle(img, result[i].pts[2], 2, cv::Scalar(255, 255, 0), -1);
			cv::circle(img, result[i].pts[3], 2, cv::Scalar(255, 255, 0), -1);
			cv::circle(img, result[i].pts[4], 2, cv::Scalar(255, 255, 0), -1);
			snprintf(strs, 64, "mask:%.2f", result[i].mask_score);
			cv::putText(img, strs, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), 1, 1.0, cv::Scalar(0, 255, 255));
		}

		cv::imshow("demo", img);
		if (cv::waitKey(1) == 'q') {
			break;
		}
	}
}

int main(int argc, char** argv) {

	const float confidence = 0.8;
	const float nms_threshold = 0.4;
	Detector detector("mnet_cov2.prototxt", "mnet_cov2.caffemodel", confidence, nms_threshold);

	test_video(detector);

	return 0;
}
