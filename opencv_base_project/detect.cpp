#include "anchor_generator.h"
#include "detect.h"
#include<iostream>


using namespace std;
using namespace cv;


Detector::Detector(const string& model_file,
	const string& weights_file,
	const float confidence,
	const float nms)
{
	net_ = cv::dnn::readNetFromCaffe(model_file, weights_file);
	confidence_threshold = confidence;
	nms_threshold = nms;
}

/*
void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& outs)
{
	std::vector<int> indices;
	std::vector <cv::Rect>bboxes;
	std::vector<float> scores;
	for (unsigned i = 0; i < boxes.size(); i++) {
		int x = boxes[i].finalbox.x;
		int y = boxes[i].finalbox.y;
		int w = boxes[i].finalbox.width;
		int h = boxes[i].finalbox.height;
		bboxes.push_back(cv::Rect(x, y, w, h));
		scores.push_back(boxes[i].score);
	}

	cv::dnn::NMSBoxes(bboxes, scores, 0.4, 0.83, indices);
	outs.clear();
	for (int i : indices) {
		outs.push_back(boxes[i]);
	}
}
*/

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
	filterOutBoxes.clear();
	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	//descending sort
	sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx][0], boxes[tmp_i][0]);
			float inter_y1 = std::max(boxes[good_idx][1], boxes[tmp_i][1]);
			float inter_x2 = std::min(boxes[good_idx][2], boxes[tmp_i][2]);
			float inter_y2 = std::min(boxes[good_idx][3], boxes[tmp_i][3]);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
			float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

std::vector<Anchor> Detector::Detect(cv::Mat& img, cv::Size &blob_size)
{
	std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
	for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
		int stride = _feat_stride_fpn[i];
		int num = ac[i].Init(stride, anchor_cfg[stride], false);
	}

	Mat blob_input = dnn::blobFromImage(img, 1.0 / 127.5, blob_size, cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32FC1);
	std::vector<Mat> targets_blobs;
	net_.setInput(blob_input, "data");

	std::vector<String>  targets_node;
	for (int i = 0; i < _feat_stride_fpn.size(); ++i)
	{
		char clsname[128]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
		char regname[128]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
		char ptsname[128]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
		char typname[128]; sprintf(typname, "face_rpn_type_prob_stride%d", _feat_stride_fpn[i]);
		targets_node.push_back(clsname);
		targets_node.push_back(regname);
		targets_node.push_back(ptsname);
		targets_node.push_back(typname);
	}

	double t = (double)cv::getTickCount();
	net_.forward(targets_blobs, targets_node);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "forward cost:" << t << "s" << endl;
	/*
	for (int j = 0; j < targets_node.size(); j++) {
		char file_name[128];
		sprintf(file_name, "%s.xml", targets_node[j].c_str());
		FileStorage fs(file_name, FileStorage::WRITE);
		fs << "data" << targets_blobs[j];
		fs.release();
	}
	*/
	std::vector<Anchor> proposals;
	int index = 0;
	for (int i = 0; i < _feat_stride_fpn.size(); ++i)
	{
		cv::Mat clsBlob = targets_blobs[index++];
		cv::Mat regBlob = targets_blobs[index++];
		cv::Mat ptsBlob = targets_blobs[index++];
		cv::Mat mskBlob = targets_blobs[index++];
		ac[i].FilterAnchor(&clsBlob, &regBlob, &ptsBlob, &mskBlob, proposals, ratio_w, ratio_h, confidence_threshold);
	}

	// nms
	std::vector<Anchor> result;
	nms_cpu(proposals, nms_threshold, result);
	return result;
}


