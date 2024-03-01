#pragma once

#include "STrack.h"
#include "YOLOv5Detector.h"
#include <string.h>

class BYTETracker
{
public:
	// BYTETracker(int frame_rate = 30, int track_buffer = 30);
	BYTETracker(
		const int& n_classes,
		const int& frame_rate,
		const int& track_buffer,
		const float& high_det_thresh,
		const float& new_track_thresh,
		const float& high_match_thresh,
		const float& low_match_thresh,
		const float& unconfirmed_match_thresh
	);
	~BYTETracker();

	std::vector<STrack> update(const std::vector<detect_result>& objects);
	// add MCMOT
	std::unordered_map<int, std::vector<STrack>> updateMCMOT(std::vector<detect_result>& objects);

	cv::Scalar get_color(int idx);

	// draw tracking
	void initMappings();
	void drawTrackSC(const std::vector<STrack>& output_stracks, const int& num_frames, const int& total_ms, cv::Mat& img);
	void drawTrackMC(const std::unordered_map<int, std::vector<STrack>>& output_stracks_dict, const int& num_frames, const int& total_ms, cv::Mat& img);

private:
	std::vector<STrack*> joint_stracks(std::vector<STrack*>& tlista, std::vector<STrack>& tlistb);
	std::vector<STrack> joint_stracks(std::vector<STrack>& tlista, std::vector<STrack>& tlistb);

	std::vector<STrack> sub_stracks(std::vector<STrack>& tlista, std::vector<STrack>& tlistb);
	void remove_duplicate_stracks(std::vector<STrack>& resa, std::vector<STrack>& resb, std::vector<STrack>& stracksa, std::vector<STrack>& stracksb);

	void linear_assignment(std::vector< std::vector<float> >& cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector< std::vector<int> >& matches, std::vector<int>& unmatched_a, std::vector<int>& unmatched_b);
	std::vector< std::vector<float> > iou_distance(std::vector<STrack*>& atracks, std::vector<STrack>& btracks, int& dist_size, int& dist_size_size);
	std::vector< std::vector<float> > iou_distance(std::vector<STrack>& atracks, std::vector<STrack>& btracks);
	std::vector< std::vector<float> > ious(std::vector< std::vector<float> >& atlbrs, std::vector< std::vector<float> >& btlbrs);

	double lapjv(const  std::vector< std::vector<float> >& cost, std::vector<int>& rowsol, std::vector<int>& colsol,
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
	float m_high_det_thresh;
	float m_new_track_thresh;
	float m_high_match_thresh;
	float m_low_match_thresh;
	float m_unconfirmed_match_thresh;
	int m_frame_id;
	int m_max_time_lost;

	// tracking object class number
	int m_N_CLASSES;

	// 3 containers of the tracker
	std::vector<STrack> m_tracked_stracks;
	std::vector<STrack> m_lost_stracks;
	std::vector<STrack> m_removed_stracks;

	std::unordered_map<int, std::vector<STrack>> m_tracked_stracks_dict;
	std::unordered_map<int, std::vector<STrack>> m_lost_stracks_dict;
	std::unordered_map<int, std::vector<STrack>> m_removed_stracks_dict;

	byte_kalman::ByteKalmanFilter m_kalman_filter;

	// cls_id_set
	std::set<int> cls_id_set;
};
