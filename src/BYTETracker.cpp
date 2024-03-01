#include "BYTETracker.h"
#include <fstream>

const char* class_names[80] = {
		 "person",
		 "bicycle",
		 "car",
		 "motorcycle",
		 "airplane",
		 "bus",
		 "train",
		 "truck",
		 "boat",
		 "traffic light",
		 "fire hydrant",
		 "stop sign",
		 "parking meter",
		 "bench",
		 "bird",
		 "cat",
		 "dog",
		 "horse",
		 "sheep",
		 "cow",
		 "elephant",
		 "bear",
		 "zebra",
		 "giraffe",
		 "backpack",
		 "umbrella",
		 "handbag",
		 "tie",
		 "suitcase",
		 "frisbee",
		 "skis",
		 "snowboard",
		 "sports ball",
		 "kite",
		 "baseball bat",
		 "baseball glove",
		 "skateboard",
		 "surfboard",
		 "tennis racket",
		 "bottle",
		 "wine glass",
		 "cup",
		 "fork",
		 "knife",
		 "spoon",
		 "bowl",
		 "banana",
		 "apple",
		 "sandwich",
		 "orange",
		 "broccoli",
		 "carrot",
		 "hot dog",
		 "pizza",
		 "donut",
		 "cake",
		 "chair",
		 "couch",
		 "potted plant",
		 "bed",
		 "dining table",
		 "toilet",
		 "tv",
		 "laptop",
		 "mouse",
		 "remote",
		 "keyboard",
		 "cell phone",
		 "microwave",
		 "oven",
		 "toaster",
		 "sink",
		 "refrigerator",
		 "book",
		 "clock",
		 "vase",
		 "scissors",
		 "teddy bear",
		 "hair drier",
		 "toothbrush"
};
const size_t num_classes = sizeof(class_names) / sizeof(class_names[0]);
static std::vector<std::string> CLASSES(class_names, class_names + num_classes);

// drawTrackSC
static std::unordered_map<std::string, int> CLASS2ID;
static std::unordered_map<int, std::string> ID2CLASS;


// ---------- Constructor: do initializations
BYTETracker::BYTETracker(const int& n_classes,
	const int& frame_rate,
	const int& track_buffer,
	const float& high_det_thresh,
	const float& new_track_thresh,
	const float& high_match_thresh,
	const float& low_match_thresh,
	const float& unconfirmed_match_thresh) :
	m_high_det_thresh(high_det_thresh),   // >m_track_thresh as high(1st)
	m_new_track_thresh(new_track_thresh),  // >m_high_thresh as new track
	m_high_match_thresh(high_match_thresh),  // first match threshold
	m_low_match_thresh(low_match_thresh),  // second match threshold
	m_unconfirmed_match_thresh(unconfirmed_match_thresh)  // unconfired match to remain dets
{
	// ----- number of object classes
	this->m_N_CLASSES = n_classes;
	std::cout << "Total " << n_classes << " classes of object to be tracked.\n";

	this->m_frame_id = 0;
	//this->m_max_time_lost = int(frame_rate / 30.0 * track_buffer);
	this->m_max_time_lost = track_buffer;
	std::cout << "Max lost time(number of frames): " << m_max_time_lost << std::endl;
	std::cout << "MCMOT tracker inited done" << std::endl;
}

BYTETracker::~BYTETracker()
{
}

std::unordered_map<int, std::vector<STrack>> BYTETracker::updateMCMOT(std::vector<detect_result>& objects)
{
	////////////////// Step 1: Get detections //////////////////
	this->m_frame_id++;

	// ---------- Track's track id initialization
	if (this->m_frame_id == 1)
	{
		STrack::init_trackid_dict(this->m_N_CLASSES);
	}

	std::unordered_map<int, std::vector<STrack>> activated_stracks;
	std::unordered_map<int, std::vector<STrack>> refind_stracks;
	std::unordered_map<int, std::vector<STrack>> removed_stracks;
	std::unordered_map<int, std::vector<STrack>> lost_stracks;
	std::unordered_map<int, std::vector<STrack>> output_stracks;

	std::unordered_map<int, std::vector<STrack*>> unconfirmed;
	std::unordered_map<int, std::vector<STrack*>> tracked_stracks;
	std::unordered_map<int, std::vector<STrack*>> strack_pool;

	// get detections
	std::unordered_map<int, std::vector<std::vector<float>>> bboxes_dict;
	std::unordered_map<int, std::vector<float>> scores_dict;

	// 对检测出的目标框进行处理
	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			std::vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].box.x;                           // x1
			tlbr_[1] = objects[i].box.y;                           // y1
			tlbr_[2] = objects[i].box.x + objects[i].box.width;    // x2
			tlbr_[3] = objects[i].box.y + objects[i].box.height;   // y2

			const float& score = objects[i].confidence;
			const int& cls_id = objects[i].classId;

			// printf("-----------------class id : %d \n", cls_id);

			bboxes_dict[cls_id].push_back(tlbr_);
			scores_dict[cls_id].push_back(score);

			cls_id_set.insert(cls_id);  // store class id
		}
	}

	// ---------- Processing each object classes
	// ----- Build bbox_dict and score_dict
	for (std::set<int>::iterator it = cls_id_set.begin(); it != cls_id_set.end(); ++it) {
		
		// class bboxes
		std::vector<std::vector<float>>& cls_bboxes = bboxes_dict[*it];

		// class scores
		const std::vector<float>& cls_scores = scores_dict[*it];

		// skip classes of empty detections of objects
		if (cls_bboxes.size() == 0)
		{
			continue;
		}

		// temporary containers
		std::vector<STrack> detections;
		std::vector<STrack> detections_low;
		std::vector<STrack> detections_cp;
		std::vector<STrack> tracked_stracks_swap;
		std::vector<STrack> resa, resb;
		std::vector<STrack*> r_tracked_stracks;

		// detections classifications
		for (int i = 0; i < cls_bboxes.size(); ++i)
		{
			std::vector<float>& tlbr_ = cls_bboxes[i];
			const float& score = cls_scores[i];

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, *it);
			if (score > this->m_high_det_thresh)  // high confidence dets
			{
				detections.push_back(strack);
			}
			else  // low confidence dets
			{
				detections_low.push_back(strack);
			}
		}

		// Add newly detected tracklets to tracked_stracks 将新检测出的轨迹加入到追踪的轨迹
		for (int i = 0; i < this->m_tracked_stracks_dict[*it].size(); i++)
		{
			// 如果轨迹没有被激活，将该轨迹放到unconfirmed里面
			if (!this->m_tracked_stracks_dict[*it][i].is_activated)
				unconfirmed[*it].push_back(&this->m_tracked_stracks_dict[*it][i]);
			// 如果激活了，直接将其放到tracked_stracks中的
			else
				tracked_stracks[*it].push_back(&this->m_tracked_stracks_dict[*it][i]);
		}

		////////////////// Step 2: First association, with IoU //////////////////
		// 将那些失去追踪的track(少于30帧)放到需要追踪的stracks的集合中
		strack_pool[*it] = joint_stracks(tracked_stracks[*it], this->m_lost_stracks_dict[*it]);
		STrack::multi_predict(strack_pool[*it], this->m_kalman_filter);  // 使用卡尔曼滤波预测下一帧目标框出现的位置

		std::vector< std::vector<float> > dists;
		int dist_size = 0, dist_size_size = 0;
		// 计算预测的目标框和高置信度目标框之间的iou, 获取距离
		dists = iou_distance(strack_pool[*it], detections, dist_size, dist_size_size);

		// 使用匈牙利算法对预测目标框和高置信度目标框进行匹配
		std::vector< std::vector<int> > matches;
		std::vector<int> u_track, u_detection;
		linear_assignment(dists, dist_size, dist_size_size, this->m_high_match_thresh, matches, u_track, u_detection);

		// 先遍历那些已经成功匹配的目标框
		for (int i = 0; i < matches.size(); i++)
		{
			STrack* track = strack_pool[*it][matches[i][0]];
			STrack* det = &detections[matches[i][1]];
			// 如果匹配的track已经被追踪过
			if (track->state == TrackState::Tracked)
			{
				// 更新det，赋予frame_id
				track->update(*det, this->m_frame_id);
				// 将track放到activated_stracks工作
				activated_stracks[*it].push_back(*track);
			}
			else
			{
				track->re_activate(*det, this->m_frame_id, false);
				// 将重新被匹配的strack放到refind_stracks中
				refind_stracks[*it].push_back(*track);
			}
		}

		////////////////// Step 3: Second association, using low score dets //////////////////
		// 遍历那些高置信度框中未匹配的目标框，将其放到detections_cp中
		for (int i = 0; i < u_detection.size(); i++)
		{
			detections_cp.push_back(detections[u_detection[i]]);
		}
		// 处理那些低置信度的目标框
		detections.clear();
		detections.assign(detections_low.begin(), detections_low.end());

		// 遍历那些先前未匹配的track
		for (int i = 0; i < u_track.size(); i++)
		{
			// 如果当前未匹配的track的状态先前为tracked状态（就是先前已经成功匹配了）
			if (strack_pool[*it][u_track[i]]->state == TrackState::Tracked)
			{
				// 将其放到r_tracked_stracks中
				r_tracked_stracks.push_back(strack_pool[*it][u_track[i]]);
			}
		}

		// 计算先前未成功匹配的track和低置信度目标框之间的iou_distance
		dists.clear();
		dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

		// 使用匈牙利算法进行匹配
		matches.clear();
		u_track.clear();
		u_detection.clear();
		linear_assignment(dists, dist_size, dist_size_size, this->m_low_match_thresh, matches, u_track, u_detection);

		// 遍历那些成功匹配的track和目标框
		for (int i = 0; i < matches.size(); i++)
		{
			STrack* track = r_tracked_stracks[matches[i][0]];
			STrack* det = &detections[matches[i][1]];
			// 如果先前未匹配track状态是tracked的状态，将track重新激活
			if (track->state == TrackState::Tracked)
			{
				track->update(*det, this->m_frame_id);
				activated_stracks[*it].push_back(*track);
			}
			else
			{
				track->re_activate(*det, this->m_frame_id, false);
				refind_stracks[*it].push_back(*track);
			}
		}

		// 遍历第二次未成功匹配的track
		for (int i = 0; i < u_track.size(); i++)
		{
			// 找出第二次未成功匹配的track在r_tracked_stracks里面
			STrack* track = r_tracked_stracks[u_track[i]];
			// 如果state不是Lost,将其track标为lost
			if (track->state != TrackState::Lost)
			{
				track->mark_lost();
				lost_stracks[*it].push_back(*track);
			}
		}

		// Deal with unconfirmed tracks, usually tracks with only one beginning frame
		// 处理那些高置信度但是未成功匹配的目标框
		detections.clear();
		detections.assign(detections_cp.begin(), detections_cp.end());

		// 计算其与先前state为未追踪状态的track
		dists.clear();
		dists = iou_distance(unconfirmed[*it], detections, dist_size, dist_size_size);

		matches.clear();
		std::vector<int> u_unconfirmed;
		u_detection.clear();
		linear_assignment(dists, dist_size, dist_size_size, this->m_unconfirmed_match_thresh, matches, u_unconfirmed, u_detection);

		for (int i = 0; i < matches.size(); i++)
		{
			STrack* track = unconfirmed[*it][matches[i][0]];
			STrack* det = &detections[matches[i][1]];
			// 更新det，赋予frame_id
			track->update(*det, this->m_frame_id);
			// 将track放到activated_stracks工作
			activated_stracks[*it].push_back(*track);
		}

		for (int i = 0; i < u_unconfirmed.size(); i++)
		{
			STrack* track = unconfirmed[*it][u_unconfirmed[i]];
			track->mark_removed();
			removed_stracks[*it].push_back(*track);
		}

		////////////////// Step 4: Init new stracks //////////////////
		// 针对那些几次未成功匹配的目标框，如果该目标框的置信度高于一定阈值，将其放到activated_stracks
		for (int i = 0; i < u_detection.size(); i++)
		{
			STrack* track = &detections[u_detection[i]];
			if (track->score < this->m_new_track_thresh)
				continue;
			track->activate(this->m_kalman_filter, this->m_frame_id);
			activated_stracks[*it].push_back(*track);
		}

		////////////////// Step 5: Update state //////////////////
		// 遍历那些已经失追的track，将那些失追频率大于阈值的track放到removed_stracks
		for (int i = 0; i < this->m_lost_stracks_dict[*it].size(); i++)
		{
			STrack& track = this->m_lost_stracks_dict[*it][i];
			if (this->m_frame_id - track.end_frame() > this->m_max_time_lost)
			{
				track.mark_removed();
				removed_stracks[*it].push_back(track);
			}
		}

		// 遍历那些tracked_stack
		for (int i = 0; i < this->m_tracked_stracks_dict[*it].size(); i++)
		{
			if (this->m_tracked_stracks_dict[*it][i].state == TrackState::Tracked)
			{
				tracked_stracks_swap.push_back(this->m_tracked_stracks_dict[*it][i]);
			}
		}
		this->m_tracked_stracks_dict[*it].clear();
		this->m_tracked_stracks_dict[*it].assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

		this->m_tracked_stracks_dict[*it] = joint_stracks(this->m_tracked_stracks_dict[*it], activated_stracks[*it]);
		this->m_tracked_stracks_dict[*it] = joint_stracks(this->m_tracked_stracks_dict[*it], refind_stracks[*it]);

		//std::cout << activated_stracks.size() << std::endl;
		this->m_lost_stracks_dict[*it] = sub_stracks(this->m_lost_stracks_dict[*it], this->m_tracked_stracks_dict[*it]);
		for (int i = 0; i < lost_stracks[*it].size(); i++)
		{
			this->m_lost_stracks_dict[*it].push_back(lost_stracks[*it][i]);
		}

		this->m_lost_stracks_dict[*it] = sub_stracks(this->m_lost_stracks_dict[*it], this->m_removed_stracks_dict[*it]);
		for (int i = 0; i < removed_stracks[*it].size(); i++)
		{
			this->m_removed_stracks_dict[*it].push_back(removed_stracks[*it][i]);
		}

		remove_duplicate_stracks(resa, resb, this->m_tracked_stracks_dict[*it], this->m_lost_stracks_dict[*it]);

		this->m_tracked_stracks_dict[*it].clear();
		this->m_tracked_stracks_dict[*it].assign(resa.begin(), resa.end());

		this->m_lost_stracks_dict[*it].clear();
		this->m_lost_stracks_dict[*it].assign(resb.begin(), resb.end());

		for (int i = 0; i < this->m_tracked_stracks_dict[*it].size(); i++)
		{
			if (this->m_tracked_stracks_dict[*it][i].is_activated)
			{
				output_stracks[*it].push_back(this->m_tracked_stracks_dict[*it][i]);
			}
		}
	}
	return output_stracks;
}


std::vector<STrack> BYTETracker::update(const std::vector<detect_result>& objects)
{
	////////////////// Step 1: Get detections //////////////////
	this->m_frame_id++;

	std::vector<STrack> activated_stracks;
	std::vector<STrack> refind_stracks;
	std::vector<STrack> removed_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> detections;
	std::vector<STrack> detections_low;

	std::vector<STrack> detections_cp;
	std::vector<STrack> tracked_stracks_swap;
	std::vector<STrack> resa, resb;
	std::vector<STrack> output_stracks;

	std::vector<STrack*> unconfirmed;
	std::vector<STrack*> tracked_stracks;
	std::vector<STrack*> strack_pool;
	std::vector<STrack*> r_tracked_stracks;

	STrack::init_trackid_dict(this->m_N_CLASSES);

	// 对检测出的目标框进行处理
	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			std::vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].box.x;                           // x1
			tlbr_[1] = objects[i].box.y;                           // y1
			tlbr_[2] = objects[i].box.x + objects[i].box.width;    // x2
			tlbr_[3] = objects[i].box.y + objects[i].box.height;   // y2

			const float& score = objects[i].confidence;

			// 先根据track_thresh将目标框分为高置信度目标框和低置信度目标框
			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, objects[i].classId);
			if (score >= this->m_high_det_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
		}
	}

	// Add newly detected tracklets to tracked_stracks 将新检测出的轨迹加入到追踪的轨迹
	for (int i = 0; i < this->m_tracked_stracks.size(); i++)
	{
		// 如果轨迹没有被激活，将该轨迹放到unconfirmed里面
		if (!this->m_tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->m_tracked_stracks[i]);
		// 如果激活了，直接将其放到tracked_stracks中的
		else
			tracked_stracks.push_back(&this->m_tracked_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU //////////////////
	// 将那些失去追踪的track(少于30帧)放到需要追踪的stracks的集合中
	strack_pool = joint_stracks(tracked_stracks, this->m_lost_stracks);
	STrack::multi_predict(strack_pool, this->m_kalman_filter);  // 使用卡尔曼滤波预测下一帧目标框出现的位置

	std::vector< std::vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	// 计算预测的目标框和高置信度目标框之间的iou, 获取距离
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	// 使用匈牙利算法对预测目标框和高置信度目标框进行匹配
	std::vector< std::vector<int> > matches;
	std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, this->m_high_match_thresh, matches, u_track, u_detection);

	// 先遍历那些已经成功匹配的目标框
	for (int i = 0; i < matches.size(); i++)
	{
		STrack* track = strack_pool[matches[i][0]];
		STrack* det = &detections[matches[i][1]];
		// 如果匹配的track已经被追踪过
		if (track->state == TrackState::Tracked)
		{
			// 更新det，赋予frame_id
			track->update(*det, this->m_frame_id);
			// 将track放到activated_stracks工作
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->m_frame_id, false);
			// 将重新被匹配的strack放到refind_stracks中
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	// 遍历那些高置信度框中未匹配的目标框，将其放到detections_cp中
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	// 处理那些低置信度的目标框
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());

	// 遍历那些先前未匹配的track
	for (int i = 0; i < u_track.size(); i++)
	{
		// 如果当前未匹配的track的状态先前为tracked状态（就是先前已经成功匹配了）
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			// 将其放到r_tracked_stracks中
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	// 计算先前未成功匹配的track和低置信度目标框之间的iou_distance
	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	// 使用匈牙利算法进行匹配
	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	// 遍历那些成功匹配的track和目标框
	for (int i = 0; i < matches.size(); i++)
	{
		STrack* track = r_tracked_stracks[matches[i][0]];
		STrack* det = &detections[matches[i][1]];
		// 如果先前未匹配track状态是tracked的状态，将track重新激活
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->m_frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->m_frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	// 遍历第二次未成功匹配的track
	for (int i = 0; i < u_track.size(); i++)
	{
		// 找出第二次未成功匹配的track在r_tracked_stracks里面
		STrack* track = r_tracked_stracks[u_track[i]];
		// 如果state不是Lost,将其track标为lost
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	// 处理那些高置信度但是未成功匹配的目标框
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	// 计算其与先前state为未追踪状态的track
	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	// 如果成功匹配，将其状态改为tracked，加入到activated_stracks中
	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->m_frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	// 针对那些几次未成功匹配的track将其标志为移除
	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack* track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	// 针对那些几次未成功匹配的目标框，如果该目标框的置信度高于一定阈值，将其放到activated_stracks
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack* track = &detections[u_detection[i]];
		if (track->score < this->m_new_track_thresh)
			continue;
		track->activate(this->m_kalman_filter, this->m_frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	// 遍历那些已经失追的track，将那些失追频率大于阈值的track放到removed_stracks
	for (int i = 0; i < this->m_lost_stracks.size(); i++)
	{
		if (this->m_frame_id - this->m_lost_stracks[i].end_frame() > this->m_max_time_lost)
		{
			this->m_lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->m_lost_stracks[i]);
		}
	}

	// 遍历那些tracked_stack
	for (int i = 0; i < this->m_tracked_stracks.size(); i++)
	{
		if (this->m_tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->m_tracked_stracks[i]);
		}
	}
	this->m_tracked_stracks.clear();
	this->m_tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->m_tracked_stracks = joint_stracks(this->m_tracked_stracks, activated_stracks);
	this->m_tracked_stracks = joint_stracks(this->m_tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->m_lost_stracks = sub_stracks(this->m_lost_stracks, this->m_tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->m_lost_stracks.push_back(lost_stracks[i]);
	}

	this->m_lost_stracks = sub_stracks(this->m_lost_stracks, this->m_removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->m_removed_stracks.push_back(removed_stracks[i]);
	}

	remove_duplicate_stracks(resa, resb, this->m_tracked_stracks, this->m_lost_stracks);

	this->m_tracked_stracks.clear();
	this->m_tracked_stracks.assign(resa.begin(), resa.end());
	this->m_lost_stracks.clear();
	this->m_lost_stracks.assign(resb.begin(), resb.end());

	for (int i = 0; i < this->m_tracked_stracks.size(); i++)
	{
		if (this->m_tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->m_tracked_stracks[i]);
		}
	}
	return output_stracks;
}

// draw tracking
void BYTETracker::initMappings()
{
	// printf("CLASSES SIZE: %d\n", CLASSES.size());
	for (int i = 0; i < CLASSES.size(); ++i)
	{
		CLASS2ID[CLASSES[i]] = i;
		ID2CLASS[i] = CLASSES[i];
		// printf("ID2CLASS[i]: %s\n", ID2CLASS[i]);
	}
}

void BYTETracker::drawTrackSC(const std::vector<STrack>& output_stracks,
	const int& num_frames, const int& total_ms,
	cv::Mat& img)
{
	for (int i = 0; i < output_stracks.size(); ++i)
	{
		cv::Scalar s = BYTETracker::get_color(output_stracks[i].track_id);
		const std::vector<float>& tlwh = output_stracks[i].tlwh;

		// Draw class name
		cv::putText(img,
			ID2CLASS[output_stracks[i].class_id],
			cv::Point((int)tlwh[0], (int)tlwh[1] - 5),
			0,
			0.6,
			cv::Scalar(0, 255, 255),
			2,
			cv::LINE_AA);

		// Draw track id
		cv::putText(img,
			cv::format("%d", output_stracks[i].track_id),  // track id
			cv::Point((int)tlwh[0], (int)tlwh[1] - 12),
			0,
			0.6,
			cv::Scalar(0, 255, 255),
			2,
			cv::LINE_AA);

		// Draw bounding box
		cv::rectangle(img,
			cv::Rect((int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3]),
			s,
			2);
	}

	cv::putText(img,
		cv::format("frame: %d fps: %d num: %d",
			num_frames, num_frames * 1000000 / total_ms, output_stracks.size()),
		cv::Point(0, 30),
		0,
		0.6,
		cv::Scalar(0, 0, 255),
		2,
		cv::LINE_AA);
}


void BYTETracker::drawTrackMC(const std::unordered_map<int, std::vector<STrack>>& output_stracks_dict,
	const int& num_frames, const int& total_ms,
	cv::Mat& img)
{
	int total_obj_count = 0;

	// hash table traversing
	for (std::set<int>::iterator it = cls_id_set.begin(); it != cls_id_set.end(); it++) 
	{
		const std::vector<STrack>& output_stracks = output_stracks_dict.at(*it);
		total_obj_count += (int)output_stracks.size();
		for (int i = 0; i < output_stracks.size(); ++i)
		{
			cv::Scalar s = BYTETracker::get_color(output_stracks[i].track_id);
			const std::vector<float>& tlwh = output_stracks[i].tlwh;
			//const int& x0 = tlwh[0];

			// std::cout << "output_stracks.class_id: " << output_stracks[i].class_id << std::endl;

			// Draw class name
			cv::putText(img,
				// ID2CLASS[output_stracks[i].class_id],
				ID2CLASS[*it],
				cv::Point((int)tlwh[0], (int)tlwh[1] - 5),
				0,
				0.6,
				cv::Scalar(0, 255, 255),
				2,
				cv::LINE_AA);

			// Draw track id
			cv::putText(img,
				cv::format("%d", output_stracks[i].track_id),  // track id
				cv::Point((int)tlwh[0], (int)tlwh[1] - 15),
				0,
				0.6,
				cv::Scalar(0, 255, 255),
				2,
				cv::LINE_AA);

			// Draw bounding box
			cv::rectangle(img,
				cv::Rect((int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3]),
				s,
				2);
		}
	}

	cv::putText(img,
		cv::format("frame: %d fps: %d num: %d",
			num_frames, num_frames * 1000000 / total_ms, total_obj_count),
		cv::Point(0, 30),
		0,
		0.6,
		cv::Scalar(0, 0, 255),
		2,
		cv::LINE_AA);
}
