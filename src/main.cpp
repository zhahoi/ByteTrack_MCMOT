#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "YOLOv5Detector.h"
#include "FeatureTensor.h"
#include "BYTETracker.h" //bytetrack

// Deep SORT parameter
const int nn_budget = 100;
const float max_cosine_distance = 0.2;

const int NUM_CLASSES(1);     // number of object classes
const int TRACK_BUFFER(240);  // frame number of tracking states buffers

// 2 Det/Track classification thresholds for the byte tracker
const float HIGH_DET_THRESH(0.5f);  // 0.5f > m_track_thresh as high(1st)
const float NEW_TRACK_THRESH(HIGH_DET_THRESH + 0.1f);   // > m_high_thresh as new track

// 3 Matching thresholds
const float HIGH_MATCH_THRESH(0.8f);  // 0.8f first match threshold
const float LOW_MATCH_THRESH(0.5f);  // 0.5f second match threshold
const float UNCONFIRMED_MATCH_THRESH(0.7f);  // 0.7: unconfirmed track match to remain dets

int main(int argc, char* argv[])
{
    //bytetrack
    int fps = 20;
    BYTETracker bytetracker(NUM_CLASSES, fps, TRACK_BUFFER,
        HIGH_DET_THRESH, NEW_TRACK_THRESH,
        HIGH_MATCH_THRESH, LOW_MATCH_THRESH,
        UNCONFIRMED_MATCH_THRESH);

    //-----------------------------------------------------------------------
    // 加载类别名称
    std::vector<std::string> classes;
    std::string file = "C:/CPlusPlus/MCMOT_ByteTrack/weights/coco_80_labels_list.txt";
    std::ifstream ifs(file);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    //-----------------------------------------------------------------------
    std::cout << "classes: " << classes.size() << std::endl;
  
    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
    detector->init(k_detect_model_path);

    std::cout << "begin read video" << std::endl;
    cv::VideoCapture capture("C:/CPlusPlus/MCMOT_ByteTrack/weights/people.mp4");

    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }
    std::cout << "end read video" << std::endl;
    std::vector<detect_result> results;
    int num_frames = 0;

    cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(1920, 1080));

    // ---------- Init mappings of CLASS_NAME and CLASS_ID
    bytetracker.initMappings();
    std::cout << "[I] mappings of class names and class ids are built.\n";

    while (true)
    {
        cv::Mat frame;

        if (!capture.read(frame)) // if not success, break loop
        {
            std::cout << "\n Cannot read the video file. please check your video.\n";
            break;
        }

        num_frames++;
        // detecting
        auto start = std::chrono::system_clock::now();
        detector->detect(frame, results);
        auto end = std::chrono::system_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //ms
        std::cout << "classes size: " << classes.size() << "," << "results size :" << results.size() << "," << "num_frames size:" << num_frames << std::endl;

        // 处理 detection results
        if (results.empty()) {
            std::cout << "No detection results in this frame." << std::endl;
            continue;
        }

        // tracking
        auto t_start = std::chrono::system_clock::now();

        std::vector<detect_result> objects;  // 存放检测框
        // 这里只针对于单个类别
        for (detect_result dr : results)
        {
            if (NUM_CLASSES == 1)
            {
                if (dr.classId == 0) //person
                {
                    objects.push_back(dr);
                }
            }
            // 针对多类别跟踪
            else if (NUM_CLASSES > 1)  // Multi-class tracking output
            {
                if (dr.classId == 2 || dr.classId == 5 || dr.classId == 9)
                {
                    objects.push_back(dr);
                }
            }

            // 增加类名访问时的索引范围检查
            if (dr.classId >= 0 && dr.classId < classes.size()) {
                std::cout << "Class name: " << classes[dr.classId] << " (Class ID: " << dr.classId << ")\n";
            } else {
                std::cerr << "Error: classId " << dr.classId << " out of range! Total classes: " << classes.size() << std::endl;
            }
        }

        // 确保检测框有效，避免出现空或无效数据
        if (objects.empty()) {
            std::cout << "No valid objects for tracking." << std::endl;
            continue;
        }

        // ---------- Update tracking results of current frame
        std::vector<STrack> output_stracks;   // single class
        std::unordered_map<int, std::vector<STrack>> output_stracks_dict;  // multi class

        if (NUM_CLASSES == 1)  // Single class tracking output
        {
            output_stracks = bytetracker.update(objects);
        }
        else if (NUM_CLASSES > 1)  // Multi-class tracking output
        {
            output_stracks_dict = bytetracker.updateMCMOT(objects);
        }

        // update time
        auto t_end = std::chrono::system_clock::now();
        total_ms += (int)std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

        // ----write tracking results for single class
        if (NUM_CLASSES == 1)
        {
            bytetracker.drawTrackSC(output_stracks, num_frames, total_ms, frame);
        }
        else if (NUM_CLASSES > 1)  // multi_class tracking
        {
            bytetracker.drawTrackMC(output_stracks_dict, num_frames, total_ms, frame);
        }

        cv::imshow("YOLOv5-6.x", frame);

        video.write(frame);

        if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }

        results.clear();
    }
    capture.release();
    video.release();
    cv::destroyAllWindows();
}
