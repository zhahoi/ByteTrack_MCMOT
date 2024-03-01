# ByteTrack_MCMOT

# MOT(Multi-object tracking) and MCMOT(Multi-class Multi-object tracking ) using yolov5 with C++ support bytetrack

## 前言

该仓库是为了学习Bytetrack目标跟踪算法而建的，代码中的Bytetrack单目标跟踪C++代码实现参考于[DeepSORT](https://github.com/shaoshengsong/DeepSORT)仓库。多类别多目标跟踪算法实现参考于[ByteTrack-MCMOT-TensorRT](https://github.com/CaptainEven/ByteTrack-MCMOT-TensorRT)这个仓库。

代码采用C++实现，目标检测支持YOLOv5 6.2,跟踪支持bytetrack。
检测模型可以直接从YOLOv5官网，导出onnx使用。
特征提取可以自己训练，导出onnx使用，onnxruntime cpu 推理，方便使用。

本文源码地址：

```c
https://github.com/zhahoi/ByteTrack_MCMOT
```

## 测试环境与依赖

- Microsoft Visual Studio 2019
  
- opencv-4.5.5
  
- eigen-3.4.0
  
- onnxruntime-win-x64-gpu-1.16.1
  
- cuda v11.8
  

## 文件下载

代码中使用的权重和检测类别，我是下载自[DeepSORT](https://github.com/shaoshengsong/DeepSORT)该仓库提供的百度云链接，这里贴上下载地址：

百度网盘 
链接：`https://pan.baidu.com/s/1igjNK2ty-H5AU_Ut08pkoA` 
提取码：0000

我们需要的内容有以下两个：

```
yolov5s.onnx
coco_80_labels_list.txt 
```

测试的视频我就不提供了，可以自由选择。

## 参数设置

在`main.cpp`中，可以通过修改NUM_CLASSES(num)中num的数字来选择是但目标跟踪还是多目标跟踪。当num=1为单目标跟踪，当num>1为多目标跟踪。

```
const int NUM_CLASSES(1); // number of object classes
```

在`main.cpp`中以下代码可以设置跟踪的类别信息：

```
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
        }
```

通过修改`dr.classId`可以指定要跟踪的类别。



## References

- [DeepSORT](https://github.com/shaoshengsong/DeepSORT)

- [ByteTrack-MCMOT-TensorRT](https://github.com/CaptainEven/ByteTrack-MCMOT-TensorRT)
