# RT-Lane-Detection
Support realtime deep neural network lane detection in Windows platform. It is on top of repository [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and paper  "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

The baseline repository implemented CULane & Tusimple datasets oriented training and inference. Wherein target frames are extracted from target video based on specific pattern. As a result, a group of images are prepared as input to model for training or inference. 

This repository adds another approach for interference. It is realtime getting image from a target video and then input to the model. The approach remove the step in which all target images are retrieved and placed in your disk. It streamlines the video capture and frame inference. New option --video-in is used to indicate target video. At least, the approach simplifies inference workload in Windows platform. New option --video-out is designed to export the labeled video or not. 

Of course, it can be easily extended to training and the other platforms on needed basis.

# Running
A new dataset category is added for the new approach. The command can be like:

````
python demo.py configs/custom.py
--dataset
Custom
--log_path
log_path
--test_model
weights/culane_18.pth
--test-cfg
CULane
--num_lanes
3
--video-in
datasets/Custom/see_one.mp4
--video-out
datasets/Custom
--frame-interval
10
````

In original implementation, a set of images lists are predefined. Here is a little enhancement to allow users to define and play own list. The command looks like:

````
python demo.py configs/culane.py
--test_model
weights/culane_18.pth
--dataset
CULane
--test-list
test6_curve.txt
--num_lanes
4
````

# Finding
When trying to detect a video I captured in my car with fixed phone camera. The detection result is quite unideal, no any lane detected successfully. You can see the result from the [video](see_one_label1.avi).

Through simple comparing with original dataset images, the most difference points to camera horizon. Accordingly, I added a new config item "crop" to remove "redundant" boundary. Then the interference result looks improved a little. Please watch the [video](see_one_label2.avi) for detail.

All above experimental is based on pre-trained model culane_18.pth. I concluded the model is overfitting on its dataset in term of object distribution in camera horizon. It is not a defect but just a character, in my view.

**Any other comment?**