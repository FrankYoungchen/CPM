# Convolutional Pose Machines（CPM） 
## test one picture 
python test_per_example.py 
## test time and pckh 
 python benchmark.py --frozen_pb_path=model.pb --anno_json_path=/home/cyk/dataset/ai_challenger/val.json \ 
                     --img_path=/home/cyk/dataset/ai_challenger/val --output_node_name=Convolutional_Pose_Machine/stage_5_out
## result
![Image text](https://github.com/FrankYoungchen/CPM/blob/master/src/test_result/picture_with_keypoint.jpg)
