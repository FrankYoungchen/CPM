import cv2
import tensorflow as tf
import cv2
import numpy as np
import os
from dataset_prepare import CocoPose
import random
import time
class project():


    def run_with_frozen_pb(self,img_path, input_w_h, frozen_graph, output_node_names,output_img_path):
        tf.reset_default_graph()
        from dataset_prepare import CocoPose
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )
        graph = tf.get_default_graph()
        image = graph.get_tensor_by_name("image:0")
        output = graph.get_tensor_by_name("%s:0" % output_node_names)
        image_0 = cv2.imread(img_path)
        w, h, _ = image_0.shape
        w_ratio,h_ratio = w/input_w_h,h/input_w_h
        image_ = cv2.resize(image_0, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)
        heatmap_x_y = []
        # time1 = time.time()
        with tf.Session() as sess:
            heatmaps = sess.run(output, feed_dict={image: [image_]})
            for _ in range(heatmaps.shape[3]):
                x, y = np.where(heatmaps[0, :, :, _] == np.max(heatmaps[0, :, :, _]))
                heatmap_x_y.append((int(2*y*h_ratio),int(2*x*w_ratio)))
                color_circle= (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                cv2.circle(image_0,(2*y*h_ratio,2*x*w_ratio),3,color_circle,-1)
        joints = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13)]
        for per_joint in joints:
            color_per_joint = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            cv2.line(image_0,heatmap_x_y[per_joint[0]],heatmap_x_y[per_joint[1]],color_per_joint,2)
        cv2.imwrite(output_img_path,image_0)

if __name__ == '__main__':
    output_img_path = './test/picture_with_keypoint.jpg'
    P = project()
    P.run_with_frozen_pb(
        "785.jpg",
        192,
        "model.pb",
        "Convolutional_Pose_Machine/stage_5_out",
        output_img_path
    )

