# Priority grasp
The code is the application of GraspNet Original program:
https://graspnet.net/

demo.py: GraspNet trained model with own point cloud data(.ply or RGBD images) to get grasp results.

priority_grasp.py: Grasp objects according to priority and display results.

evaluate_priority_grasp.py: Evaluation code of priority grasp use point metrics.

Command using code.py:
1 CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/checkpoint-rs.tar
2 bash ./command_demo.sh
