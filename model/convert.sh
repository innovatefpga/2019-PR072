python /home/robot/YOLO/tensorflow-yolo-v3/convert_weights_pb.py --class_names /home/robot/YOLO/bdd100k/bdd100k.names --data_format NHWC --weights_file /home/robot/YOLO/bdd100k/yolo3-bdd100k.backup --tiny --output_graph tiny_yolov3.pb


python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model tiny_yolov3.pb --tensorflow_use_custom_operations_config yolo_v3_tiny.json --input_shape [1,416,416,3]
