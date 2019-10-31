export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
aocl program acl0 /opt/intel/openvino/bitstreams/a10_devkit_bitstreams/2019R1_A10DK_FP16_TinyYolo.aocx
cd ~/inference_engine_samples_build/intel64/Release/
./myprojectback -i cam0 -m ~/YOLO/bdd100k/tiny_yolov3.xml -d HETERO:FPGA,CPU
