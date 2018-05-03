docker run -it \
-v /local_data2/TinyMind:/data \
-v /local_data2/TinyMind/mydata:/data/data \
-v /local_data2/TinyMind/output:/data/output \
uhub.ucloud.cn/uaishare/cpu_uaitrain_ubuntu-14.04_python-2.7.6_caffe-1.0.0:v1.0 \
/bin/bash -c "cd /data && /usr/bin/python /data/predict_result_GPU.py"
