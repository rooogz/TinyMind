docker run -it \
-v /local_data2/TinyMind_myres26:/data \
-v /local_data2/TinyMind_myres26/my_data:/data/data \
-v /local_data2/TinyMind_myres26/output:/data/output \
uhub.ucloud.cn/uaishare/cpu_uaitrain_ubuntu-14.04_python-2.7.6_caffe-1.0.0:v1.0 \
/bin/bash -c "cd /data && /usr/bin/python /data/train.py --solver=/data/solver_myres26.prototxt --use_cpu=True --work_dir=/data --data_dir=/data/data --output_dir=/data/output --log_dir=/data/output/log" 
