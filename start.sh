
# 启动tfserving
tensorflow_model_server --rest_api_port=8809 \
    --model_name=cola \
    --model_base_path=/home/bert/models/cola &\

#启动tornado-app服务
python3 server_tornado.py