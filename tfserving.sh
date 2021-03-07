docker run -ti \
    --name cola \
    -p 8809:8809 \
    --mount type=bind,source=/Users/hupidong/Work/Learning/bert/bert/models,target=/home/models/bert tensorflow/serving \
    --entrypoint=tensorflow_model_server \
    --rest_api_port=8809 \
    --model_name=cola \
    --model_base_path=/home/models/bert/cola
