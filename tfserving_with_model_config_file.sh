docker run -ti \
    --name cola_1 \
    -p 8810:8809 \
    --mount type=bind,source=/Users/hupidong/Work/Learning/bert/bert/models,target=/home/models/bert tensorflow/serving \
    --entrypoint=tensorflow_model_server \
    --rest_api_port=8809 \
    --model_config_file=/home/models/bert/model.config \
    --allow_version_labels_for_unavailable_models=true
