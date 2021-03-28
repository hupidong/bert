FROM hupidong/tfserving_py3:latest

LABEL author=hupidong2006@163.com

WORKDIR /home/bert

COPY ./models /home/bert/models/
COPY ./server_tornado.py /home/bert/
COPY ./server_flask.py /home/bert/
COPY ./tokenization.py /home/bert/
COPY ./bert_tiny /home/bert/bert_tiny/
COPY ./start.sh /home/bert/

ENTRYPOINT ["sh", "/home/bert/start.sh"]
