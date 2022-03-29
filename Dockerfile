FROM 10jqkaaicubes/cuda:10.0-py3.7.9

COPY ./ /home/jovyan/span_abstract

RUN cd /home/jovyan/span_abstract && python -m pip install -r requirements.txt