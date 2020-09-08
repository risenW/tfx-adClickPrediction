FROM tensorflow/tfx:0.23.0
WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"