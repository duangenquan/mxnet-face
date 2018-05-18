FROM duangenquan/base:1.0

COPY . /opt/app/
RUN cd /opt/app/mxnet && \
    make clean && \ 
    make -j 4 USE_OPENCV=1 USE_BLAS=openblas &&\
    cd python && \
    pip2 install --upgrade pip && \
    pip install -e .


