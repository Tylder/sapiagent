FROM lofgrendaniel87/tensorflow-gpu-jupyter-dev

RUN apt-get update \
  && apt-get install -y \
    graphviz \
    libgraphviz-dev

RUN python --version

RUN python -m pip install pandas
RUN python -m pip install numpy
RUN python -m pip install scipy
RUN python -m pip install scikit-learn
RUN python -m pip install seaborn
RUN python -m pip install matplotlib

RUN python -m pip install pyclick pytweening

WORKDIR /code

CMD ["/usr/sbin/sshd", "-D"]