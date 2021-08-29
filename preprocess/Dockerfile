FROM tensorflow/tensorflow

LABEL maintainer='Roudra Saha'
RUN mkdir ./data

RUN pip3 install pandas scikit-learn

COPY preprocess.py ./data

COPY test.csv ./data
COPY train.csv ./data


WORKDIR ./data

CMD python preprocess.py