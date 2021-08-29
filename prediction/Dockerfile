FROM tensorflow/tensorflow

LABEL maintainer='Roudra Saha'
RUN mkdir /data

RUN pip3 install pandas scikit-learn

COPY prediction.py /data


WORKDIR /data

CMD ["python", "prediction.py"]
