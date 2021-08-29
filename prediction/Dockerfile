FROM tensorflow/tensorflow

LABEL maintainer='Roudra Saha'


RUN pip3 install pandas scikit-learn

COPY prediction.py .


WORKDIR .

CMD ["python", "prediction.py"]