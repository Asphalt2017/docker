FROM tensorflow/tensorflow

LABEL maintainer='Roudra Saha'

RUN mkdir /data
RUN pip3 install pandas scikit-learn

COPY model_ann_one.py ./data

#COPY X_test.csv ./user_data
#COPY X_train.csv ./user_data
#COPY y_train.csv ./user_data
#COPY traveller_id.csv ./user_data

CMD python model_ann_one.py
