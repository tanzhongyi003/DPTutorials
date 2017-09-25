import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import paddle.v2 as paddle

TRAINING_DATA = None

def load_data():
    global TRAINING_DATA, TEST_DATA_X, TEST_DATA_Y, TEST_DATA
    X, Y = load_planar_dataset()
    TRAINING_DATA = np.hstack([X.T,Y.T])


def train(): 
    global TRAING_DATA
    load_data()
    def reader():
        for d in TRAINING_DATA:
            yield d[:-1],d[-1:]
    return reader

def test():
    global TEST_DATA
    load_data()
    def reader():
        for d in TRAINING_DATA:
            yield d[:-1],d[-1:]
    return reader


def main():


    # init
    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
    hidden = paddle.layer.fc(input =x, size= 4, act = paddle.activation.Tanh())
    y_predict = paddle.layer.fc(input =hidden, size= 1, act = paddle.activation.Sigmoid())
    y_label = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))

    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict, label=y_label)
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.01)

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

     # mapping data
    feeding = {'x': 0, 'y': 1}
    

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            print "Pass %d, Batch %d, Cost %f" % (
                event.pass_id, event.batch_id, event.cost)


    # training
    reader = paddle.reader.shuffle(train(), buf_size=50000)
    batch_reader = paddle.batch(reader,batch_size=512)
    trainer.train(
        batch_reader,
        feeding=feeding,
        event_handler=event_handler,
        num_passes=2000)

    # infer test
    test_data_creator = train()
    test_data = []
    test_label = []

    for item in test_data_creator():
        test_data.append((item[0],))
        test_label.append(item[1])

    probs = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=test_data)

    right_number =0
    total_number = len(test_data)
    for i in xrange(len(probs)):
        if float(probs[i][0]) >= 0.5 and test_label[i] ==1 :
            right_number += 1
        elif float(probs[i][0]) < 0.5 and test_label[i] ==0:
            right_number += 1

    print("right_number is {0} in {1} samples".format(right_number,total_number))
    print("training accuracy is {0} ".format(float(right_number)/total_number))


if __name__ == '__main__':
    main()
