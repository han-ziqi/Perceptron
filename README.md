## README - Perceptron

This is a Implementation of [Perceptron](https://en.wikipedia.org/wiki/Perceptron), is a project about [numpy](https://en.wikipedia.org/wiki/NumPy) and [pandas](https://en.wikipedia.org/wiki/Pandas_(software))

- Perception is inspired by neurons in biology, where when a single neuron is connected to several other neurons, these incoming neurons have different activation values and thus different results are obtained from a single neuron. The above process can be analysed from a mathematical point of view, where a neuron can be judged by the sum of vectors and weights multiplied together.

This standalone .py file including 

1. implement a binary Perceptron

2. read and train **train.data** and discriminate between: 

   - class 1 and class 2
   - class 2 and class 3
   - class 1 and class 3

3. Add L2-regularisation

---


## Running the code

Simply put the two datasets **train.data** and **test.data** in the same directory with this perceptron.py in supported software e.g. PyCharm. 

##### Here is a demo for running
![Perceptron demo](https://github.com/han-ziqi/Perceptron/raw/master/demo/Perceptron-2.jpeg)


## Binary Perceptron

If you want to run this part, please ensure **the data file is already in the current directory**, and then **please remove the comment symbol between the 131 and 145 lines**. You can see result in my dashboard in console

## Multi-Class Perceptron

 In inspiration with the 1-vs-rest method, I expanded the method by adding each of the three classes and comparing it with the full results. If you want to run the Multi-Class Perceptron, **just remove the multi-line comment on lines 217-231**. You can see result in my dashboard in console

## L2-regularisation 

 To prevent overfitting, l2 regularisation can be added to the multi-class perceptron by **first uncommenting lines 150 and 151** and **defining the values of coefficient 0.01, 0.1, 1.0, 10.0, 100.0 in line 151**. Subsequently This can be done by uncommenting line 195.

