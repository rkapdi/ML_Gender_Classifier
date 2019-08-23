'''
Python based Machine Learning program using the SciKit.learn library.
Program uses a Decision Tree, K Neighbors, Support Vector Machine and Perceptron to classify the gender of someone, given their body matrics.

How Does a decision tree algorithm work?

A DecisionTree classifies data by creating branches for every possible outcome.
It is like a flow chart that stores data, for each data point it asks a Y/N question and
depending on the answer, moves in a corresponding direction.
The more data points it recievers, the further it will build the nodes.
For any unlabled node, the tree will ask a series of questions until it lables it. The given label will be our classification.

How does a K Neighbors algorithm work?

Find k nearest points about a new point, classify the new point based on the most frequent label out of these k points.

How does a Support Vector Machine algorithm work?

SVMs are supervised learning models used for classification. Given a set of training data with each data point belonging to one group or another, SVM builds a model that assigns new data points to one group or other and consequently SVM models represent data points in space, mapped so that data points are divided by a clear gap as wide as possible. The new data points are then mapped to the same space and their group classification is predicted based on which group they fall into.

How does a Perceptron algorithm work?

A perceptron is a simple model of a biological neuron in an artificial neural network.
Perceptron is a supervised learning algorithm of a binary classifier type. It uses a function to decide if an input belongs to some specific class.
The perceptron algorithm classifies patterns and groups by finding the linear separation between different objects and patterns that are received through numeric or visual input.

The more data we train it on, the more accurate it's classification will be.

Progam data set size is: 11
'''


from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron


#[height, weight, shoe size]
#X: a variable that is a list of lists containing height, weight and shoe size of a person
X = [[181,50,44], [178, 59, 80], [150, 32 ,65 ], [154,54,40], [190, 86,50], [185,88,61]
    , [150,100, 90], [125,92, 35], [150, 48, 20], [177, 95, 30], [172, 95, 55]]

#Y: list of lables; each label is a gender that is associated with body metrics that is associated with the previous list.
Y = ['male', 'female','female','female','female','male','male','male','female','male','male']


#clf1 is a variable to store our Decision Tree Model(DecisionTreeClassifier)
clf1 =  tree.DecisionTreeClassifier()
clf1 = clf1.fit(X,Y)

#clf2 is a variable to store our KNeighborsClassifier Model(KNeighborsClassifier)
clf2 = KNeighborsClassifier()
clf2 = clf2.fit(X,Y)

#clf3 is a variable to store our Perceptron Model(Perceptron)
clf3 = Perceptron()
clf3 = clf3.fit(X,Y)

#clf4 is a variable to store our Support Vector Machine Model(SupportVectorMachine)
clf4 = SVC()
clf4 = clf4.fit(X,Y)


# we test the model by classifying the gender of someone, given a new list of body metrics.
prediction1_DecisionTree = clf1.predict([[190,70,43]])
prediction2_KNeighbors = clf2.predict([[190,70,43]])
prediction3_Perceptron = clf3.predict([[190,70,43]])
prediction4_SVC = clf4.predict([[190,70,43]])


result = F"DecisionTreeClassifier: {prediction1_DecisionTree}\nKNeighborsClassifier: {prediction2_KNeighbors}\nPerceptron: {prediction3_Perceptron}\nSupport Vector Machine: {prediction4_SVC}"

print (result)
