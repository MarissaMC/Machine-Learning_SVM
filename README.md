Support Vector Machine
=====================
The code written with Matlab.

Train linear SVM (primal form)

Input
----------
train_data: N*D matrix, each row as a sample and each column as a feature

train_label: N*1 vector, each row as a label

C: tradeoff parameter (on slack variable side)

Output
----------
w: feature vector (column vector)

b: bias term

Usage
-----------------
[w,b] = trainsvm(train, train_label, C);
accu = testsvm(test, test_label, w, b);
