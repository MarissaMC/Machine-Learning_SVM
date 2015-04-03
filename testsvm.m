function accu = testsvm(test_data, test_label, w, b)
% Test linear SVM
% Input:
%  test_data: M*D matrix, each row as a sample and each column as a
%  feature
%  test_label: M*1 vector, each row as a label
%  w: feature vector
%  b: bias term
%
% Output:
%  accu: test accuracy (between [0, 1])
%
% CSCI 576 2014 Fall, Homework 3

[n_d,n_f]=size(test_data);
Y=test_data*w+repmat(b,n_d,1);

for i=1:n_d
    if Y(i)>0
        Y_judge(i)=1;
    else
        Y_judge(i)=-1;
    end
end

accu=sum(Y_judge'==test_label)/n_d;

