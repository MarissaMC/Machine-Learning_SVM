function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%
% CSCI 576 2014 Fall, Homework 3

data=[ones(size(train_data,1),1),train_data];

[n_d,n_f]=size(data);
%x=[w;l];
f=[zeros(n_f,1);repmat(C,n_d,1)];
%H=[ones(n_f,1);zeros(n_d,1)];
Y=zeros(n_d,n_f);

for i=1:n_d
    m=repmat(train_label(i),n_f,1);
    Y(i,:)=m'.*data(i,:);
end
a=diag(repmat(-1,n_d,1));
A=[-Y,a];
B=-ones(n_d,1);
h=[0;ones(n_f-1,1)];
H=[diag(h),zeros(n_f,n_d);zeros(n_d,n_f),zeros(n_d,n_d)];
k=repmat(-inf,n_f,1);
lb=[k;zeros(n_d,1)];
%opts = optimoptions('quadprog','Algorithm','active-set','Display','off');
[x,fval,exitflag] = quadprog(H,f,A,B,[],[],lb,[]);

b=x(1);
j=2:61;
w=x(j);
%l=x(62:end);

%obj=0.5*norm(w)^2+C*sum(l)