function [ pre,rec,acc,f1,fpr,fnr ] = conf2pr( c )
%CONF2PR Summary of this function goes here
% c is a binary confusion matrix from confusionmat
% tp:c11
% fn:c12
% fp:c21
% tn:c22
    if c(1,1)==0
        pre = 0;
    else
        pre = c(1,1)/(c(1,1)+c(2,1));
    end
    rec = c(1,1)/(c(1,1)+c(1,2));
    acc = (c(1,1)+c(2,2))/(sum(sum(c)));
    f1=2*(pre*rec)/(pre+rec);
    fpr=c(2,1)/(c(2,1)+c(2,2));
    fnr=c(1,2)/(c(1,1)+c(1,2));
end

