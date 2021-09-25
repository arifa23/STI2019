clc;
clear all;
close all;

% mutual information
load actrain.txt % loads input data
data=actrain(:,3:end);% input starting from row 3 upto end are considered sample data 

c1 =actrain(:,1);% first row input is cosidered as class label
[n m]=size(c1);
for Column = 1:m
    Alphabet = unique(c1(:,Column));%finds each class label
    Frequency = zeros(size(Alphabet));% for each class label
    for symbol = 1:length(Alphabet)
        Frequency(symbol) = sum(c1(:,Column) == Alphabet(symbol));%total no of each class label appearing
    end
    P = Frequency / sum(Frequency);% calculates probablity of each class label
    H_classlabel(Column) = -sum(P .* log2(P));% calculates marginal entropy of each class label
end
 figure,%plots histogram of class label
 subplot(1,2,1)
 bar( Alphabet,Frequency)
xlabel('Class Label');
ylabel('Frequency');
title('Class Label histogram');
[n1, m1] = size(data);%n= row , m= column of sample data
H_sample = zeros(1,m);


for Column = 1:m1
    Alphabet1 = unique(data(:,Column));%finds each class of sample
    Frequency1 = zeros(size(Alphabet1));
    for symbol = 1:length(Alphabet1)
        Frequency1(symbol) = sum(data(:,Column) == Alphabet1(symbol));%total no of each sample class  appearing
    end
    P1 = Frequency1 / sum(Frequency1);% calculates probablity of each sample
    H_sample(Column) = -sum(P1 .* log2(P1));% calculates marginal entropy of each sample
end
 subplot(1,2,2)%plots histogram of sample
 bar( Alphabet1,Frequency1)
 xlabel('Pixel value');
 ylabel('Frequency');
 title('Sample histogram');

m=zeros(1,m1);
for i=1:m1
    indrow = double(c1(:)) + 1;%no of class label
    indcol = double(data(:,i)) + 1;% data in i th column
    jointHistogram = accumarray([indrow indcol], 1);% creates combination of (xi, yi)
    jointProb = jointHistogram / numel(indrow);%calculates joint probabilty of (xi, yi)
    indNoZero = jointHistogram ~= 0;%initially probability 0
    jointProb1DNoZero = jointProb(indNoZero);%satarts from (0,0)
    jointEntropy = -sum(jointProb1DNoZero.*log2(jointProb1DNoZero));%calculates joint entropy h(x,y)
    m(i)=H_sample(i)+ H_classlabel-jointEntropy;%calculates mi = h(x)+h(y)-h(x,y)
end

pc=1:m1;
figure,%plots mi vs pc graph
plot(pc,m,'LineWidth',3)
grid on;
xlabel('Features');
ylabel('Mutual information');
title('Mutual information between class label and samples');

[spam, idx]=sort(m(:), 'descend');
mi_pc=idx(1:20)'%takes best 10 mi 
