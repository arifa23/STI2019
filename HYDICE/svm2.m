result1 = [];
result2 = [];
resultcv = [];
for i= 1:10
    load PC_DC_Train.txt;
    train=PC_DC_Train;
    clear PC_DC_Train;
    label_train = train(:,1);

gscatter(train(:,5),train(:,6),label_train);
 b=legend('Water','Street','Grass','Trees','Roof','Shadow');
set(b,'Location','SouthEast');
xlabel('Principal Copmponent 3', 'FontWeight','bold');
ylabel('Principal Copmponent 4','FontWeight','bold');    

    train(:,1:2)=[];
    train = train(:,1:i);
    mx_train = max(train(:));
    mn_train = min(train(:));
    
    %test data;
    load PC_DC_Test.txt;
    test = PC_DC_Test;
    clear PC_DC_Test;
    label_test = test(:,1);
    test(:,1:2)=[];
    test = test(:,1:i);
    mx_test = (max(test(:)));
    mn_test = min(test(:));
    
    train = (train-mn_train) ./ (mx_train-mn_train);
    test = (test-mn_test) ./ (mx_test-mn_test);

    addpath('I:\libsvm-3.22\matlab');
 
   bestc=5; bestg=6;%pca
 
    
%   bestcv=0; bestc=0; bestg=0;
%    for c = 0:1:5
%       for g = 0:0.5:8
%             cmd=['-t 2 -v 10 -c ',num2str(c), ' -g 4', num2str(g)];
%            cv = svmtrain(label_train, train, cmd);
%          if(cv>bestcv)
%                bestcv=cv; bestc=c; bestg=g;
%             end
%             fprintf('%g   %g  %g (best c=%g, g=%g, rate=%g)\n', c, g, cv, bestc, bestg, bestcv);
%        end
%     end
%      
%     resultcv = [resultcv bestcv];
    cmd=['-t 2 -c ',num2str(bestc), ' -g 4', num2str(bestg)];
    
    
   model = svmtrain(label_train,train,cmd);
  % fprintf('\n\nTraining Data Accuracy: \n\n ');
%    [predict_label, trainingaccuracy, dec_values]= svmpredict(label_train, train, model);
%    result1 = [result1 trainingaccuracy(1)];
   
    fprintf('\n\nTest Data Accuracy: \n\n ');
    [predict_label, testaccuracy, dec_values] = svmpredict(label_test, test, model);
    result2 = [result2 testaccuracy(1)];
end



