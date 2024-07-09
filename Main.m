clear;clc; addpath(genpath('.\'))

load birds.mat
dataset_name = 'birds';

%% Optimization Parameters
alphas   = 0.01;
betas    = 0.1;
gammas   = 0.01;
lambda1s = 0.1;
lambda2s = 0.1;

% alphas   = [0.001,0.01,0.1,1];
% betas    = [0.0001,0.001,0.1,1];
% gammas   = [0.0001,0.1,1];
% lambda1s = [1,10,100];
% lambda2s = [0.0001,0.001,0.1,1];
optmParameter.sigma   = 3;

optmParameter.maxIter           = 200; % 最大迭代次数
optmParameter.minimumLossMargin = 0.00001; % 两次迭代的最小损失间距  0.0001
minloss = 0.00001;
optmParameter.bQuiet            = 1;

%% 提取数据集
% numB=2; [train_data, train_target]=trans(train_data,train_target,numB);
index=find(train_target==-1);
train_target(index)=0;
train_target = train_target';

[~,num_feature] = size(train_data); [~,num_label] = size(train_target);

%% 计算C，G，R
Aeq=ones(1,num_feature); FF=zeros(num_feature,num_feature); FL=zeros(num_feature,num_label);
k_clos_neibor_num = 5;epsilon = 5;[n,~] = size(train_data');

t0 = clock;
for i=1:num_feature
    for j=1:num_label
		% The relevance betwwen feature and label
		FL(i,j)=mi(train_data(:,i),train_target(:,j));
    end 
end

for i=1:num_feature
    for j=1:num_feature
        % The redundancy betwwen features
        FF(i,j)=mi(train_data(:,i),train_data(:,j));
    end
end
for i=1:num_feature
    FF(i,i)=1;
end

E = pdist2(train_data',train_data');
for i =1:n
    temp = E(i,:);
    Ws =sort(temp);
    temp  = (temp <= Ws(k_clos_neibor_num));
    E(i,:) = temp;
end
for i = 1:n - 1
   for j = 2:n
       if E(i,j) == 1
          L2 = norm(train_data(:,i)-train_data(:,j));
          E(i,j) = exp(-(L2^2)/epsilon); 
          E(j,i) = E(i,j);
       end
   end
end
L = diag(sum(E)) - E;
C = FL; G = FF;

%% 参数
alpha_num = length(alphas);
beta_num = length(betas);
gamma_num = length(gammas);
lambda1_num = length(lambda1s);
lambda2_num = length(lambda2s);
Para_num =  beta_num * alpha_num * gamma_num *lambda1_num * lambda2_num;
Result_NEW  = zeros(6,50);
Avg_Means = zeros(6,Para_num);
Avg_Stds = zeros(6,Para_num);

a  = 0;
b  = 0;
g  = 0;
l1 = 0;
l2 = 0;
k = 1;

%% 网格调参
for alpha = alphas
    optmParameter.alpha = alpha;
    a = a + 1;
    for lambda1 = lambda1s
        optmParameter.lambda1 = lambda1;
        l1 = l1 + 1;
        for beta = betas
            optmParameter.beta = beta;
            b = b + 1;
            for gamma = gammas
                optmParameter.gamma = gamma;
                g = g +1;
                for lambda2 = lambda2s
                    optmParameter.lambda2 = lambda2;
                    l2 = l2 +1;
                    fprintf('MFS_MRC Running %s alpha - %d/%d beta - %d/%d gamma - %d/%d lambda1 - %d/%d lambda2 - %d/%d \n',dataset_name,a,alpha_num,b,beta_num,g,gamma_num,l1,lambda1_num,l2,lambda2_num);
                    %% Training
                    [Z,loss,iter]  = MFS_MRC(train_data,train_target,num_feature,num_label,optmParameter,C,G,L);
                    
                    [dumb, feature_idx] = sort(sum(Z,2),'descend');
                    time = etime(clock, t0);

                    %% Begin MLKNN
                    load('birds.mat')
                    Num=10;
                    Smooth=1;

                    for i = 1:50
                        fprintf('Running the program with the selected features - %d/%d \n',i,num_feature);
                        
                        f=feature_idx(1:i);
                        [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f),train_target,Num,Smooth);
                        [Outputs,Pre_Labels]=MLKNN_test(train_data(:,f),train_target,test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);
                        
                        Result_NEW(:,i) = EvaluationAll(Pre_Labels,Outputs,test_target);

                    end
                    
                    Avg_Means(1:6,k) = mean(Result_NEW,2);%平均值 2代表行
                    Avg_Stds(1:6,k) = std(Result_NEW,1,2);%标准差
                    Avg_Means(7,k) = alpha;
                    Avg_Means(8,k) = beta;
                    Avg_Means(9,k) = gamma;
                    Avg_Means(10,k) = lambda1; 
                    Avg_Means(11,k) = lambda2; 
                    Avg_Means(12,k) = iter-1; 
                    Avg_Means(13,k) = minloss; 
                    k = k + 1;
                end
                l2 = 0;
            end
            g = 0;
        end
        b = 0;
    end
    l1 = 0;
end

Avg_Means = Avg_Means';
Avg_Stds = Avg_Stds';
