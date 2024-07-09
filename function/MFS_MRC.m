function [Z,loss,iter]  = MFS_MRC(train_data,~,num_feature,num_label,optmParameter,C,G,L)
%% optimization parameters
alpha            = optmParameter.alpha;    % 1范数
beta             = optmParameter.beta;     % 2，1范数
gamma            = optmParameter.gamma;    % 独立性
lambda1          = optmParameter.lambda1;  % 实例相关性
lambda2          = optmParameter.lambda2;  % 局部结构
sigma            = optmParameter.sigma;
maxIter          = optmParameter.maxIter;
miniLossMargin   = optmParameter.minimumLossMargin;

%% LP initializtion
X = train_data;

%% optimization initialization
num_dim = size(X,2);
% I2 = eye(num_dim);
XTX = X'*X;
% 1范数约束的W
W   = (XTX + sigma*eye(num_dim)) \ C;
Zeta_Wk = W;
% 2，1范数约束的M
M   = (XTX + sigma*eye(num_dim)) \ C;
Zeta_Mk = M;
%% Iterative  
iter = 1; 
oldloss = 0;
tk1 = 1;

%% 计算LIP
A = gradL21(M);
varepsilon = 0.01;
I1 = eye(num_feature,num_label);
Lf = sqrt(4*norm(I1)^2 + 2*norm(beta*A)^2);
s1 = varepsilon*sqrt(2*alpha);
s2 = num_dim*sqrt(alpha/2)+sqrt((num_dim^2*alpha/2)+Lf*varepsilon);
mu=s1/s2;
    
Lip=Lf+(alpha*num_dim)/mu;

%% s-proximal gradient(S-APG)
while iter <= maxIter
    A = gradL21(M);
    n=size(W,1);
    I=eye(n,n);

    % calculate the graid of F_mu 
    grad_M_F=(Zeta_Wk+Zeta_Mk) - C + beta*A*Zeta_Mk;
       
    grad_W_F_1=(Zeta_Wk+Zeta_Mk) - C;
    PS=softthres(alpha*Zeta_Wk, mu);
    grad_W_F_2=(alpha^2/mu)*Zeta_Wk - (alpha/mu)*PS;
    grad_W_F=grad_W_F_1+grad_W_F_2; 

    % calculate W(k),M(k)
    r1=(1/Lip);
    Wk=Zeta_Wk-r1*grad_W_F;
    Mk=Zeta_Mk-r1*grad_M_F;

    % calculate W^k+1,M^k+1
    q1=gamma/Lip;
    q2=lambda1/Lip;
    q3=lambda2/Lip;
    W_1 = (q1*(M*M')+q2*G+q3*L+I)\(Wk-q2*G*M-q3*L*M);
    M_1 = (q1*(W*W')+q2*G+q3*L+I)\(Mk-q2*G*W-q3*L*W);
    
    % 更新 tk，中间 Zeta_Wk, Zeta_Mk
    tk_1=tk1;
    tk1=(1 + sqrt(4*tk_1^2 + 1))/2;

    Zeta_Wk  = W_1 + (tk_1 - 1)/tk1 * (W_1 - W);
    Zeta_Mk  = M_1 + (tk_1 - 1)/tk1 * (M_1 - M);
    
    W=W_1;
    M=M_1;

    %% 开始计算损失函数的值
    O1 = ((M+W) - C);
    DiscriminantLoss = (1/2)*trace(O1'* O1);
    WM_correlationloss = (gamma/2)*trace((W'*M)' * (W'*M));
    sparsity1    = alpha*norm(W,1);
    sparsity2    = beta*trace(M'*A*M);
    sample_correlationloss = (lambda1/2)*trace(((W+M))'*G*(W+M));
    Local = (lambda2/2)*trace(((W+M))'*L*(W+M))';

    totalloss = DiscriminantLoss + WM_correlationloss + sparsity1 + sparsity2 + sample_correlationloss + Local;
       
    loss(iter,1) = totalloss;
    if abs(oldloss - totalloss) <= miniLossMargin
        %本次迭代的结果与上次的结果相差少于预订的最小损失间距时，结束循环
        break;
    elseif totalloss <=0
        break;
    else
        oldloss = totalloss;
    end
    
    iter=iter+1;
end
Z = W+M;
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function A = gradL21(P)
num = size(P,1);
A = zeros(num,num);
for i=1:num
    temp = norm(P(i,:),2);
    if temp~=0
        A(i,i) = 1/temp;
    else
        A(i,i) = 0;
    end
end
end
