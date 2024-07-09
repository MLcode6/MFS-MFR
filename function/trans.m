function [train target]=trans(train_data,train_target,numB)
[m,n]=size(train_data);
target=train_target';
train=[];
for i=1:n
    res=dis_efi( train_data(:,i), numB );
    train=[train res];
end
index=find(target==-1);
target(index)=0;

%%%
% 这两段代码合起来完成了对数据的一系列处理。

% 首先，第一个代码段中的`trans`函数接受输入的`train_data`和`train_target`，并根据`numB`的值对数据进行转换和处理。
% 它通过调用`dis_efi`函数，将`train_data`中的每一列向量转换为一个新的特征向量，并将这些特征向量存储在`train`矩阵中。同时，它将`train_target`转置为行向量，并将其存储在`target`变量中。最后，它将`target`中值为-1的元素改为0。

% 接下来，第二个代码段中的`dis_efi`函数被`trans`函数调用。它接受输入的向量`vec`和`numB`，并返回一个处理后的结果向量`res`。

% 在`dis_efi`函数中，它首先计算向量`vec`的长度，并根据`numB`的值将其分割成`numB`个等间距的区间。然后，它对`vec`进行排序，并将排序后的结果存储在`res`中。

% 接下来，它通过循环，根据分割点的位置，将对应位置的值作为阈值，并将这些阈值存储在`val`中。然后，使用`unique`函数将`val`中的重复值去除，得到一个不重复的阈值向量。

% 最后，它创建一个全1向量`res`，并通过循环，将`vec`中大于每个阈值的元素所对应的位置在`res`中标记为`k+1`。

% 综合起来，这两段代码的目的是将输入的数据进行转换和处理，生成新的特征向量，并将目标值进行调整，以便后续的数据分析或机器学习任务。
%%%