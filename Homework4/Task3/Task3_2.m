clc;
clear;
close all;

disp('Experiment: Network Scale Comparison');
% 加载数据集
[X, T] = valve_dataset;

% 定义不同隐藏层节点数
hidden_nodes_list = [5, 10, 20, 50];
scale_errors = [];

for hidden_nodes = hidden_nodes_list
    % 创建 NARX 网络
    net = narxnet(1:2, 1:2, hidden_nodes);
    
    % 准备数据
    [Xs, Xi, Ai, Ts] = preparets(net, X, {}, T);
    
    % 训练网络
    net = train(net, Xs, Ts, Xi, Ai);
    Y = net(Xs, Xi, Ai);
    
    % 计算误差
    error = mse(net, Ts, Y);
    scale_errors = [scale_errors; hidden_nodes, error];
    
    % 绘制结果
    figure;
    plotresponse(Ts, Y);
    title(['Hidden Nodes: ', num2str(hidden_nodes)]);
    xlabel('Time');
    ylabel('Response');
end

% 输出结果表格
disp('Network Scale Comparison Results:');
disp(array2table(scale_errors, 'VariableNames', {'HiddenNodes', 'MSE'}));
