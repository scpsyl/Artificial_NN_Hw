clc;
clear;
close all;

disp('Experiment: Network Structure Comparison');
% 加载数据集
[X, T] = valve_dataset;

% 定义不同延迟设置
delays_list = {[1:1], [1:3], [1:5]};
structure_errors = [];

for delays = delays_list
    % 创建 NARX 网络
    net = narxnet(delays{:}, delays{:}, 10);
    
    % 准备数据
    [Xs, Xi, Ai, Ts] = preparets(net, X, {}, T);
    
    % 训练网络
    net = train(net, Xs, Ts, Xi, Ai);
    Y = net(Xs, Xi, Ai);
    
    % 计算误差
    error = mse(net, Ts, Y);
    structure_errors = [structure_errors; length(delays{:}), error];
    
    % 绘制结果
    figure;
    plotresponse(Ts, Y);
    title(['Delays: 1:', num2str(length(delays{:}))]);
    xlabel('Time');
    ylabel('Response');
end

% 输出结果表格
disp('Network Structure Comparison Results:');
disp(array2table(structure_errors, 'VariableNames', {'Delays', 'MSE'}));
