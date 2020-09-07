model.n = 12; % number of notes per octave (need to adjust in other places)
model.m_train = 1000;
model.m_test = 2000;
model.num_filters = 6;
model.activation = @sigmoid;
model.lambda = 0;

costFn = @NNCost;
model.predict = @(pitches, show) predictNN(model, pitches, show);

filters = rand(model.num_filters, model.n);
b1 = rand(model.num_filters, model.n);
w = rand(model.num_filters, 1);
b2 = rand(model.n, 1);

% costFn = @NNCostFC;
% predictFn = @predictNNFC;
% params_size = 2 * model.num_filters * n + model.num_filters + n + n*n;

learning_rate = 1;
momentum = 0.9;
max_iter = 10000;

optFn = @(cst_fn, p) gradientDescent(cst_fn, p, learning_rate, max_iter);
%optFn = @(cst_fn, p) momentumGradientDescent(cst_fn, p, learning_rate, momentum, max_iter);
%optFn = @(cst_fn, p) acceleratedDescent(cst_fn, p, learning_rate, momentum, max_iter);
%optFn = @(cst_fn, p) adaGrad(cst_fn, p, learning_rate, max_iter);

%% generate data and random initialization of parameters
params = [ filters(:); b1(:); w(:); b2(:) ];
[model.X_train, model.Y_train] = generateData(model.m_train);

%% test if numerical gradients match actual gradients

% checkGradients(@(p) costFn(model, p), params);
% 
% pause;

%% optimize parameters

[params, param_history, grad_history] = optFn(@(p) costFn(model, p), params);

save 'nnparams' params

%% retrieve training accuracy
correct_count = 0;
fprintf('\nCalculating accuracy');
for i=1:model.m_train
    out = model.predict(model.X_train(i, :), false);
    correct_count = correct_count + isequal(out, model.Y_train(i, :));
    
    % make 10 dots in total for any size
    if mod(i, floor(model.m_train / 10)) == 1
        fprintf('.');
    end
end
accuracy = correct_count / model.m_train;
fprintf('\nTraining accuracy: %.2f%', accuracy * 100);


%% retrieve testing accuracy
[X_test, y_test] = generateData(model.m_test);

correct_count = 0;
fprintf('\nCalculating accuracy');
for i=1:model.m_test
    out = model.predict(X_test(i, :), false);
    correct_count = correct_count + isequal(out, y_test(i, :));
    
    % make 10 dots in total for any size
    if mod(i, floor(model.m_test / 10)) == 1 
        fprintf('.');
    end
end
accuracy = correct_count / model.m_test;
fprintf('\nTesting accuracy: %.2f %', accuracy * 100);

model.predict(convertToPitchSpace(0), true);

displayConvergence(param_history, grad_history);