function params = adaGrad(costFunction, params, learning_rate, max_iter)

    [cost, gradient] = costFunction(params);
    oldCost = cost + 1; % dummy value
    iter = 1;
    history = cost;
    param_accum = zeros(size(params));
    epsilon = 1e-8;
    % abs(oldCost - cost) > 1e-10 && 
    while iter < max_iter
        oldCost = cost;
        [cost, gradient] = costFunction(params);
        param_accum = param_accum + gradient .^ 2;
        params = params - (learning_rate .* gradient) ./ sqrt(param_accum + epsilon);
        %if mod(iter, 10) == 0
            fprintf("\nIter %d, Cost: %d", iter, cost);
        %end
        iter = iter + 1;
        
        history = [history cost];
    end
    figure
    plot(1:iter, movmean(history, 10));
    figure
end