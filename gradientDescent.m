function [params, param_history, history] = gradientDescent(costFunction, params, learning_rate, max_iter)

    [cost, gradient] = costFunction(params);
    oldCost = cost + 1; % dummy value
    iter = 1;
    history = cost;
    param_history = zeros(size(params'));
    
    % abs(oldCost - cost) > 1e-10 && 
    while iter < max_iter
        oldCost = cost;
        params = params - learning_rate * gradient;
        param_history = [param_history; params'];
        [cost, gradient] = costFunction(params);
        %if mod(iter, 10) == 0
            fprintf("\nIter %d, Cost: %d", iter, cost);
        %end
        iter = iter + 1;
        
        history = [history cost];
    end
end