function [params, param_history, history] = acceleratedDescent(costFunction, params, learning_rate, momentum, max_iter)

    [cost, gradient] = costFunction(params);
    oldCost = cost + 1; % dummy value
    iter = 1;
    history = cost;
    last_change = 0;
    param_history = zeros(size(params'));
    
    % abs(oldCost - cost) > 1e-8 && 
    while iter < max_iter
        oldCost = cost;
        updated_params = params - momentum * last_change;
        % use look-ahead gradient
        [cost, gradient] = costFunction(updated_params);
        param_history = [param_history; params'];
        
        change = momentum * last_change + learning_rate * gradient;
        params = params - change;
        last_change = change;
        
        %if mod(iter, 10) == 0
            fprintf("\nIter %d, Cost: %d", iter, cost);
        %end
        iter = iter + 1;
        
        history = [history cost];
    end
end