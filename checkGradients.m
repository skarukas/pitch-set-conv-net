function checkGradients(costFunction, params)
    [~, gradient] = costFunction(params);
    
    diff = 1e-5;
    fprintf("\nChecking gradients");
    for i = 1:size(params, 1)
        hi = params;
        hi(i) = hi(i) + diff;
        
        lo = params;
        lo(i) = lo(i) - diff;
        numericalGradient = (costFunction(hi) - costFunction(lo)) ...
                                   / (2 * diff);
        
        %fprintf("\n%d    %d", gradient(i), numericalGradient);
        err = abs(numericalGradient - gradient(i));
        
        if mod(i, floor(size(params, 1) / 30)) == 0 
            % make 30 dots in total for any size of params
            fprintf('.'); 
        end
        
        if err > diff
            fprintf("\nToo much error! for %d %d", i, err);
            fprintf("\n%d    %d", gradient(i), numericalGradient);
            return;
        end
    end
    fprintf("\nGradients seem fine");
end