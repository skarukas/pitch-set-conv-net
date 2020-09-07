function [g, derivative] = sigmoid(z)
    g = sig(z);
    derivative = sig(z) .* (1 - sig(z));
    
    function y = sig(z)
        y = 1 ./ (1 + exp(-z));
    end
end