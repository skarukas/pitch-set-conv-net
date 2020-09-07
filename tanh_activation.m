function [y, grad] = tanh_activation(z)
    y = tanh(z);
    grad = 1 - tanh(z) .^ 2;
end