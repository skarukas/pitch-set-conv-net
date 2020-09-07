function [result, derivative] = relu(z)
    result = max(0, z);
    derivative = double(z >= 0);
end