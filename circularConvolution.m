function y = circularConvolution(a, b)
    n = size(a, 1);
    indices = 0:n-1;
    
    y = zeros(n, 1);
    
    for k = 1:n
        ind = mod(indices - k + n + 1, n) + 1;
        shifted = b(ind);
        y(k) = a' * shifted;
    end
end