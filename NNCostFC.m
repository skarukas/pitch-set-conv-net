function [cost, gradient] = NNCostFC(X, Y, params, lambda, num_filters)
    %% implementing batch here
    batch_size = 10;
    subset = randi(size(X, 1), batch_size, 1);
    
    X = X(subset, :);
    Y = Y(subset, :);
    
    
    g = model.activation; % activation function
    [m, n] = size(X); % n = 12
    cost = 0;
    
    b1_idx = n*num_filters+1;
    w_idx = b1_idx+n*num_filters;
    w2_idx = w_idx+num_filters;
    b2_idx = w2_idx+n*n;
    
    filters = params(1:n*num_filters);
    filters = reshape(filters, num_filters, n); % n_f x n
    b_1 = params(b1_idx:b1_idx+n*num_filters-1); 
    b_1 = reshape(b_1, num_filters, n); % n_f x n
    w = params(w_idx:w_idx+num_filters-1); % n_f x 1
    w2 = params(w2_idx:w2_idx+n*n-1);
    w2 = reshape(w2, n, n);  % n x n
    b_2 = params(b2_idx:b2_idx+n-1); % n x 1
    
    d_f = zeros(size(filters));
    d_b1 = zeros(size(b_1));
    d_w = zeros(size(w));
    d_w2 = zeros(size(w));
    d_b2 = zeros(size(b_2));
    
    for i = 1:m
        x = X(i, :)'; % n x 1
        y = Y(i, :)';
        a_2 = zeros(num_filters, n); % n_f x n
        a_2_d = zeros(num_filters, n);
        
        % conv layer
        for j = 1:num_filters
            z_2 = circularConvolution(x, filters(j, :)') + b_1(j, :)'; % n x 1
            [a_2(j, :), a_2_d(j, :)] = g(z_2); % n x 1
        end
        
        % hidden layer
        z_3 = w' * a_2 + b_2'; % 1 x n
        [a_3, a_3_d] = g(z_3); % 1 x n
        
        % hidden layer 2
        z_4 = (w2 * a_3')'; % 1 x n
        [a_4, a_4_d] = g(z_4); % 1 x n
        
        % reg gradient is wrong
        % L2 loss
        cost = cost + sum((a_4' - y) .^ 2, 1);

        % compute gradient
        d_a4 = a_4 - y'; % 1 x n
        
        delta_4 = d_a4 .* a_4_d; % 1 x n
        delta_3 = (delta_4 * w2) .* a_3_d; % 1 x n
        delta_2 = (w * delta_3) .* a_2_d; % n_f x n
        
        for j = 1:num_filters
            d_f(j, :) = d_f(j, :) + circularConvolution(x, delta_2(j, :)')';
        end
        
        % accumulate gradient
        d_w2 = d_w2 + (a_3' * delta_4)';
        d_w = d_w + a_2 * delta_3';
        d_b2 = d_b2 + delta_3';
        d_b1 = d_b1 + delta_2;

    end
    
    % add regularization and scaling
    cost = (cost + lambda*sum(params .^ 2)) / (2 * m);
    
    d_f = (d_f + lambda*sum(filters, 'all')) / m;
    d_b1 = (d_b1 + lambda*sum(b_1, 'all')) / m;
    d_w2 = (d_w2 + lambda*sum(w2, 'all')) / m;
    d_w = (d_w + lambda*sum(w, 'all')) / m;
    d_b2 = (d_b2 + lambda*sum(b_2, 'all')) / m;
    
    gradient = [d_f(:); d_b1(:); d_w(:); d_w2(:); d_b2(:)];
end