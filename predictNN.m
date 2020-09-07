function output = predictNN(model, pitches, show)
    num_filters = model.num_filters;
    x = pitches';
    g = model.activation;
    n = 12;

    
    %% load params
    load 'nnparams' params;

    b1_idx = n*num_filters+1;
    w_idx = b1_idx+n*num_filters;
    b2_idx = w_idx+num_filters;
    
    filters = params(1:n*num_filters);
    filters = reshape(filters, num_filters, n); % n_f x n
    b_1 = params(b1_idx:b1_idx+n*num_filters-1); 
    b_1 = reshape(b_1, num_filters, n); % n_f x n
    w = params(w_idx:w_idx+num_filters-1); % n_f x 1
    b_2 = params(b2_idx:b2_idx+n-1); % n x 1

    
    %% run NN
    a_2 = zeros(num_filters, n); % n_f x n
        
    % conv layer
    for j = 1:num_filters
        z_2 = circularConvolution(x, filters(j, :)') + b_1(j, :)'; % n x 1
        a_2(j, :) = g(z_2); % n x 1
    end

    % hidden layer
    z_3 = w' * a_2 + b_2'; % 1 x n
    a_3 = g(z_3); % 1 x n
    
    y = a_3;
    output = round(y);
    
    if show
        %% display layers
        num_plots = 8;

        % using 1 - to invert colors
        subplot(num_plots, 1, 1);
        imshow(1-x', []);
        title('input pitch set');

        subplot(num_plots, 1, 2);
        imshow(1-filters, []);
        title('filter(s)');
        
        subplot(num_plots, 1, 3);
        imshow(1-b_1, []);
        title('bias(es)');

        subplot(num_plots, 1, 4);
        imshow(1-a_2, []);
        title('filter activations(s)');

        subplot(num_plots, 1, 5);
        imshow(1-w', []);
        title('hidden weights');
        
        subplot(num_plots, 1, 6);
        imshow(1-b_2', []);
        title('hidden bias units');

        subplot(num_plots, 1, 7);
        imshow(1-y);
        title('result');

        subplot(num_plots, 1, 8);
        imshow(1-output, []);
        title('output pitch set');
    end
end