function [X, y] = generateData(m)
    X = [];
    % random pitch sets
    for i = 1:m
        X = [X; round(rand(1, 12))];
    end
    
    %% operate upon data to create y
    %y = [X(:, 11:12) X(:, 1:10)]; % learn to transpose up by a half step
    y = flip([X(:, 11:12) X(:, 1:10)], 2); % learn to flip the pitch set
    
end