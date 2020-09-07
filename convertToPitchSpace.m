% takes a vector of any size and converts it into a 
%    binary vector of pitch classes
function pc_set = convertToPitchSpace(pitches)
    pc_set = false(1, 12);
    mask = mod(pitches, 12) + 1;
    pc_set(mask) = true;
end