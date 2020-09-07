function run(pitches)
    x = convertToPitchSpace(pitches);
    predictNN(x, 6, true)
end