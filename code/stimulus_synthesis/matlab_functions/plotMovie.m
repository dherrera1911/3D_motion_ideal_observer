function plotMovie(movie)
figure();
colormap(gray);
for i = 1:size(movie, length(size(movie)))
    %imshow(sqrt(movie(:,:,i)), [0, 256], 'InitialMagnification', 700);
    imshow(movie(:,:,i), [0, 256^2], 'InitialMagnification', 700);
    pause(0.1);
end

