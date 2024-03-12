
videos = load('./data/binocular_videos.mat');
videos = videos.BV;

l = videos.Lccd(:,:,:,:);
r = videos.Rccd(:,:,:,:);
v = 15
v = 49
nFrames = 15
maxVal = max(max(max(max(l(:,:,1:nFrames,v)))), max(max(max(r(:,:,1:nFrames,v)))))
for n = 1:nFrames
    % export n frame of the left video
    imwrite(l(:,:,n,v)/maxVal, sprintf('./data/video_frames/left_%d.png',n));
    imwrite(r(:,:,n,v)/maxVal, sprintf('./data/video_frames/right_%d.png',n));
    imwrite(real(sqrt(l(:,:,n,v)/maxVal)), sprintf('./data/video_frames/left_gamma_%d.png',n));
    imwrite(real(sqrt(r(:,:,n,v)/maxVal)), sprintf('./data/video_frames/right_gamma_%d.png',n));
end



