function wImage = contrastWindowing(im, W)
    [imWeb, imDC] = contrastImage(im, W);
%    imWeb = bsxfun(@times, imWeb, reshape(W, size(imWeb)));
    imWeb = imWeb .* W;
    wImage = imWeb*imDC + imDC;
