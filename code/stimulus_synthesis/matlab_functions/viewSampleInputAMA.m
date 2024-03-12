function matrixIm = viewSampleInputAMA(S3D, ind)
%
vectorIm = S3D.Iccd(:,ind);
timeDim = S3D.Isz(3);
spaceDim = S3D.Isz(2);
matrixIm = unvectorizeVideoS3D(vectorIm, timeDim, spaceDim);
imshow(mat2gray(matrixIm), 'InitialMagnification', 700);

