function matrixIm = unvectorizeVideoS3D(vectorStim, timeDim, spaceDim)
  % The first half of the vector is left eye, the second half is right eye
  matrixIm = reshape(vectorStim, 1, spaceDim, timeDim*2);
  matrixIm = squeeze(matrixIm);
  matrixIm = matrixIm';
  matrixIm = [matrixIm(1:timeDim,:), matrixIm(timeDim+1:timeDim*2,:)];

