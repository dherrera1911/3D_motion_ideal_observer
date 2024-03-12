function filteredInput = filterS3D(S3D, indices2keep)
%
S3D.ctgIndSpd = S3D.ctgIndSpd(indices2keep);
S3D.ctgIndDir = S3D.ctgIndDir(indices2keep);
S3D.ctgIndMotion = S3D.ctgIndMotion(indices2keep);
S3D.ctgIndRetSpd = S3D.ctgIndRetSpd(indices2keep);
S3D.Iccd = S3D.Iccd(:,indices2keep);
S3D.Iret = S3D.Iret(:,indices2keep);

filteredInput = S3D;

