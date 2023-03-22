function [params] = stereoCalibrate(imgpath1,imgpath2,squareSize)
leftImages = imageDatastore(imgpath1,'FileExtensions',{'.jpg','.png'});
rightImages = imageDatastore(imgpath2,'FileExtensions',{'.jpg','.png'});
[imagePoints,boardSize] = detectCheckerboardPoints(leftImages.Files,rightImages.Files);
% 方格的边长，单位为mm
squareSize = squareSize;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);
I = readimage(leftImages,1);
imageSize = [size(I,1),size(I,2)];
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
% rotation matrix
rotationOfCamera = params.RotationOfCamera2;
% translation matrix
transOfCamera = params.TranslationOfCamera2;
% left camera intrinsic matrix
leftCameraMatrix = params.CameraParameters1.IntrinsicMatrix;
% left camera K1,K2,K3
lrtemp = params.CameraParameters1.RadialDistortion;
lrsize = size(lrtemp);
if lrsize(2) < 3
    lrtemp(end+1) = 0;
end
leftRadialDistortion = lrtemp;
% left camera P1,P2
leftTangDistortion = params.CameraParameters1.TangentialDistortion;
% right camera intrinsic matrix
rightCameraMatrix = params.CameraParameters2.IntrinsicMatrix;
% right camera K1,K2,K3
rrtemp = params.CameraParameters2.RadialDistortion;
rrsize = size(rrtemp);
if rrsize(2) < 3
    rrtemp(end+1) = 0;
end
rightRadialDistortion = rrtemp;
% right camera P1,P2
rightTangDistortion = params.CameraParameters2.TangentialDistortion;
params = {rotationOfCamera,transOfCamera,leftCameraMatrix,leftRadialDistortion,leftTangDistortion,...
          rightCameraMatrix,rightRadialDistortion,rightTangDistortion,imageSize};

