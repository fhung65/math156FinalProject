load('Indian_pines_gt.mat')
load('Indian_pines_corrected.mat')



data = reshape(indian_pines_corrected, 145*145, 200);



%% Adding location feature
lambda = 200;
b = [1:145]';
temp = repmat(b, 1, 145);
a = reshape(temp', 145*145, 1);%x coord
b = repmat(b, 145, 1);% y coord
c = [a,b];
%%

data=  [data, c*lambda];

%% 
ipc = indian_pines_corrected;
sampleRange = 29*[1:4]; % 145/5 = 29
sampleY = repmat(sampleRange', 4, 1);
sampleX = reshape(repmat(sampleRange', 1, 4)', 16, 1);

lastPoint = [randi(145), randi(145)];
samplePoints = [sampleY, sampleX; lastPoint];
%data2 = reshape(data, 145, 145, 202);

initMeans = zeros(17,202);
for i = 1:17
    pt = samplePoints(i,:);
    rngY = [pt(1)-2:pt(1)+2];
    rngX = [pt(2)-2:pt(2)+2];
    samples = reshape(ipc(rngY,rngX,:), 25,200);
    avg = mean(samples,1);
    initMeans(i,:) = [avg, lambda*pt(2), lambda*pt(1)];
end

%initMeans = data2(sampleRange,sampleRange,:);

%initMeans = reshape(initMeans,16,202);
%lastMean = data2(randi(145),randi(145),:);
%initMeans = [initMeans; reshape(lastMean, 1, 202)];
%%
[C, labels] = km(data', 17, 1000, initMeans');
[C, labels] = km_noinit(data', 17, 1000);



