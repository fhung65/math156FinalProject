load('Indian_pines_gt.mat')
load('Indian_pines_corrected.mat')
data = reshape(indian_pines_corrected, 145*145, 200);

%% Adding location feature
lambda = 30;
b = [1:145]';
temp = repmat(b, 1, 145);
a = reshape(temp', 145*145, 1);%x coord
b = repmat(b, 145, 1);% y coord
c = [a,b];
%%

data=  [data, c*lambda];

%% 
ipc = indian_pines_corrected;
sampleRange = 13*[1:10]; % 145/5 = 29
sampleY = repmat(sampleRange', 10, 1);
sampleX = reshape(repmat(sampleRange', 1, 10)', 100, 1);

samplePoints = [sampleY, sampleX];


initMeans = zeros(100,202);
for i = 1:100
    pt = samplePoints(i,:);
    rngY = [pt(1)-2:pt(1)+2];
    rngX = [pt(2)-2:pt(2)+2];
    samples = reshape(ipc(rngY,rngX,:), 25,200);
    avg = mean(samples,1);
    initMeans(i,:) = [avg, lambda*pt(2), lambda*pt(1)];
end

%% Using a lot of cluster centers first, then cluster the cluster centers
[C, labels] = km(data', 100, 1000, initMeans');
idx = 1:100;
sidx = randsample(idx, 21);
[C2, labels2] = km(C, 21, 100, C(:,sidx));


labels3 = labels2(labels);


%% 
data2 = data(:,1:200);

[height, width] = size(data2');
data2_demean = data2'-repmat(mean(data2',2), 1, width);
[U,S,V] = svd(data2_demean);

eig_val = diag(S);


princ_data = data2*U(:,1:115);
princ_initMeans = initMeans(:,1:200)*U(:,1:115);

[C, labels] = km(princ_data', 100, 1000, princ_initMeans');

idx = 1:100;
sidx = randsample(idx, 21);
[C2, labels2] = km(C, 21, 100, C(:,sidx));
labels3 = labels2(labels);



%%
temp_data = reshape(data2, 145, 145, 200);
conv_data = zeros(145, 145, 200);
a = zeros(145, 145);
a(1:2,1:2) = 1/4;
for i=1:200
    conv_data(:,:,i) = ifft2(fft2(temp_data(:,:,1)).*fft2(a));
end
conv_data = [reshape(conv_data, 145*145, 200), data2(:,end-1:end)];



[C, labels] = km(conv_data', 20, 1000, initMeans');

idx = 1:100;
sidx = randsample(idx, 21);
[C2, labels2] = km(C, 21, 100, C(:,sidx));
labels3 = labels2(labels);












