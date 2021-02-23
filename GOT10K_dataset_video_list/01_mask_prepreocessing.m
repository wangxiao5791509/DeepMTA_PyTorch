%% 
clc; clear all; close all; warning off;

path = '/media/wangxiao/4T_wangxiao/GOT-10K_dataset/train/';
files = dir(path);
files = files(3:end);

for i = 1:size(files, 1)
    
    disp(['==>> deal with ', num2str(i), '/', num2str(size(files, 1))]);
    
    videoName = files(i).name;
    imgPath = [path videoName '/'];
    firstFrame = imread([imgPath '00000001.jpg']);
    
    gt_name = ['groundtruth.txt'];
    gt_file = importdata([path videoName '/' gt_name]);
    initial_BBox = gt_file(1, :);
    target_Object = imcrop(firstFrame, initial_BBox);
    target_Object = imresize(target_Object, [320 640]);
    savePath = [path videoName '/'];
    imwrite(target_Object, [savePath, 'init_targetObject.png']);
    
    maskSavePath = [path videoName '/resizedImage/'];
    mkdir(maskSavePath);
    imgfiles = dir([imgPath, '*.jpg']);
    for j=1:size(imgfiles, 1)
        image = imread([imgPath imgfiles(j).name]);
        image = imresize(image, [320 640]);
        imwrite(image, fullfile(maskSavePath, imgfiles(j).name),'jpg');
    end
    
    
    maskSavePath = [path videoName '/mask_imgs/'];
    mkdir(maskSavePath);
    imgfiles = dir([imgPath, '*.jpg']);
    for j=1:size(imgfiles, 1)
        image = imread([imgPath imgfiles(j).name]);
        
        BBox = gt_file(j, :);
        
        if BBox(1) <= 0 BBox(1)=1; end
        if BBox(2) <= 0 BBox(2)=1; end
        if BBox(3) <= 0 BBox(3)=1; end
        if BBox(4) <= 0 BBox(4)=1; end
        
        BinaryMap = zeros(size(image, 1), size(image, 2));
        for iidex = floor(BBox(1)):floor(BBox(1)+BBox(3))
            for jidex = floor(BBox(2)):floor(BBox(2)+BBox(4))
                BinaryMap(jidex, iidex) = 255;
            end
        end
        BinaryMap = imresize(BinaryMap, [320, 640]); 
        imwrite(BinaryMap, fullfile(maskSavePath, imgfiles(j).name),'jpg');
        
        
        
    end
    
    
end







