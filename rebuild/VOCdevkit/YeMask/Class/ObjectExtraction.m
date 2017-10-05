imgFolder = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/rebuild/VOCdevkit/VOC2012/JPEGImages/';
folderList = dir;
folderList = folderList(4:end-2);
currentDirectory = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/rebuild/VOCdevkit/YeMask/Class/';
dstPath = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/rebuild/VOCdevkit/YeMask/ImageClass/';
for folderIndex = 1:length(folderList)
    targetFolder = strcat(currentDirectory,folderList(folderIndex).name);
    cd (targetFolder);
    fileName = dir('*.jpg');
    for fileIndex = 1:length(fileName)
        jpgFileName = fileName(fileIndex).name;
        jpgFileName(end-2:end) = 'jpg';
        jpgFileName = strcat(imgFolder,jpgFileName);
        srcImg = imread(jpgFileName);
        srcImg = im2double(srcImg);
        
        mask = imread(fileName(fileIndex).name);
        mask = im2double(mask);
        mask(mask > 0.5) = 1;
        mask(mask < 0.5) = 0;
        
        output = srcImg .* mask;
        
        % Background color augmentation
        greyBackGroundImg = output;
        greyBackGroundImg(greyBackGroundImg == 0) = 0.5;
        
        whiteBackGroundImg = output;
        whiteBackGroundImg(whiteBackGroundImg == 0) = 1.0;
        
        % Output black background (original) images
%         outputFileName = strcat(dstPath,folderList(folderIndex).name,'/',fileName(fileIndex).name);
%         imwrite(output,outputFileName);

        % Output grey background images
        outputFileName = strcat(dstPath,folderList(folderIndex).name,'/grey_',fileName(fileIndex).name);
        imwrite(greyBackGroundImg,outputFileName);
        
        % Output white background images
        outputFileName = strcat(dstPath,folderList(folderIndex).name,'/white_',fileName(fileIndex).name);
        imwrite(whiteBackGroundImg,outputFileName);
    end
end