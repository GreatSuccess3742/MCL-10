% Put the file under the mask folder
file = dir ('*.png');
[numOfFile,dummy] = size(file);
dstPath = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/rebuild/VOCdevkit/YeMask/Class/';

for fileIndex = 1 : numOfFile
    img = imread(file(fileIndex).name);
    numberOfObjects = length(unique(img)) - 2;
    objectClass = unique(img);
    if(objectClass(end) ~= 255)
        numberOfObjects = length(unique(img)) -1;
    end
    for objectIndex = 1:numberOfObjects
        curObjectClass =  objectClass(objectIndex + 1);
        singleObjectMask = img;
        singleObjectMask(img ~= curObjectClass) = 0;
        singleObjectMask(singleObjectMask == curObjectClass) = 255;
        outputFilename = strcat(dstPath,int2str(curObjectClass),'/',file(fileIndex).name);
        outputFilename(end-2:end) = 'jpg';
        imwrite(singleObjectMask,outputFilename);
    end
end