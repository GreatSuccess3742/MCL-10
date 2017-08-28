srcPath = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/data_generation/MCL-10_256X256/';
dstPath = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/data_generation/MCL-10_256X256_Black/';
className = 'truck/';
srcFiles = dir(strcat(srcPath,className,'*.jpg'));  % the folder in which ur images exists
imgSize = 256;
for i = 1 : length(srcFiles)
    filename = strcat(srcPath,className,srcFiles(i).name);
    my_image = imread(filename);
    image_thresholded = my_image;
    for h = 1 : imgSize
        for w = 1 : imgSize
            if( (image_thresholded(h,w,1) > 230) && (image_thresholded(h,w,2) > 230) && (image_thresholded(h,w,3) > 230))
                image_thresholded(h,w,:) = 0;
            end
        end
    end
    
%     % read in tiff image and convert it to double format
%     my_image = my_image(:,:,1);
%     % allodoge space for thresholded image
%     image_thresholded = zeros(size(my_image));
%     % loop over all rows and columns
%     for ii=1:size(my_image,1)
%         for jj=1:size(my_image,2)
%          % get pixel value
%              pixel=my_image(ii,jj);
%           % check pixel value and assign new value
%                if pixel> 230
%                     new_pixel=128;
%                else
%                     new_pixel = pixel;
%                end
%           % save new pixel value in thresholded image
%           image_thresholded(ii,jj)=new_pixel;
%         end
%     end
new_filename = strcat(dstPath,className,srcFiles(i).name);
% imwrite(image_thresholded,new_filename)
imwrite(image_thresholded,new_filename,'jpg')
% display result
% figure()
% % subplot(1,2,1)
% % imshow(my_image,[])
% % % title('original image')
% subplot(1,2,2)
% imshow(image_thresholded,[])
% title('thresholded image')
end