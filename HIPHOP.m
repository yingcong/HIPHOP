% HIPHOP feature extraction
% NOTE: this function requires caffe library for generating the featuremap
% make sure that the caffe is installed correctly and add */caffe/matlab
% in the working path
%
% input:
% dataset_dir: the directory of the dataset
% imageExtensionName: the extension name of the image. it should be supported 
% by the "imread" function
% prototxtFiledataset_dir: the dataset_dir of the deploy prototxt file
% cafffemodeldataset_dir: the dataset_dir of the corresponding caffemodel 
% opt: optional options. For expert only. You may need cross-validation when
% switching to another CNN model rather than Alexnet.
% - batch: run feature extraction in batch for acceleration. 
%   if you have out-of-memory problem, please reduce this parameter
% - pixelOfStripe: specify the length of each horizontal stripe
% - edges: the edge vector for converting the activation intensity to histogram in HIP
% - top_k: the top-k featuremap index for HOP part
% - ratio: set the ratio between HIP and HOP d
% 
% output:
% Feat: the column-wise feature matrix (84096 dim * nSample)
% name_list: a cell that contains the names of the samples.
%
% If you have any questions/suggestions regarding the code or the paper, 
% please feel free to contact me.  Email: yingcong.ian.chen@gmail.com
% Ying-Cong Chen, 1/24/2017
%
% ----------------------------------------------------------------------------
% Please kindly cite our paper if you find this code is useful:              %
% Ying-Cong Chen, Xiatian Zhu, Wei-Shi Zheng, and Jianhuang Lai,             %
% Person Re-Identification by Camera Correlation Aware Feature Augmentation  %
% Pattern Analaysis and Machine Intelligence, 2017                           %
% ----------------------------------------------------------------------------
function [Feat,name_list] = HIPHOP(net,dataset_dir,imageExtensionName,opt)

% The default parameters of HIPHOP
batch = 32; % if you have out-of-memory problem, please reduce this parameter
pixelOfStripe = 5;
top_k = 20;
edges = linspace(0,255,16);
if nargin > 3
    if isfield(opt,'batch')
        batch = opt.batch;
    end
    if isfield(opt,'pixelOfStripe')
        pixelOfStripe = opt.pixelOfStripe;
    end
    if isfield(opt,'top_k')
        top_k = opt.top_k;
    end
    if isfield(opt,'edges')
        edges = opt.edges;
    end
end
files = dir([dataset_dir,'/*.',imageExtensionName]);
% forward the model and get the feature maps of the 1st and 2ed conv layers 
dim = 84096;
Feat = zeros(dim,numel(files));
name_list = cell(numel(files),1);

if numel(files) - batch >= 1 
    net.blobs('data').reshape([227,227,3,batch]);
    for i = 1 : batch : numel(files) - batch
        I_batch = zeros(227,227,3,batch);
    %     I_batch = [];
        for j = 0 : 1 : batch -1
            name = files(i+j).name;
            name_list{i+j} = name;
            I = imread([dataset_dir,'/',name]);
            I = single(I);
            I = imresize(I,[227,227]);
            I = I(:,:,[3,2,1]);
            I = permute(I,[2,1,3]);
            %         I_batch{k} = I;
            I_batch(:,:,:,j+1) = I;
        end
        %         net.blobs('data').reshape([227,227,3,size(I_batch,4)]);
        net.forward({I_batch});
        conv1 = net.blobs('conv1').get_data();
        conv2 = net.blobs('conv2').get_data();
        
        parfor j = 0 : batch -1
            Feat(:,j+i) =  HIPHOP_extraction(conv1(:,:,:,j+1),conv2(:,:,:,j+1),pixelOfStripe,edges,top_k);
        end
        fprintf('Feature Extraction in batch. Current index:%d. %2.2f%% finished.\n',batch -1+i,100*(batch -1+i)/(numel(files)));
    end
    i_final = i;
else
    i_final = 1-batch;
end
net.blobs('data').reshape([227,227,3,1]);
fprintf('from %d to %d\n',i_final+ batch,numel(files));
k=1;
for j = i_final + batch:numel(files)
    name = files(j).name;
    name_list{j} = name;
    I = imread([dataset_dir,'/',name]);
    I = single(I);
    I = imresize(I,[227,227]);
    I = I(:,:,[3,2,1]);
    I = permute(I,[2,1,3]);
    net.forward({I});
    conv1 = net.blobs('conv1').get_data();
    conv2 = net.blobs('conv2').get_data();
    Feat(:,j) =  HIPHOP_extraction(conv1,conv2,pixelOfStripe,edges,top_k);
    fprintf('current index:%d. %2.2f%% finished.\n',j,100*j/(numel(files)));
%     fprintf('i=%d size(Feat,2)= %d \n',j,size(Feat,2));
end
fprintf('finish feature extraction in one set\n')
end

%% -----------------------------------------------------------------
%% HIPHOP_extraction: function description
function feat = HIPHOP_extraction(conv1,conv2,pixelOfStripe,edges,top_k)
HIP1 = HIP(conv1,pixelOfStripe,edges);
HIP2 = HIP(conv2,pixelOfStripe,edges);
HOP1 = HOP(conv1,pixelOfStripe,top_k);
HOP2 = HOP(conv2,pixelOfStripe,top_k); 
feat_layer1 = [HIP1;HOP1];
feat_layer2 = [HIP2;HOP2];
feat = [normc(feat_layer1);normc(feat_layer2)];
% feat = sqrt(feat);
end 



function hist_layer = HIP(conv,pixelOfStripe,edges)
conv = permute(conv,[2,1,3]); % rotate the feature map to fit matlab standard
mask = fspecial('gaussian',size(conv,1),0.3*size(conv,1));
% extract histograms for each stripe of the conv
rows = size(conv,1);
stripeNum = floor(rows/pixelOfStripe);
hist_layer = [];
for level = 1 : size(conv,3)
    hist_map = [];
    for i = 1 : stripeNum
        start_ = pixelOfStripe * (i-1) + 1;
        end_ = pixelOfStripe * i;
        if i == stripeNum
            end_ = rows;
        end
        box = conv(start_ : end_,:, level);
        box_mask = mask(start_ : end_,:);
        hist_sripe = hip_stripe(box,box_mask,edges)';
        hist_sripe = hist_sripe / (norm(hist_sripe) + eps);
        hist_map = [hist_map; hist_sripe];
    end
    hist_layer = [hist_layer;hist_map];
end
end
% extract HIP feature for each stripe
function hist_ = hip_stripe(box,mask,edges)
edges = edges(:);
box = box(:)';
box_mat=  repmat(box,[numel(edges),1]);
edges_mat = repmat(edges,[1,numel(box)]);
dif = abs(edges_mat - box_mat);
[~,idx] = min(dif,[],1);
mask = mask(:)';
hist_ = zeros(1,numel(edges));
for i = 1 : numel(idx)
    hist_(idx(i)) = hist_(idx(i)) + mask(i);
end
end

%%
function hist_layer = HOP(conv,pixelOfStripe,top_k)
% permute the conv
conv = permute(conv,[2,1,3]);

mask = fspecial('gaussian',size(conv,1),0.3*size(conv,1));

modeNum = size(conv,3);

[~,hop_map] = sort(conv,3,'descend');
hop_map = hop_map(:,:,1:top_k);
hop_map = uint16(hop_map);

% extract histograms for each stripe of the conv
rows = size(conv,1);
stripeNum = floor(rows/pixelOfStripe);
hist_layer = [];

for mode_level = 1 : top_k
    hist_map = [];
    for i = 1 : stripeNum
        start_ = pixelOfStripe * (i-1) + 1;
        end_ = pixelOfStripe * i;
        if i == stripeNum
            end_ = rows;
        end
        box = hop_map(start_ : end_,:, mode_level);
        box_mask = mask(start_ : end_,:);
        hist_sripe = hop_stripe(box,modeNum,box_mask);
        hist_sripe = hist_sripe / (norm(hist_sripe) + eps);    
        hist_map = [hist_map; hist_sripe];
    end
    hist_layer = [hist_layer;hist_map];
end
end

% get the histogram from competetive mode
function hist_ = hop_stripe(box,bin,mask)
box = box(:);
mask = mask(:);
hist_ = zeros(bin,1);
for i = 1 : numel(box)
    hist_(box(i)) = hist_(box(i)) + mask(i);
end
end
