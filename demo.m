% This demo shows how to extract HIPHOP feature 
% It depends on the caffe/matcaffe framework (http://caffe.berkeleyvision.org/)
% make sure that both caffe and matcaffe are installed successfully
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
clear,clc,close all
caffe_path = '/mnt/sda1/ycchen/research/caffe-master';
while ~exist(caffe_path,'dir')
	caffe_path=input('please input your caffe path:','s');
end
addpath([caffe_path,'/matlab'])

prototxtFilePath = 'Alexnet.prototxt';
caffemodelPath = 'Alexnet.caffemodel';
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(prototxtFilePath, caffemodelPath, 'test');

dataset_dir = 'images';
imageExtensionName = 'bmp';
opt.batch = 2; % only for demo. you can increase it if you work on larger dataset
[X,name_list] = HIPHOP(net,dataset_dir,imageExtensionName,opt);
size(X)
name_list
