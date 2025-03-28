function [bbox] = Copy_of_demo(no_compile)

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

startup;

if ~exist('no_compile', 'var')
  fprintf('compiling the code...');
  compile;
  fprintf('done.\n\n');
end

load('voc-dpm/2012/cow_final');
% load('VOC2007/cow_final.mat');
model.vis = @() visualizemodel(model, ...
                  1:2:length(model.rules{model.start}));
bbox = test('cow_dataset/images/000000032700.jpg', model, 1);

function [bbox] = test(imname, model, num_dets)
cls = model.class;
fprintf('///// Running demo for %s /////\n\n', cls);

% load and display image
im = imread(imname);
clf;
image(im);
axis equal; 
axis on;
title('input image');
disp('input image');
disp('press any key to continue'); pause;
disp('continuing...');

% load and display model
model.vis();
disp([cls ' model visualization']);
disp('press any key to continue'); pause;
disp('continuing...');

% detect objects
tic;
[ds, bs] = imgdetect(im, model, -1);
toc;
top = nms(ds, 0.5);
% top = top(1:min(length(top), num_dets));
ds = ds(top, :);
bs = bs(top, :);
clf;
if model.type == model_types.Grammar
  bs = [ds(:,1:4) bs];
end
showboxes(im, reduceboxes(model, bs));
title('detections');
disp('detections');
disp('press any key to continue'); pause;
disp('continuing...');

if model.type == model_types.MixStar
  % get bounding boxes
  bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
  bbox = clipboxes(im, bbox);
  top = nms(bbox, 0.5);
  clf;
  showboxes(im, bbox(top,:));
  title('predicted bounding boxes');
  disp('bounding boxes');
  disp('press any key to continue'); pause;
end

fprintf('\n');
