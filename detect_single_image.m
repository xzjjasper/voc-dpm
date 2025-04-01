function [boxes, scores] = detect_single_image(image_path, model_path)
    % DETECT_SINGLE_IMAGE Perform DPM detection on a single image
    %
    % Parameters:
    %   image_path - Path to the input image
    %   model_path - Path to the DPM model file
    %
    % Returns:
    %   boxes  - Nx4 matrix of bounding boxes [x1, y1, x2, y2]
    %   scores - Nx1 vector of detection scores
    
    % Load model
    model_data = load(model_path);
    if isfield(model_data, 'model')
        model = model_data.model;
    else
        model = model_data;
    end
    
    % Read image
    im = imread(image_path);
    
    % Perform detection
    [ds, bs] = imgdetect(im, model, -1);
    
    % Apply non-maximum suppression
    top = nms(ds, 0.5);
    ds = ds(top, :);
    bs = bs(top, :);
    
    % Get bounding boxes
    if isfield(model, 'bboxpred')
        bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
        bbox = clipboxes(im, bbox);
        top = nms(bbox, 0.5);
        bbox = bbox(top, :);
    else
        bbox = reduceboxes(model, bs);
    end
    
    % Extract boxes and scores
    boxes = bbox(:, 1:4);
    scores = bbox(:, 5);
end 