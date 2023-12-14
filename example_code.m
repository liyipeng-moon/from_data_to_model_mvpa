%cd C:\Users\DELL\Desktop\example\
close all ; clear; clc;
addpath(genpath(pwd))
tic
load('category_label.mat')
load("average_roi_beta.mat") %beta averaged across 6 trials in 3rois
boot_times =50;
rng(1009)
%% 2-calss svm in 3 rois
interested_idx = [image_category(1).picset, image_category(6).picset]; % decode animal or face?
label_array = [zeros(1,length(image_category(1).picset)), ones(1,length(image_category(6).picset))];
all_image = length(label_array);
leave_out_num = 20;
for bb = 1:boot_times
    leave_out_idx = randperm(all_image,leave_out_num); % split data into train and test sets
    train_idx = setdiff(1:all_image, leave_out_idx);
    for rr = 1:length(average_beta)
        roi_now = average_beta{rr,2};
        beta_now = average_beta{rr,1}(interested_idx,:);
        mdl = fitcsvm(beta_now(train_idx, :), label_array(train_idx)); % train a svm model
        predicted_label = predict(mdl, beta_now(leave_out_idx, :)); % predict test data
        predicted_acc(rr,bb) = length(find(predicted_label'==label_array(leave_out_idx)))./leave_out_num; %check accuracy
        roi_legend{rr} = roi_now;
    end
end

figure
bar(mean(predicted_acc,2));
yline(0.5) % change performance
xticklabels(roi_legend)
ylabel('Predicting Accuracy')
title('Between-Class(animal or face) svm in 3 ROIS')



%% multi-class classification

tic
load('roi_beta.mat') % beta in 3 rois for 6 trials
load("category_label.mat") % category label
% load('toy_data.mat') % beta in LOC for 6 trials

% sort categories into 4 sets
image_category(3).name = 'man-made object';
image_category(3).picset = [image_category(2).picset, image_category(3).picset, image_category(4).picset, image_category(7).picset, image_category(8).picset, image_category(11).picset, image_category(12).picset];
image_category(2)=image_category(6);
image_category(4).name='natural object';
image_category(4).picset = [image_category(5).picset, image_category(9).picset];
image_category(5:end)=[];
category_legend={}; 

leave_out_num = 96;

confuse_matrix = zeros(boot_times, 4, length(image_category), length(image_category));
for bb = 1:boot_times % we do this for boot_times

    % combine 6 trials, and pick data
    pooled_response{1}=[];pooled_response{2}=[];pooled_response{3}=[];pooled_response{4}=[];label_array=[];
    for cc = 1:length(image_category)
        sample_image = randperm(length(image_category(cc).picset), 40);
        image_idx = image_category(cc).picset(sample_image);
        for trial_num = 1:length(roi_beta)
            pooled_response{1} = [ pooled_response{1};roi_beta{trial_num}.ffa(image_idx, :)];
            pooled_response{2} = [ pooled_response{2};roi_beta{trial_num}.eba(image_idx, :)];
            pooled_response{3} = [ pooled_response{3};roi_beta{trial_num}.ppa(image_idx, :)];
            pooled_response{4} = [ pooled_response{4};roi_beta{trial_num}.loc(image_idx, :)];
        end
        label_array = [label_array, cc*ones(1,trial_num*40)];
        category_legend{cc}=image_category(cc).name;
    end

    % split data into train & test sets
    test_idx = randperm(length(label_array),leave_out_num); 
    train_idx = setdiff(1:length(label_array), test_idx);
    for rr = 1:4 % do this for 4 rois
            train_data = double(pooled_response{rr}(train_idx, :));
            train_label = label_array(train_idx);
            test_data = double(pooled_response{rr}(test_idx, :));
            test_label = label_array(test_idx);

            % train a multi-label svm model and predict labels for test sets
            model=svmtrain(train_label', train_data);
            [pred, acc, ~] = svmpredict(test_label', test_data,model);
            acc_save(rr,bb)= acc(1); % save accuracy
            
            % save confuse matrix
            for sample_now = 1:length(test_label)
                confuse_matrix(bb, rr ,test_label(sample_now), pred(sample_now))=confuse_matrix(bb, rr ,test_label(sample_now), pred(sample_now))+1;
            end
    end
    waitbar(bb/boot_times)
end

figure
bar(mean(acc_save,2));
yline(100/length(image_category))
xticklabels(roi_legend)
ylabel('classification accuracy(%)')
title('multi-class classification')

figure
mean_confuse_matrix = squeeze(mean(confuse_matrix,1));
for rr = 1:4
    subplot(2,2,rr)

    heatmap(category_legend,category_legend,squeeze(mean_confuse_matrix(rr, :,:)))
    title(['classification result of ' roi_legend{rr}])
end
sgtitle('multi-label classification ')
set(gcf, 'position',[1100,280,930,702])

%% RSA

load('average_roi_beta.mat')
load("category_label.mat")
% sort categories
image_idx = [];
for cc = 1:length(image_category)
    image_idx = [image_idx image_category(cc).picset];
end
image_idx = image_idx(1:2:500);
figure
for rr = 1:length(average_beta)
    subplot(2,2,rr)
    beta_now = average_beta{rr,1};

    beta_now = beta_now(image_idx, :);
    
    pd = squareform(pdist(beta_now,'correlation'));
    %pd = tril(pd, -1);
    %pd(pd~=0)=zscore(pd(pd~=0));
    %pd =reshape(zscore(pd(:)), 500,[]);
    imagesc(pd)
    title(['RDM for ' roi_legend{rr}])
    axis off
    colorbar
    pd_save{rr}=pd;
end
colormap('turbo')
set(gcf, 'position',[744,150,1100,900])


figure
for rx = 1:length(average_beta)
    for ry = 1:length(average_beta)
        rs(rx, ry) = corr(pd_save{rx}(find(tril(ones(size(pd)), -1))) , pd_save{ry}(find(tril(ones(size(pd)), -1))));
    end
end
heatmap(roi_legend,roi_legend,rs)
title('similarity across 4 rois')
set(gcf, 'position',[744,653,473,396])
toc