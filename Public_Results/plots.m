clc; close all; clear all; 
  names ={ 
  'FISTA100';
  'LISTA0t';
  'LISTA1t';
  'LISTA3t';
  'LISTA0ut';
  'LISTA1ut';
  'LISTA3ut';
  'ReLU0ut';
  'ReLU1ut';
  'ReLU3ut';
  'ReLU0t';
  'ReLU1t';
  'ReLU3t'};
train_loss1 = 0.01*[1.8501 1.5420 1.3103 1.2546 1.5420 1.3131 1.3881 1.6988 1.9291 1.3273 1.7787 1.7053 1.5074]; 
test_loss1 = 0.01*[1.8512 1.5419 1.3106 1.2549 1.5419 1.3134 1.3879 1.6977 1.9268 1.3277 1.7773 1.7038 1.5090]; 
train_loss2 = 0.01*[1.8501 1.5421 1.3376 1.2641 1.5427 1.3225 1.3465 1.7677 1.7796 1.2990 1.8800 2.0160 1.2766]; 
test_loss2 = 0.01*[1.8512 1.5421 1.3379 1.2645 1.5426 1.3228 1.3469 1.7664 1.7781 1.2994 1.8782 2.0129 1.2773]; 

figure(1); 
title('LASSO Loss') 
plot(train_loss1,'b--*'); hold on; 
plot(train_loss2,'r--*')
set(gca,'XTick', 1:13)
plot(1:size(names,1),test_loss1,'ro'); 
set(gca, 'XTickLabel',names); 
xlabel('encoder architecture')
ylabel('lasso loss') 

line([1.5 1.5], get(gca, 'ylim'));
line([4.5 4.5], get(gca, 'ylim'));
line([7.5 7.5], get(gca, 'ylim'));
line([10.5 10.5], get(gca, 'ylim'));

figure(2)
hold on; 
scatter_train1 = [
 0.0999  0.9613;
 0.0488  0.6024;
 0.0414  0.7604;
 0.0395  0.7829;
 0.0488  0.6028;
 0.0414  0.7661;
 0.0429  0.7712;
 0.0552  0.7994;
 0.0606  0.8036;
 0.0432  0.8486;
 0.0578  0.8117;
 0.0556  0.7889;
 0.0559  0.8398];

scatter_test1 = [
 0.1001  0.9613;
 0.0488  0.6025;
 0.0415  0.7604;
 0.0396  0.7829;
 0.0489  0.6029;
 0.0414  0.7661;
 0.0429  0.7712;
 0.0552  0.7996;
 0.0606  0.8036;
 0.0433  0.8487;
 0.0578  0.8118;
 0.0556  0.7889;
 0.0560  0.8399];

scatter_train2 = [
 0.0999  0.9613;
 0.0488  0.6032;
 0.0419  0.7280;
 0.0396  0.7584;
 0.0489  0.6035;
 0.0415  0.7419;
 0.0417  0.7081;
 0.0566  0.7789;
 0.0576  0.7984;
 0.0420  0.8331;
 0.0596  0.7914;
 0.0628  0.8102;
 0.0422  0.8395];

scatter_test2 = [
 0.1001  0.9613;
 0.0488  0.6033;
 0.0419  0.7281;
 0.0397  0.7584;
 0.0490  0.6036;
 0.0416  0.7419;
 0.0418  0.7082;
 0.0567  0.7790;
 0.0576  0.7984;
 0.0421  0.8332;
 0.0596  0.7915;
 0.0628  0.8103;
 0.0423  0.8396];


xlabel('Relative Rec. Error') 
ylabel('Sparsity') 
lscatter(scatter_train1(:,1),scatter_train1(:,2),names,'TextColor','b');
lscatter(scatter_train2(:,1),scatter_train2(:,2),names,'TextColor','r');
% plot(scatter_train(:,1),scatter_train(:,2),'b*');  
% plot(scatter_test(:,1),scatter_test(:,2),'r.');  



