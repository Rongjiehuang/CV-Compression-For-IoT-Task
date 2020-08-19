%% Cityscapes

mIoU = [67.95,67.26,66.81,64.92];
size = [2280,1120,466,232];
plot(mIoU,size,'--','color','r')
hold on
% plot(mIoU,size,'o','color','r')
set(gca,'YDir','reverse')
scatter(67.95,2280,'o','b','linewidth',2)
scatter(67.26,1120,'^','r','linewidth',2)
scatter(66.81,466,'v','g','linewidth',2)
scatter(64.92,232,'x','y','linewidth',2)
scatter(66.79,203,'p','m','linewidth',2)
legend('','PNG','JPEG Q=100','JPEG Q=95','JPEG Q=85','GRACE')
xlabel('mIoU(%)')
ylabel('Image Size(KB)')
title('CityScapes')

%% ImageNet
figure;
Prec1 = [75.138 ,75.076 ,74.986];
Prec5 = [92.452 ,92.444 ,92.370];
size = [217,135,79];
plot(Prec1,size,'--','color','r')
hold on

% set(gca,'YDir','reverse')
scatter(75.138,217,'o','b','linewidth',2)
scatter(75.076,135,'^','r','linewidth',2)
scatter(74.986,79,'v','g','linewidth',2)
legend('Top1-Prec','JPEG Q=100','JPEG Q=95','JPEG Q=85')
xlabel('Prec(%)')
ylabel('Image Size(KB)')
title('ImageNet')


figure;
plot(Prec5,size,'--','color','b')
hold on
% set(gca,'YDir','reverse')
scatter(92.452,217,'o','b','linewidth',2)
scatter(92.444,135,'^','r','linewidth',2)
scatter(92.370,79,'v','g','linewidth',2)
legend('Top5-Prec','JPEG Q=100','JPEG Q=95','JPEG Q=85')
xlabel('Prec(%)')
ylabel('Image Size(KB)')
title('ImageNet')



