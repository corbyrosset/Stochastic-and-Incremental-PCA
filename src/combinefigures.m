function [] = combinefigures ()

open('cross-val-IPCA.png');
h_ipca = get(gca, 'Children');
x_ipca = get(h_ipca, 'XData');
y_ipca = get(h_ipca, 'YData');

open('cross-val-PCA.fig');
h_pca = get(gca, 'Children');
x_pca = get(h_pca, 'XData');
y_pca = get(h_pca, 'YData');


figure 
subplot(121)
plot(x_ipca, y_ipca);
subplot(122)
plot(x_pca, y_pca);
saveas(gcf, 'cross-val-All', 'fig');

end