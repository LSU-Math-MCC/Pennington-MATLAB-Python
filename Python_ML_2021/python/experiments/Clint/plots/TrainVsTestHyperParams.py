import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as mcoll
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

label_font = {'family': 'normal',
              #'weight': 'bold',
              'size': 16}
font = {'family': 'normal', 'size': 14}
mpl.rc('font', **font)

#results_dict ={'regressor': {0: 'Ridge', 1: 'Ridge', 2: 'Ridge', 3: 'Ridge', 4: 'Ridge', 5: 'Ridge', 6: 'Ridge', 7: 'Ridge', 8: 'Ridge', 9: 'Ridge', 10: 'Ridge', 11: 'Ridge', 12: 'Ridge', 13: 'Ridge', 14: 'Ridge', 15: 'Ridge', 16: 'Ridge', 17: 'Ridge', 18: 'Ridge', 19: 'Ridge'}, 'target': {0: 'TOTAL_LEAN', 1: 'TOTAL_LEAN', 2: 'TOTAL_LEAN', 3: 'TOTAL_LEAN', 4: 'TOTAL_LEAN', 5: 'TOTAL_LEAN', 6: 'TOTAL_LEAN', 7: 'TOTAL_LEAN', 8: 'TOTAL_LEAN', 9: 'TOTAL_LEAN', 10: 'TOTAL_LEAN', 11: 'TOTAL_LEAN', 12: 'TOTAL_LEAN', 13: 'TOTAL_LEAN', 14: 'TOTAL_LEAN', 15: 'TOTAL_LEAN', 16: 'TOTAL_LEAN', 17: 'TOTAL_LEAN', 18: 'TOTAL_LEAN', 19: 'TOTAL_LEAN'}, 'dataset': {0: 'TrimmedStykuDataSet', 1: 'TrimmedStykuDataSet', 2: 'TrimmedStykuDataSet', 3: 'TrimmedStykuDataSet', 4: 'TrimmedStykuDataSet', 5: 'TrimmedStykuDataSet', 6: 'TrimmedStykuDataSet', 7: 'TrimmedStykuDataSet', 8: 'TrimmedStykuDataSet', 9: 'TrimmedStykuDataSet', 10: 'TrimmedStykuDataSet', 11: 'TrimmedStykuDataSet', 12: 'TrimmedStykuDataSet', 13: 'TrimmedStykuDataSet', 14: 'TrimmedStykuDataSet', 15: 'TrimmedStykuDataSet', 16: 'TrimmedStykuDataSet', 17: 'TrimmedStykuDataSet', 18: 'TrimmedStykuDataSet', 19: 'TrimmedStykuDataSet'}, 'SEX': {0: 'M/F', 1: 'M/F', 2: 'M/F', 3: 'M/F', 4: 'M/F', 5: 'M/F', 6: 'M/F', 7: 'M/F', 8: 'M/F', 9: 'M/F', 10: 'M/F', 11: 'M/F', 12: 'M/F', 13: 'M/F', 14: 'M/F', 15: 'M/F', 16: 'M/F', 17: 'M/F', 18: 'M/F', 19: 'M/F'}, 'features': {0: 'BMI', 1: 'BMI', 2: 'BMI', 3: 'BMI', 4: 'BMI', 5: 'BMI', 6: 'BMI', 7: 'BMI', 8: 'BMI', 9: 'BMI', 10: 'BMI', 11: 'BMI', 12: 'BMI', 13: 'BMI', 14: 'BMI', 15: 'BMI', 16: 'BMI', 17: 'BMI', 18: 'BMI', 19: 'BMI'}, 'n_samples': {0: 341, 1: 341, 2: 341, 3: 341, 4: 341, 5: 341, 6: 341, 7: 341, 8: 341, 9: 341, 10: 341, 11: 341, 12: 341, 13: 341, 14: 341, 15: 341, 16: 341, 17: 341, 18: 341, 19: 341}, 'alpha': {0: 0.0, 1: 2.1052631578947367, 2: 4.2105263157894735, 3: 6.315789473684211, 4: 8.421052631578947, 5: 10.526315789473683, 6: 12.631578947368421, 7: 14.736842105263158, 8: 16.842105263157894, 9: 18.94736842105263, 10: 21.052631578947366, 11: 23.157894736842103, 12: 25.263157894736842, 13: 27.36842105263158, 14: 29.473684210526315, 15: 31.57894736842105, 16: 33.68421052631579, 17: 35.78947368421053, 18: 37.89473684210526, 19: 40.0}, 'mean_train_r2': {0: 0.7708434538325916, 1: 0.7705415037816964, 2: 0.7697014629171941, 3: 0.7684096504702909, 4: 0.7667382852688893, 5: 0.7647480085608687, 6: 0.7624899107882992, 7: 0.7600071688251994, 8: 0.7573363754130322, 9: 0.7545086239470101, 10: 0.751550397727558, 11: 0.7484843021090415, 12: 0.7453296697948113, 13: 0.7421030632189809, 14: 0.7388186930622467, 15: 0.7354887681319803, 16: 0.7321237888427613, 17: 0.7287327941727444, 18: 0.725323570100517, 19: 0.7219028260376031}, 'std_train_r2': {0: 0.017425916478019693, 1: 0.017442836752373226, 2: 0.017488917827476604, 3: 0.017558251828410986, 4: 0.017646115960811355, 5: 0.017748687112634186, 6: 0.017862834577285557, 7: 0.01798596853879334, 8: 0.018115928282036984, 9: 0.018250898728920276, 10: 0.018389347244879683, 11: 0.018529975041049633, 12: 0.018671679178241012, 13: 0.018813522358421136, 14: 0.018954708513937293, 15: 0.019094562779947262, 16: 0.01923251483662203, 17: 0.019368084887708505, 18: 0.019500871738079247, 19: 0.01963054257077958}, 'mean_test_r2': {0: 0.7039659443428269, 1: 0.7044367682976294, 2: 0.7042602696220985, 3: 0.70353891316085, 4: 0.7023583761364517, 5: 0.7007905626468944, 6: 0.6988960238975856, 7: 0.6967259119850212, 8: 0.6943235652835836, 9: 0.691725801170425, 10: 0.6889639749663206, 11: 0.6860648511468123, 12: 0.6830513230572032, 13: 0.6799430097959541, 14: 0.6767567530621763, 15: 0.673507032186129, 16: 0.6702063119729367, 17: 0.6668653351611119, 18: 0.6634933690568426, 19: 0.6600984141217197}, 'std_test_r2': {0: 0.12001390565069095, 1: 0.11767184266484297, 2: 0.11552132156538539, 3: 0.1135192026995404, 4: 0.11163199975067661, 5: 0.10983381348493365, 6: 0.10810469313796556, 7: 0.10642934278232558, 8: 0.10479610333109017, 9: 0.10319615320367605, 10: 0.10162288156721831, 11: 0.10007139730000235, 12: 0.09853814445146518, 13: 0.09702060115859909, 14: 0.09551704392731702, 15: 0.09402636310919402, 16: 0.09254791849131613, 17: 0.09108142633642825, 18: 0.08962687110112291, 19: 0.08818443653465885}, 'mean_fit_time': {0: 0.46854658126831056, 1: 0.0013966560363769531, 2: 0.0009922027587890626, 3: 0.0009913444519042969, 4: 0.0010106086730957032, 5: 0.0009978294372558593, 6: 0.0010009288787841796, 7: 0.0003989696502685547, 8: 0.0009946346282958985, 9: 0.0005983352661132812, 10: 0.0007983684539794922, 11: 0.00099639892578125, 12: 0.000997161865234375, 13: 0.0007976531982421875, 14: 0.0005976676940917969, 15: 0.0005981922149658203, 16: 0.0009975433349609375, 17: 0.000598287582397461, 18: 0.0007975578308105468, 19: 0.000598287582397461}}
#data = pd.DataFrame(results_dict)

data = pd.read_csv('../../reports/paper_data.csv')

# See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
#colormaps = ['gray', 'pink', 'summer', 'hot', 'Oranges', 'YlOrBr', 'cool', 'spring', 'Greens', 'YlGn']
#colormaps = ['Purples', 'RdPu', 'Blues', 'GnBu', 'Oranges', 'YlOrBr', 'cool', 'spring', 'Greens', 'YlGn']
paramSets = np.unique(data['features'])

def make_colormap(seq):
    """ Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


rgb = mcolors.ColorConverter().to_rgb
colormaps = []
for c in ['blue','cyan','red','magenta','green','black']:
    colormaps += [make_colormap([rgb(c), rgb('white')])]


fig, ax = plt.subplots()

lines = []
for params, cmap in zip(paramSets, colormaps):
    data0 = data.loc[data['features'] == params]
    x = data0['mean_train_r2']
    y = data0['mean_test_r2']
    alpha = data0['alpha']
    ax.scatter(x, y, c=alpha, cmap=cmap,
               alpha=.5,
               s=10
               )
    lines += [Line2D([0], [0], color=cmap(.2), linewidth=3)]

# Plot the line y = x
lims = [
    np.min([
        ax.get_xlim(),
        ax.get_ylim()
    ]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k-', zorder=0, color='grey', linestyle='--', linewidth=2.5)

# Label plot
target_dict = {'TOTAL_FAT': 'Fat Mass', 'TOTAL_LEAN': 'Lean Mass', 'TOTAL_PFAT': 'Percent Fat'}
#plt.title(f'{data["regressor"][0]} on {target_dict[data["target"][0]]}', **font)
plt.xlabel('Mean train R²', **label_font)
plt.ylabel('Mean test R²', **label_font)

plt.legend(lines, paramSets, loc='upper left')

# All bullshit to plot a colorbar
alpha_max = np.max(data['alpha'])
norm = mpl.colors.Normalize(vmin=0, vmax=alpha_max)
sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('gray'), norm=norm) # define scalar mappable like it's in the plot
sm.set_array([])
cb = plt.colorbar(sm,
             ticks=[0,alpha_max],
             boundaries=np.arange(0, alpha_max + .001, .001))
cb.set_label('α (regularization parameter)', **label_font)

plt.savefig(f'{data["regressor"][0]} {data["target"][0]} alpha 0 {alpha_max}.png'.replace(' ','_'))
plt.show()