import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


if 'kz272' in os.path.expanduser('~'):
    code_path = '/gpfs/loomis/home.grace/kz272/scratch60/MSRR/Code/Sims20210615_Python/Grace_MSRR_Output'
elif 'kyzhou' in os.path.expanduser('~'):
    code_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output'
    data_path = '/Users/kyzhou/Dropbox/MSRR/Code/Empirical/Data'
elif 'malamud' in os.path.expanduser('~'):
    code_path = '/Users/malamud/Dropbox/MY_STUFF/RESEARCH/MSRR/Code/SemyonOutput'
    data_path = '/Users/malamud/Dropbox/MY_STUFF/RESEARCH/MSRR/Code/Empirical/Data'
else:
    code_path = 'SemyonOutput'

if not os.path.exists(code_path):
    os.mkdir(code_path)


def plot_TheoreticalMSE_logz(list_of_z, mse_theoretical, c_M, b_star, show = False):
    '''
    This function plots the theoretical MSE vs. log10(z)
    :param list_of_z:
    :param mse_theoretical:
    :param c_M:
    :param b_star:
    :return:
    '''
    save_path = os.path.join(code_path, 'Theoretical MSE vs. log10(z)')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, 'c_M %.4f' % (c_M))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, ax = plt.subplots()
    plt.plot(np.log10(list_of_z), mse_theoretical)
    plt.savefig(save_path + f'mse_theoretical_c_M %.4f_bstar %s.jpeg'%(c_M, b_star))

    # set x-axis label
    ax.set_xlabel(r'log10(z)')
    # set y-axis label
    ax.set_ylabel('Theoretical MSE', color="black", fontsize=12)
    plt.legend()

    plt.title('Theoretical MSE vs. log10(z), c_M %.4f, b_star %s' %(c_M, b_star))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + 'Theoretical MSE vs. log10(z), c_M %.4f, b_star %s' %(c_M, b_star)+ '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')

def plot_b_star(list_of_z, list_of_b_star, c_M, show = False):
    '''
    This function plots log(m(-z;c)) vs. log(z)
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''

    save_path = os.path.join(code_path, 'list_of_b_star')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, ax = plt.subplots()
    plt.plot(np.log10(list_of_z), np.log10(list_of_b_star), 'o', linestyle='solid', linewidth=1)

    # set x-axis label
    ax.set_xlabel(r'log10(z)')
    # set y-axis label
    ax.set_ylabel(r'log10($b_{*}$)', color="black", fontsize=12)
    plt.legend()

    plt.title(r'log10($b_*$) vs. log10(z), $c_M$ = %.4f' %(c_M))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + 'log10(b_star) vs. log10(z), c_M = %.4f' %(c_M)+ '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def plot_logm_logz(list_of_z, list_of_m, c_M, show = False):
    '''
    This function plots log(m(-z;c)) vs. log(z)
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''

    save_path = os.path.join(code_path, 'log10m_vs_log10z')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, ax = plt.subplots()
    plt.plot(np.log10(list_of_z), np.log10(list_of_m), linestyle='solid', linewidth=1)

    # set x-axis label
    ax.set_xlabel(r'log10(z)')
    # set y-axis label
    ax.set_ylabel('log10(m(-z;c))', color="black", fontsize=12)
    plt.legend()

    plt.title('log10(m(-z;c)) vs. log10(z), c_M = %.4f' %(c_M))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + 'log10(m(-z;c)) vs. log10(z), c_M = %.4f' %(c_M)+ '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def plot_zm_z(list_of_z, list_of_m, c_M, show = False):
    '''
    This function plots log(m(-z;c)) vs. log(z)
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''

    save_path = os.path.join(code_path, 'zm_vs_log10(z)')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, ax = plt.subplots()
    plt.plot(np.log10(list_of_z), list_of_z * list_of_m, linestyle='solid', linewidth=1)

    # set x-axis label
    ax.set_xlabel(r'log10(z)')
    # set y-axis label
    ax.set_ylabel('z*m(-z;c)', color="black", fontsize=12)
    plt.legend()

    plt.title('z*m(-z;c) vs. log10(z), c_M = %.4f' %(c_M))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + 'z*m(-z;c) vs. log10(z), c_M = %.4f' %(c_M) + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def series_refined_plot(save_path, c_M, array, title_str, list_of_z, para_str_title, para_str, \
                SRU_500_array = None, show = False, new_folder = True, perform_folder = False):
    '''
    This function plots the series of R-squared, SRU, returns, and risks
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''
    if new_folder:
        save_path = os.path.join(save_path, 'Refined')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if perform_folder:
        save_path = os.path.join(save_path, title_str)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    # list_of_z = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, \
    #                   1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6]
    z_select_idx = [1, 4, 6, 8, 10, 11, 12, 13]

    for i in z_select_idx: # range(array.shape[1]):
        plt.plot(c_M, array[:, i], linestyle='solid', linewidth=1,
                      label='log10(z) = %s' % (np.log10(list_of_z[i])))
    plt.axvline(x=1, color='grey', linewidth=1, linestyle='dashed')
    if title_str == 'SRU' and SRU_500_array is not None:
        plt.axhline(y=np.mean(SRU_500_array), color='black', linewidth=1, linestyle='dashed', label='Market SRU %.4f' % (np.mean(SRU_500_array)))

    # set x-axis label
    ax.set_xlabel(r'c')
    # set y-axis label
    ax.set_ylabel(title_str, color="black", fontsize=12)
    # plt.legend()
    if title_str in ['R-square']:
        plt.ylim([-0.5, 0.1])
    if title_str in ['Return']:
        plt.ylim([-0.2, 0.6])
    if title_str in ['Risk']:
        plt.ylim([0, 1])
    if title_str in ['MSE']:
        plt.ylim([1.2, 2.5])


    # plt.xlim([0, 1.1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.99])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4, prop={'size': 10})

    plt.title(title_str +'\n' +para_str_title)
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + title_str + para_str + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def series_plot(save_path, c_M, array, title_str, list_of_z, para_str_title, para_str, \
                SRU_500_array = None, show = False):
    '''
    This function plots the series of R-squared, SRU, returns, and risks
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''
    save_path = os.path.join(save_path, 'Unrefined')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    z_select_idx =range(1, len(list_of_z))
    for i in z_select_idx: # range(array.shape[1]):
        plt.plot(c_M, array[:, i], linestyle='solid', linewidth=1,
                      label='log10(z) = %s' % (np.log10(list_of_z[i])))
    plt.axvline(x=1, color='grey', linewidth=1, linestyle='dashed')
    if title_str == 'SRU' and SRU_500_array is not None:
        plt.axhline(y=np.mean(SRU_500_array), color='black', linewidth=1, linestyle='dashed', label='Market SRU %.4f' % (np.mean(SRU_500_array)))

    # set x-axis label
    ax.set_xlabel(r'c')
    # set y-axis label
    ax.set_ylabel(title_str, color="black", fontsize=12)
    # plt.legend()
    # plt.ylim([-0.3, 0.3])
    # plt.xlim([0, 1.1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.99])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True, ncol=4, prop={'size': 10})

    plt.title(title_str +'\n' +para_str_title)
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + title_str + para_str + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def series_plot_for_misspecified(save_path, c_M, array, title_str, list_of_z, para_str_title, para_str,show = False):
    '''
    This function plots the series of SRU and MSE for mis-specified model
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''
    # save_path = os.path.join(save_path, title_str)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(array.shape[1]):
        plt.plot(c_M, array[:, i], linestyle='solid', linewidth=1,
                      label='log10(z) = %s' % (np.log10(list_of_z[i])))
    plt.axvline(x=1, color='grey', linewidth=1, linestyle='dashed')

    # set x-axis label
    ax.set_xlabel(r'$c_1 = M_1/T$')
    # set y-axis label
    ax.set_ylabel(title_str, color="black", fontsize=12)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.99])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True, ncol=4, prop={'size': 10})

    plt.title(title_str +'\n' +para_str_title)
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + title_str + para_str + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def series_plot_for_misspecified_final(save_path, c_M, array, title_str, Rsq_true, list_of_z):
    '''
    This function plots the series of SRU and MSE for mis-specified model
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''

    line_width = 1.5
    fig, ax = plt.subplots()
    ax.plot(c_M, array[:, 0], linestyle='solid', linewidth=line_width, label='Ridgeless')
    for i in range(1, array.shape[1]):
        ax.plot(c_M, array[:, i], linewidth=line_width, label='z = %s' % (list_of_z[i]))
    plt.axvline(x=1, color='grey', linewidth=1, linestyle='dashed')

    # set x-axis label
    ax.set_xlabel(r'$c_1$')
    ax.set_ylabel(title_str, fontsize=12)

    # if title_str == 'R2':
    #     ax.set_ylabel(r'$R^2$', fontsize=12)
    #     if Rsq_true is not None:
    #         ax.plot(c_M, Rsq_true * np.ones(len(c_M)), color='red', linestyle='dotted', linewidth=line_width,
    #                 label='True')
    #     ax.legend(loc='upper right')
    #     plt.ylim([-0.3, 0.3])
    # else:
    #     ax.set_ylabel(title_str, fontsize=12)
    #     if title_str == 'Volatility':
    #         plt.ylim([0, 6])
    #     if title_str == 'MSE':
    #         # ax.legend(loc='upper right')
    #         plt.ylim([0.5, 6])

    if title_str == 'R2':
        ax.set_ylabel(r'$R^2$', fontsize=12)
        ax.legend(loc='upper right')
        plt.ylim([-0.3, 0.3])
    elif title_str == 'SR':
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
    elif title_str == 'ER':
        ax.set_ylabel('Expected Return', fontsize=12)
    elif title_str == 'Vol':
        ax.set_ylabel('Volatility', fontsize=12)
        plt.ylim([0, 6])
    elif title_str == 'MSE':
        ax.set_ylabel('MSE', fontsize=12)
        plt.ylim([0.5, 6])
    elif title_str == 'Bnorm':
        ax.set_ylabel(r'$\Vert \hat\beta \Vert ^2$', fontsize=12)
        plt.ylim([0, 6])

    fig.savefig(save_path + '/MisSpec' + title_str + '.eps', format='eps', bbox_inches='tight')
    fig.savefig(save_path + '/MisSpec' + title_str + '.png', bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')

def std_plot(save_path, c_M, array, title_str, list_of_z, para_str_title, para_str, show = False):
    '''
    This function plots the series of R-squared, SRU, returns, and risks
    :param save_path:
    :param c_M:
    :param array: len(c) \times len(z)
    :param title_str:
    :param list_of_z:
    :param para_str_title:
    :param para_str:
    :return:
    '''

    title_str = title_str + ' Std'
    # save_path = os.path.join(save_path, title_str)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    fig, ax = plt.subplots()
    for i in range(array.shape[1]):
        plt.plot(c_M, array[:, i], 'o', linestyle='solid', linewidth=1,
                      label='log10(z) = %s' % (np.log10(list_of_z[i])))
    plt.axvline(x=1, color='grey', linewidth=1, linestyle='dashed')

    # set x-axis label
    ax.set_xlabel(r'c')
    # set y-axis label
    ax.set_ylabel(title_str, color="black", fontsize=12)
    # plt.legend()
    # plt.ylim([-0.3, 0.3])
    # plt.xlim([0, 1.1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.90])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True, ncol=3)

    plt.title(title_str +'\n' +para_str_title)
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + title_str + para_str + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def boxplot(save_path, c_M, T, dist, title_str, para_str_title, para_str, outliers, z, show = False):
    '''
    Box plot to check distributions of array dist
    :param save_path:
    :param c_M:
    :param T:
    :param dist:
    :param title_str:
    :param para_str_title:
    :param para_str:
    :param outliers:
    :return:
    '''

    c_M_str = np.array(["%.2f" % round(number, 2) for number in c_M])
    c_M_str[c_M == 1] = '1'

    if outliers:
        title_str = title_str + ' with Outliers'
    else:
        title_str = title_str + ' without Outliers'

    save_path = os.path.join(save_path, title_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(dist.T_TRAIN, showfliers=outliers)
    # plt.axvline(x=T-1, color='grey', linewidth=1, linestyle='dashed')
    xticks_idx = list(range(0,len(c_M)-1,1)) + [len(c_M)-1]
    plt.xticks(xticks_idx, c_M_str[xticks_idx])

    plt.title(title_str + '\n'+ para_str_title+ '\n' + '_z %s'%(z) )
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + title_str + para_str + '_z %s'%(z) + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')


def scatterplot(save_path, c_M, Ret_dist1, Risk_dist1, para_str_title, para_str, z, show = False):
    '''
    Scatter plot to check the relationship between Ret_dist1 and Ret_dist2
    :param save_path:
    :param c_M:
    :param c_idx:
    :param Ret_dist1:
    :param Risk_dist1:
    :param para_str_title:
    :param para_str:
    :return:
    '''
    save_path = os.path.join(save_path, 'Return_sqrtRisk_Scatter')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, 'z %s'%(z))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # c_idx = 15
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(Ret_dist1, np.sqrt(Risk_dist1))

    plt.xlabel('Return')
    plt.ylabel('sqrt(Risk)')

    title_str = 'Scatter plot of Return vs. sqrt(Risk)'
    plt.title(title_str + '\n' + para_str_title + '\n' + 'c = %.2f'%(c_M)+ '\n' + 'z %s'%(z))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(save_path + '/' + title_str + para_str + '_c %.2f'%(c_M) + '_z %s'%(z) + '.png')
    if show:
        plt.show(block=False)
        plt.pause(1)
    plt.close('all')

