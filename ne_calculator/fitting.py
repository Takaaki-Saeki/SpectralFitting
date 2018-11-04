
"""
  fitting.py

  Created by Saeki Takaaki on 2018/11/2.
  Copyright © 2018年 Saeki Takaaki. All rights reserved.

"""

# 基本的なpythonモジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# lmfit.modelsから線形モデル、voigtモデルのインポート
from lmfit.models import VoigtModel
from lmfit.models import LinearModel


def input_calibrated_data():
    # data1~data10までを読み込み、強度校正した結果を返す関数
    
    data = []
    # データ番号とリストのインデックスを対応させるためにりindex:0にからのリストを入れてある
    # 本当はNoneとか入れて例外処理すべき
    data.append([])
    col = range(0, 44)
    for i in range(1, 11):
        d = pd.read_csv('data{}.csv'.format(i), header=None)
        d = d.dropna(how="all")
        d = d.dropna(how="all", axis=1)
        g = []
        g.append(d[0])
        for j in range(1, 44):
            d_array = np.array(d[j])
            c_array = np.array(coeff['coeff'])
            cd = d_array * c_array
            g.append(cd)
        g = pd.DataFrame(g).T
        g.columns = col
        data.append(g)
    return data


def max_index(data=None):
    # data1~data10について、ピーク値が最大となる列名をリストとして返す関数
    
    max_idx = []
    # データ番号とリストのインデックスを対応させるためにindex:0には-1を入れてある
    # 本当は例外処理すべき
    max_idx.append(-1)
    for i in range(1, 11):
        idx = data[i].max()[data[i].max() == data[i].max().max()].index[0]
        max_idx.append(idx)
    return max_idx


def data_cut(data=None, indx_list=None):
    # 各データを468.0~495.0までの波長帯だけ残してカットする関数
    
    data_cut_list = []
    idx_cut = []
    # データ番号とリストのインデックスを対応させるために、index:0にはからのリストを入れてある
    # 本当はNone入れて例外処理した方が良い
    data_cut_list.append([])
    idx_cut.append([])
    df_cut = []
    df_cut.append([])
    for i in range(1, 11):
        d = data[i].loc[(data[i][0] >= 468.0) & (data[i][0] <= 495), indx_list[i]]
        idx = data[i].loc[(data[i][0] >= 468.0) & (data[i][0] <= 495), 0]
        data_cut_list.append(np.array(d))
        idx_cut.append(np.array(idx))
        df_cut.append(pd.DataFrame({'w': idx_cut[i], 'i': data_cut_list[i]}))

    return df_cut


def voigt_fit(df_cut=None, data_num=None, sigma=0.15):
    # フォークト関数によるフィッティングを行う関数

    x = df_cut[data_num]['w']
    y = df_cut[data_num]['i']

    # 線形モデルを定義
    # パラメータオブジェクトparsの生成
    lin = LinearModel(prefix='lin_')
    pars = lin.guess(y, x=x)

    # 1つめのピーク
    voigt1 = VoigtModel(prefix='v1_')
    pars.update(voigt1.make_params())
    pars['v1_center'].set(473.3, min=470, max=475)
    pars['v1_sigma'].set(sigma, min=sigma, max=sigma + 0.0000000001)
    pars['v1_amplitude'].set(10000, min=1)
    pars['v1_gamma'].set(1, min=0.1, max=2, vary=True)

    # 2つめのピーク
    voigt2 = VoigtModel(prefix='v2_')
    pars.update(voigt2.make_params())
    pars['v2_center'].set(476.5, min=475, max=478)
    pars['v2_sigma'].set(sigma, min=sigma, max=sigma + 0.0000000001)
    pars['v2_amplitude'].set(10000, min=1)
    pars['v2_gamma'].set(1, min=0.1, max=2, vary=True)

    # 3つめのピーク
    voigt3 = VoigtModel(prefix='v3_')
    pars.update(voigt3.make_params())
    pars['v3_center'].set(480.6, min=476, max=483)
    pars['v3_sigma'].set(sigma, min=sigma, max=sigma + 0.0000000001)
    pars['v3_amplitude'].set(10000, min=1)
    pars['v3_gamma'].set(1, min=0.1, max=2, vary=True)

    # 4つめのピーク
    voigt4 = VoigtModel(prefix='v4_')
    pars.update(voigt4.make_params())
    pars['v4_center'].set(484.7, min=483, max=487)
    pars['v4_sigma'].set(sigma, min=sigma, max=sigma + 0.0000000001)
    pars['v4_amplitude'].set(10000, min=1)
    pars['v4_gamma'].set(1, min=0.1, max=2, vary=True)

    # 5つめのピーク
    voigt5 = VoigtModel(prefix='v5_')
    pars.update(voigt5.make_params())
    pars['v5_center'].set(487.8, min=485, max=490)
    pars['v5_sigma'].set(sigma, min=sigma, max=sigma + 0.0000000001)
    pars['v5_amplitude'].set(10000, min=1)
    pars['v5_gamma'].set(1, min=0.1, max=2, vary=True)

    # 6つめのピーク
    voigt6 = VoigtModel(prefix='v6_')
    pars.update(voigt6.make_params())
    pars['v6_center'].set(493.3, min=490, max=494)
    pars['v6_sigma'].set(sigma, min=sigma, max=sigma + 0.0000000001)
    pars['v6_amplitude'].set(10000, min=1)
    pars['v6_gamma'].set(1, min=0.1, max=2, vary=True)

    mod = voigt1 + voigt2 + voigt3 + voigt4 + voigt5 + voigt6 + lin

    # 初期値
    init = mod.eval(pars, x=x)

    # 最適値
    out = mod.fit(y, pars, x=x)
   
    # パラメータの情報などを表示
    print(out.fit_report(min_correl=0.5))
    # 一つ一つのvoigt関数を表示するかどうか
    plot_components = False

    # 　結果のプロット
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b')
    plt.plot(x, init, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity (a.u.)')

    if plot_components:
        comps = out.eval_components(x=x)
        plt.plot(x, comps['v1_'], 'b--')
        plt.plot(x, comps['v2_'], 'b--')
        plt.plot(x, comps['v3_'], 'b--')
        plt.plot(x, comps['v4_'], 'b--')
        plt.plot(x, comps['v5_'], 'b--')
        plt.plot(x, comps['v6_'], 'b--')
        plt.plot(x, comps['lin_'], 'k--')

        plt.show()

    plt.savefig('result{}.jpg'.format(data_num))
 

    return out.params['v3_gamma'].value


def calculate_ne(gamma_list, w):
    # data1~data10より得られたシュタルク広がりから、neの平均値、標準誤差を計算する関数
    # wにはテーブルの値をそのまま入れれば良い(単位変換不要)
    ne_array = np.array(gamma_list) / w * 10 ** (17) * 100 ** 3
    mu = np.mean(ne_array)
    std = np.std(ne_array)
    ste = std / np.sqrt(10)

    return mu, ste


if __name__ == '__main__':

    coeff = pd.read_csv('coeff.csv')
    
    data = input_calibrated_data()
    
    max_idx = max_index(data)

    mu_list = []
    ste_list = []
    data_num_list = []
    for k in range(-2, 11):
        data_num_list.append(k)
        
        idx_list = np.array(max_idx) + k
        idx_list = list(idx_list)

        d_cut = data_cut(data, idx_list)

        gamma_list = []
        for i in range(1, 11):
            gamma_list.append(voigt_fit(d_cut, i, 0.15))

        mu, ste = calculate_ne(gamma_list, 6.93*10**(-3))

        mu_list.append(mu)
        ste_list.append(ste)

    plt.errorbar(data_num_list, mu_list, yerr=ste_list, fmt='ro', ecolor='g')
    plt.title('position - ne')
    plt.show()
