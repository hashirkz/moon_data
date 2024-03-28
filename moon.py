import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import math as m
from datetime import datetime
from glob import glob
import os

rootp = lambda p: os.path.join(os.path.realpath(os.path.dirname(__file__)), p)

# MOON_PHASE = {
#	  'nm': 'new moon',
#	  'xc': 'waxing crescent',
#	  'fq': 'first quarter',
#	  'xg': 'waxing gibbous',
#	  'fm': 'full moon',
#	  'wg': 'wanning gibbous',
#	  'tq': 'third quarter',
#	  'wc': 'wanning crescent',
# }

# read img into np m,n,3 tensor
def read_img(p: str) -> np.ndarray:
	img = plt.imread(p)
	return img

# grayscale m,n,3 img to m,n
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	return 0.2989*r + 0.5870*g + 0.1140*b

# normalize img px values to 0-1
def normalize(img: np.ndarray) -> np.ndarray:
	minn, maxx = np.min(img), np.max(img)
	return (img - minn) / (maxx - minn)	

# fit general sin function to data points x, y
# a * sin(x + c) + d
# 1 period so b = 2pi / p
def fit_sin(x: np.ndarray, y: np.ndarray) -> tuple:
	ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
	Fyy = abs(np.fft.fft(y))
	guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
	guess_amp = np.std(y) * 2.**0.5
	guess_offset = np.mean(y)
	guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
	# guess = np.array([guess_amp, 0., guess_offset])
	
	# b = 2*m.pi / (29.5)
	sinf = lambda x, a, b, c, d: a*np.sin(b*x+c)+d
	popt, _ = curve_fit(sinf, x, y, p0=guess)
	return popt

# find lum of img *defined as lit up pxs / total num pxs
def luminance(img: np.ndarray, tol: float=0.8) -> float:
	m, n = img.shape
	return np.count_nonzero(img > tol) / (m*n)
	
# mapping dates phases luminances from p
def all_lum(p: str, tol: float=0.8) -> tuple:
	dates, phases, lums = [], [], []

	for fp in glob(os.path.join(p, '*.jpg')):
		# read img -> grayscale -> normalize 0-1 -> find lum
		img = read_img(fp)
		img = rgb_to_gray(img)
		img = normalize(img)
		lum = luminance(img, tol=tol)
		lums.append(lum)
		
		bname = os.path.basename(fp).split('.')[0]
		date, phase = bname[:bname.rfind('_')], bname[bname.rfind('_')+1:]
		date = datetime.strptime(date, '%y_%b_%d_%H_%M').date()

		dates.append(date)
		phases.append(phase)
		
	return np.array(dates), np.array(phases), np.array(lums)

def scatter_moon_data(moon_data: pd.DataFrame, savep: str='moon_plot2.jpg') -> None:
	xs = moon_data['num_days'].to_numpy()

	a, b, c, d = fit_sin(xs, moon_data['lum_mean'].to_numpy())
	print(f'estimated_period: {2*m.pi/b}d\n')

	sinf = lambda x : a * np.sin(b*x + c) + d

	xs = np.linspace(xs[0], xs[-1], 100)
	ys = sinf(xs)

	# plotting lum mean
	plt.figure(figsize=(20, 12))
	plt.scatter(
		moon_data['date'].to_numpy(), 
		moon_data['lum_mean'].to_numpy(),
		marker='x',
		color='purple'
	)

	# sin trend line
	plt.plot (
		xs,
		ys,
		color='blue',
		alpha=0.5,
		label=f'{a}sin({b}x+{c})+{d}'
	)

	# std error bars 
	plt.errorbar(
		moon_data['date'].to_numpy(), 
		moon_data['lum_mean'].to_numpy(), 
		yerr=moon_data['lum_std'].to_numpy(), 
		fmt='none', 
		color='blue', 
		capsize=5
	)
	
	# axis labels and grid
	plt.xlabel('date yy-mm-dd', labelpad=40)
	plt.ylabel('mean 0-1 luminance [unit]', labelpad=40)
	plt.grid(True)
	plt.legend()
	plt.savefig(savep)

if __name__ == '__main__':
	
	tol = input("lum ? ")
	tol = 0.8 if not tol else float(tol)
	dates, phases, lums = all_lum('./crop', tol=tol)
	
	moon_data = pd.DataFrame({
		'date': dates,
		'phase': phases,
		'lum': lums
	})

	moon_data['date'] = pd.to_datetime(moon_data['date'])
	moon_data = moon_data.groupby(by='date').agg({
		'phase': 'first',
		'lum': ['mean', 'std']
	})

	moon_data.columns = moon_data.columns.droplevel()
	moon_data = moon_data.rename(columns={'first': 'phase', 'mean': 'lum_mean', 'std': 'lum_std'})
	moon_data['lum_std'] = moon_data['lum_std'].fillna(0)
	moon_data = moon_data.reset_index()
	moon_data['num_days'] = (moon_data['date'] - pd.to_datetime('1970-01-01')).dt.days

	scatter_moon_data(moon_data)
	cols = ['date','phase','lum_mean','lum_std']
	print(moon_data[cols])

	moon_data[cols].to_csv('./moon.csv', index=False)


