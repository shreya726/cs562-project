import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# For testing performance 
def graph_performance():
	x = [20, 375, 1000, 8926]
	y = [11.406751, 138.981136, 387.523, 1645.713] #unsupervised 
	#y_fcm = [0,0,0,0]
	y_fcm = [0.2620726,3.056954,144.12319,17.347055]
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure()
	
	ax = fig.add_subplot(2, 1, 1)
	ax.set_yscale('log')
	
	
	plt.xlabel("Size of input data set")
	plt.ylabel("Runtime in seconds")
	
	unsupervised, = plt.plot(x, y, color='g', label='Unsupervised shapelets')

	# Predicted quadratic runtime of fuzzy c means
	n_squared, = plt.plot(x, [i**2 for i in x], color='orange', label=r"O($n^2$)")

	# Compare to predicted runtime

	fontP = FontProperties()
	fontP.set_size('small')

	fcm, = plt.plot(x, y_fcm, color='r', label='Fuzzy c-means')
	#plt.legend([unsupervised, fcm, n_squared], ['Unsupervised shapelets', 'Fuzzy c-means', r"O($n^2$)"], prop=fontP)
	#fig.show()

	

	plt.show()
graph_performance()