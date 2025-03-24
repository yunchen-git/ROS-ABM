import Analysis as af
import Simulation as sim
import pickle as p

def getGradientImages(path):
    data=sim.TimeSeries(path)
    grad_names=data._grad_names

    for i in range(len(grad_names)):
        af.save_2D_gradient_images(grad_names[i],path,data,axis="xy")


if __name__=='__main__':
    path='/Users/huyunchen/Documents/PhD/test/34.0/'
    getGradientImages(path)


