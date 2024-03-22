from scipy.io import loadmat
import numpy as np
import libKellyVoC as lvoc

# load the dataset
GYdata = loadmat('GYdata.mat')
X_, Y_, dates_ = GYdata['X'], GYdata['Y'], GYdata['dates']
trnwin_ = 120
maxP_ = 12000

# generate benchmark
benchmark_predictions, benchmark_predictions_rescaled, benchmark_square_weights = lvoc.benchmark_sim(
                                                                     X = X_,
                                                                     Y = Y_,
                                                                     trnwin = trnwin_,
                                                                     stdize = True,
                                                                     #rescale_predictions=False,
                                                                    )

np.save(f"benchmark.predictions.trnwin{trnwin_}", benchmark_predictions)
np.save(f"benchmark.predictions.rescaled.trnwin{trnwin_}", benchmark_predictions_rescaled)

#np.save("benchmark.square.weights", benchmark_square_weights)


# Initialize an empty list to store the results
prediction_arrays = []
prediction_arrays_rescaled = []
#square_weight_arrays = []

# each loop iteration takes about ~50+ seconds to complete
for simulation in range(1,200):
    print(f'[*] Starting new Simulation Run ({simulation})')
    # run simulations
    predictions_, predictions_rescaled,  square_weights_ = lvoc.Kelly_rff_ridge_sim( 
                                                          X = X_,
                                                          Y = Y_,
                                                          iSim = simulation, 
                                                          maxP = maxP_, 
                                                          gamma = 2,
                                                          trnwin = trnwin_,
                                                          stdize = True,
                                                          #rescale_predictions=False,
                                                        )
    prediction_arrays.append(predictions_)
    prediction_arrays_rescaled.append(predictions_rescaled)
   # square_weight_arrays.append(square_weights_)

    # Concatenate the results along the iSim axis (axis=3)
    concatenated_predictions = np.concatenate(prediction_arrays, axis=3)
    concatenated_predictions_rescaled = np.concatenate(prediction_arrays_rescaled, axis=3)

  #  concatenated_square_weights = np.concatenate(square_weight_arrays, axis=3)

    # save at each loop iteration
    np.save(f"simulated.predictions.trnwin{trnwin_}", concatenated_predictions)
    np.save(f"simulated.predictions.rescaled.trnwin{trnwin_}", concatenated_predictions_rescaled)
  #  np.save("simulated.square.weights.trnwin12", concatenated_square_weights)