if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from FlexibleWM import *
    import time
    from datetime import datetime
    import ipdb

    # Start the timer
    start_time = time.time()

    # Set the current time as 'name_simu' (e.g., formatted as Year-Month-Day_Hour-Minute-Second)
    time_simu = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Simulation start time: {time_simu}")

    # sim_name = 'trials/{time_simu}'
    sim_name = 'PlotFiringRate'

    folder_path = f'./FlexibleWM/{sim_name}'

    dictionnary={
        'name_simu':f'FlexibleWM/{sim_name}',
        'Number_of_trials':4,
        'specific_load':2, # number of items to remember. The specific location is random
        'compute_tuning_curve':False, # Reuse tuning curve
        'same_network_to_use':True,
        'plot_raster':False,
        'num_cores':1,
        # 'create_a_specific_network':True,

        } # Add here any parameter you want to change from default. Defaults values are at the beginning of FlexibleWM.py
    MyModel = FlexibleWM(dictionnary)
    gcPython.collect() 

    num_trials = MyModel.specF['Number_of_trials']

    Matrix_all_results = numpy.ones((MyModel.specF['Number_of_trials'],2))*numpy.nan
    Matrix_abs_all = numpy.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools']))*numpy.nan
    Matrix_angle_all = numpy.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools']))*numpy.nan
    Results_ml_spikes = numpy.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools']))*numpy.nan
    Drift_from_ml_spikes = numpy.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools']))*numpy.nan
    Matrix_initial_input = numpy.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools']))*numpy.nan

    if not (MyModel.specF['same_network_to_use'] and os.path.exists(MyModel.specF['path_for_same_network'])):
        MyModel.initialize_weights()

    if MyModel.specF['num_cores'] == 1:
        for index_simulation in tqdm(range(num_trials), desc="Trials"):
            result = MyModel.run_a_sim(index_simulation)

            Matrix_abs_all[index_simulation] = result[0]
            Matrix_angle_all[index_simulation] = result[1]
            Matrix_initial_input[index_simulation] = result[2]
            Results_ml_spikes[index_simulation] = result[3]
            Drift_from_ml_spikes[index_simulation] = result[4]
            Matrix_all_results[index_simulation] = result[5]

    elif MyModel.specF['num_cores'] > 1:
        with ProcessPoolExecutor(max_workers=MyModel.specF['num_cores']) as executor:
            futures = [executor.submit(MyModel.run_a_sim, index_simulation) for index_simulation in range(num_trials)]
            for future in tqdm(as_completed(futures), total=num_trials, desc="Trials"):
                # for future in as_completed(futures):
                result = future.result() 
                index_simulation = result[-1]
                # result = matrix_abs, matrix_angle, matrix_initial_input, results_ml_spikes, drift_from_ml_spikes, matrix_memory_results
                Matrix_abs_all[index_simulation] = result[0]
                Matrix_angle_all[index_simulation] = result[1]
                Matrix_initial_input[index_simulation] = result[2]
                Results_ml_spikes[index_simulation] = result[3]
                Drift_from_ml_spikes[index_simulation] = result[4]
                Matrix_all_results[index_simulation] = result[5]

    else:
        error

    # saving  
    numpy.savez_compressed(MyModel.specF['path_sim'], Matrix_all_results=Matrix_all_results,Matrix_abs_all=Matrix_abs_all, Matrix_angle_all=Matrix_angle_all, Results_ml_spikes=Results_ml_spikes, Drift_from_ml_spikes=Drift_from_ml_spikes,Matrix_initial_input = Matrix_initial_input)
    print("All results saved in the folder")
    # gcPython.collect()

    # End the timer
    end_time = time.time()

    # Calculate the total running time
    total_time = end_time - start_time
    print(f"Total running time: {total_time:.2f} seconds")


    # Load the .npz file
    npzfile = np.load(f'{folder_path}/simulation_results.npz')

    # List all the arrays stored in the .npz file
    print("Array names:", npzfile.files)

    # # Access individual arrays by their names
    Matrix_all_results = npzfile['Matrix_all_results']
    Matrix_abs_all = npzfile['Matrix_abs_all']
    Matrix_angle_all = npzfile['Matrix_angle_all']
    Results_ml_spikes = npzfile['Results_ml_spikes']
    Drift_from_ml_spikes = npzfile['Drift_from_ml_spikes']
    Matrix_initial_input = npzfile['Matrix_initial_input']

    load = dictionnary['specific_load']
    acc = np.sum(Matrix_all_results[:,0]/Matrix_all_results.shape[0])*100
    print()
    print(f'load_number = {load}, Percent Maintained: {acc:.1f}%')
    print()
    # plt.imshow(Matrix_all_results)
    # Example: print one of the arrays
    # print(array1)

    # # Close the file after accessing the arrays
    npzfile.close()