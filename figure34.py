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
        'Number_of_trials':60,
        'specific_load':7, # number of items to remember. The specific location is random
        'compute_tuning_curve':False, # Reuse tuning curve
        'same_network_to_use':True,
        'plot_raster':False,
        'num_cores':6,
        # 'create_a_specific_network':True,

        } # Add here any parameter you want to change from default. Defaults values are at the beginning of FlexibleWM.py
    MyModel = FlexibleWM(dictionnary)
    gcPython.collect() 

    num_trials = MyModel.specF['Number_of_trials'] #MyModel.specF['simtime']
    time_steps = np.arange(0, 1.1, MyModel.specF['window_save_data'])
    print('time steps: ', time_steps)
    time_steps_num = time_steps.shape[0]

    Matrix_abs_all = np.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools'],time_steps_num))*np.nan
    Matrix_angle_all = np.ones((MyModel.specF['Number_of_trials'],MyModel.specF['N_sensory_pools'],time_steps_num))*np.nan
    Matrix_proba_maintained = np.ones((MyModel.specF['Number_of_trials'],time_steps_num))*np.nan
    Matrix_proba_spurious = np.ones((MyModel.specF['Number_of_trials'],time_steps_num))*np.nan

    if not (MyModel.specF['same_network_to_use'] and os.path.exists(MyModel.specF['path_for_same_network'])):
        MyModel.initialize_weights()

    if MyModel.specF['num_cores'] == 1:
        for index_simulation in tqdm(range(num_trials), desc="Trials"):
            result = MyModel.run_a_sim(index_simulation)

            Matrix_abs_all[index_simulation] = result[0]
            Matrix_angle_all[index_simulation] = result[1]
            Matrix_proba_maintained[index_simulation] = result[2]
            Matrix_proba_spurious[index_simulation] = result[3]


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
                Matrix_proba_maintained[index_simulation] = result[2]
                Matrix_proba_spurious[index_simulation] = result[3]
                

    else:
        error

    # saving  
    np.savez_compressed(MyModel.specF['path_sim'], Matrix_abs_all=Matrix_abs_all,Matrix_angle_all=Matrix_angle_all,Matrix_proba_maintained=Matrix_proba_maintained,Matrix_proba_spurious=Matrix_proba_spurious)
    print("All results saved in the folder")
    # gcPython.collect()

    # End the timer
    end_time = time.time()

    # Calculate the total running time
    total_time = end_time - start_time
    print(f"Total running time: {total_time:.2f} seconds")


    load = dictionnary['specific_load']
    print('Matrix_proba_maintained',Matrix_proba_maintained.shape)
    acc = np.sum(Matrix_proba_maintained/Matrix_proba_maintained.shape[0],axis=0)*100
    print('acc:',acc.shape)
    acc[0:2] = 100
    plt.figure()
    plt.plot(time_steps,acc)
    plt.axvspan(0, 0.1, color='yellow', alpha=0.3, label="stimuli on")
    plt.ylim([0,110])
    plt.title('Forgetting during the delay period')
    plt.xlabel('Time from stinuli onset')
    plt.ylabel('Memory accuracy (%)')

    plt.show()

    # figure()
    # subplot(211)
    # plot_raster(S_rcn.i, S_rcn.t, time_unit=second, marker=',', color='k')
    # ylabel('Random neurons')
    # subplot(212)
    # plot_raster(S_rn.i, S_rn.t, time_unit=second, marker=',', color='k')
    # for index_sn in range(self.specF['N_sensory_pools']+1) :
    #     plot(np.arange(0,self.specF['simtime']+self.specF['window_save_data']/2.,self.specF['window_save_data']),index_sn*self.specF['N_sensory']*np.ones(int(round(self.specF['simtime']/self.specF['window_save_data']))+1),color='b', linewidth=0.5)
    # ylabel('Sensory neurons')
    # savefig(self.specF['path_to_save']+'rasterplot_trial'+str(index_simulation)+'.png')
    # close()
    # print(f'load_number = {load}, Percent Maintained: {acc:.1f}%')
    # print()
    # plt.imshow(Matrix_all_results)
    # Example: print one of the arrays
    # print(array1)

    # # Close the file after accessing the arrays
    # npzfile.close()