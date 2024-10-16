# Bouchacourt and Buschman 2019
# A Flexible Model of Working Memory
# This is a simplified version of the code. Specific analyses of the paper can be pushed on demand. 


import pickle, numpy, scipy, pylab, os
import pdb
import scipy.stats
import math 
import sys
import os.path
import logging 
from brian2 import *
BrianLogger.log_level_error()
from brian2tools import *
import cython
prefs.codegen.target = 'cython' 
import gc as gcPython
import ipdb
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import logging
logging.getLogger().setLevel(logging.WARNING)  # Set logging level to WARNING or ERROR to mute INFO messages

class FlexibleWM:
  """Class running one trial of the network. We give it a dictionary including parameters not taking default value, as well as the name of the folder for saving."""
  def __init__(self,spec):
    self.specF={} # empty dictionnary
    self.specF['name_simu'] = spec.get('name_simu','FlexibleWM') # Name of the folder to save results and eventual raster plots
    self.specF['Number_of_trials'] = spec.get('Number_of_trials',1) # increase in order to run multiple trials with the same network
    self.specF['num_cores'] = spec.get('num_cores',1) # num of cores to use in find_tuning_curve

    # Timing
    self.specF['clock'] = spec.get('clock',0.1) # Integration with time-step in msec
    self.specF['simtime'] = spec.get('simtime',1.1) # Simulation time in sec
    self.specF['RecSynapseTau'] = spec.get('RecSynapseTau',0.010) # Synaptic time constant
    self.specF['window_save_data'] = spec.get('window_save_data',0.1)   # the time step for saving data like rates, it will also be the by default timestep for saving spikes
    self.specF['num_stimuli_gird'] = spec.get('num_stimuli_gird',10 )
    
    if self.specF['window_save_data']!=0.1 or self.specF['simtime']!=1.1 :
      print(" ----------- WARNING : Make sure simtime divided by window_save_data is an integer ----------- ")

    # Network architecture
    self.specF['N_sensory'] = spec.get('N_sensory',512) # Number of recurrent neurons per SN
    self.specF['N_sensory_pools'] = spec.get('N_sensory_pools',8)  # Number of ring-like SN
    self.specF['N_random'] = spec.get('N_random',1024) # Number of neuron in the random network
    self.specF['self_excitation'] = spec.get('self_excitation',False) # if you want to run the SN by itself, the self excitation has to be True
    self.specF['with_random_network'] = spec.get('with_random_network',True) # with the RN, self excitation is False, so that a SN by itself cannot maintain a memory
    self.specF['fI_slope'] = spec.get('fI_slope',0.4)  # slope of the non-linear f-I function

    # Parameters describing recurrence within the SN
    self.specF['RecWNegativeWidth']= spec.get('RecWNegativeWidth',0.25);   # width of negative surround suppression
    self.specF['RecWPositiveWidth'] = spec.get('RecWPositiveWidth',1);    # width of positive amplification
    self.specF['RecWAmp_exc'] = spec.get('RecWAmp_exc',2) # amplitude of weight function
    self.specF['RecWAmp_inh'] = spec.get('RecWAmp_inh',2) # amplitude of weight function
    self.specF['RecWBaseline'] = spec.get('RecWBaseline',0.28)    # baseline of weight matrix for recurrent network

    # Parameters describing the weights between SN and RN
    self.specF['eibalance_ff'] = spec.get('eibalance_ff',-1.) # if -1, perfect feed-forward balance from SN to RN
    self.specF['eibalance_fb'] = spec.get('eibalance_fb',-1.) # if -1, perfect feedback balance from RN to SN
    self.specF['RecToRndW_TargetFR'] = spec.get('RecToRndW_TargetFR',2.1) # parameter (alpha in the paper) used to compute the feedforward weight, before balancing
    self.specF['RndToRecW_TargetFR'] = spec.get('RndToRecW_TargetFR',0.2) # parameter (beta in the paper) used to compute the feedback weight, before balancing
    self.specF['RndRec_f'] = spec.get('RndRec_f',0.35) # connectivity (gamma in the paper)
    self.specF['factor'] = spec.get('factor',1000) # factor for computing weights values (see Methods of the paper)

    # Saving/Using or not a pre-saved network
    self.specF['same_network_to_use'] = spec.get('same_network_to_use',False) # if we want to initialise the network with weights previously saved. If no previous network, create one and save it.
    self.specF['path_for_same_network'] = spec.get('path_for_same_network',self.specF['name_simu']+'/network.npz') # path for the weights

    # Stimulation
    self.specF['specific_load'] = spec.get('specific_load',False) # whether to use a specific load for all trials, or having it random
    self.specF['number_of_specific_load'] = spec.get('number_of_specific_load',None) # number of the load. random number of load if == None
    self.specF['start_stimulation'] = spec.get('start_stimulation',0.1)
    self.specF['end_stimulation'] = spec.get('end_stimulation',0.2)
    self.specF['input_strength'] = spec.get('input_strength',10) # strength of the stimulation
    self.specF['N_sensory_inputwidth'] = spec.get('N_sensory_inputwidth',32)
    self.specF['InputWidthFactor'] = spec.get('InputWidthFactor',3)
    self.specF['InputWidth'] = round(self.specF['N_sensory']/float(self.specF['N_sensory_inputwidth']))  # the width for input stimulation of the gaussian distribution

    # ML decoding
    self.specF['decode_spikes_timestep'] = spec.get('decode_spikes_timestep',0.1) # decoding window
    self.specF['path_load_matrix_tuning_sensory'] = spec.get('path_load_matrix_tuning_sensory',self.specF['name_simu']+'/Matrix_tuning.npz') # path for tuning curve matrix
    self.specF['compute_tuning_curve'] = spec.get('compute_tuning_curve',True)

    # Define path_to_save and eventual raster plot
    self.specF['plot_raster'] = spec.get('plot_raster',True)   # plot a raster
    self.specF['path_to_save'] = self.define_path_to_save_results_from_the_trial()
    self.specF['path_sim'] = self.specF['path_to_save']+'simulation_results.npz'
    self.specF['path_psth'] = self.specF['path_to_save']+'simulation_psth.npz'



  def define_path_to_save_results_from_the_trial(self) :
    path_to_save = self.specF['name_simu']+'/'
    if not os.path.exists(path_to_save):
      try : 
        os.makedirs(path_to_save)
      except OSError:
        pass
    return path_to_save


  def apply_matFuncMask(self, m, target, mask, axis_ziou): 
    for i in range(m.shape[0]) :
      for j in range(m.shape[1]) :
        if mask[i,j]:
          if axis_ziou :
            m[i,j]= float(self.specF['factor'])*target/numpy.sum(mask[:,j])
          else :
            m[i,j]= float(self.specF['factor'])*target/numpy.sum(mask[i,:])
    return m


  def compute_activity_vector(self,R_rn_timed) :
    neuron_angs = numpy.arange(1,self.specF['N_sensory']+1,1)/float(self.specF['N_sensory'])*2*math.pi
    exp_neuron_angs = numpy.exp(1j*neuron_angs,dtype=complex)
    Matrix_abs_timed = numpy.zeros(self.specF['N_sensory_pools'])
    Matrix_angle_timed = numpy.zeros(self.specF['N_sensory_pools'])
    R_rn_2 = numpy.ones(R_rn_timed.shape,dtype=complex)
    for index_pool in range(self.specF['N_sensory_pools']) :
      R_rn_2[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']] = numpy.multiply(R_rn_timed[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']], exp_neuron_angs, dtype=complex)
      R_rn_3 = numpy.mean(R_rn_2[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']])
      Matrix_angle_timed[index_pool] = numpy.angle(R_rn_3)*self.specF['N_sensory']/(2*math.pi)
      Matrix_abs_timed[index_pool] = numpy.absolute(R_rn_3)
    Matrix_angle_timed[Matrix_angle_timed<0]+=self.specF['N_sensory']
    return Matrix_abs_timed, Matrix_angle_timed


  def compute_drift(self,Angle_output,Angle_input):
      Angle_output_rad = Angle_output*2*math.pi/float(self.specF['N_sensory'])
      Angle_input_rad = Angle_input*2*math.pi/float(self.specF['N_sensory'])
      difference_angle = ((Angle_output_rad-Angle_input_rad+math.pi)%(2*math.pi))-math.pi
      difference_complex = numpy.exp(1j*difference_angle,dtype=complex)
      difference_angle2 = numpy.angle(difference_complex)
      return difference_angle2


  def ml_decode(self,number_of_spikes_rn,Matrix_tc) :
    # Input: psth_rn of the specfic pool; tuning matrix (standard)
    # Returns: index of the most possible stimuli (i.e. remembered stiluli)
    Matrix_likelihood_per_stim = numpy.zeros(self.specF['N_sensory'])
    for index_stim in range(self.specF['N_sensory']) : # loop over all possible stimuli
      Matrix_likelihood_per_stim[index_stim] = numpy.dot(number_of_spikes_rn,numpy.log(Matrix_tc[:,index_stim]))
      # ipdb.set_trace()
    S_ml = numpy.argmax(Matrix_likelihood_per_stim)
    if isinstance(S_ml, numpy.ndarray) :
      pdb.set_trace()
    return S_ml 


  def give_input_stimulus(self,current_InputCenter) :
    inp_vect = numpy.zeros(self.specF['N_sensory'])
    inp_ind = numpy.arange(int(round(current_InputCenter-self.specF['InputWidthFactor']*self.specF['InputWidth'])),int(round(current_InputCenter+self.specF['InputWidthFactor']*self.specF['InputWidth']+1)),1)
    inp_scale = scipy.stats.norm.pdf(inp_ind-current_InputCenter,0,self.specF['InputWidth'])
    inp_scale/=float(numpy.amax(inp_scale))
    inp_ind = numpy.remainder(inp_ind-1, self.specF['N_sensory']) 
    inp_vect[inp_ind] = self.specF['input_strength']*inp_scale
    inp_vect[:] = inp_vect[:]-numpy.sum(inp_vect[:])/float(inp_vect[:].shape[0])
    return inp_vect

  def compute_matrix_tuning_sensory(self) :
    Matrix_tuning = numpy.zeros((self.specF['N_sensory'], self.specF['N_sensory'])) # number of neurons in each pool, number of possible stimuli
    for index_stimulus in range(self.specF['N_sensory']) :
      Vector_stim = self.give_input_stimulus(index_stimulus)
      for index_sensoryneuron in range(self.specF['N_sensory']) :
        S_ext = Vector_stim[index_sensoryneuron]
        Matrix_tuning[index_sensoryneuron,index_stimulus] = 0.4*(1+math.tanh(self.specF['fI_slope']*S_ext-3))/self.specF['RecSynapseTau']
    numpy.savez_compressed(self.specF['path_load_matrix_tuning_sensory'],Matrix_tuning=Matrix_tuning)
    return Matrix_tuning

  def compute_psth_for_mldecoding(self,time_matrix,spike_matrix,End_of_delay,timestep) :
    time_length = int(round(End_of_delay/timestep))
    Matrix = numpy.zeros((self.specF['N_sensory_pools']*self.specF['N_sensory'],time_length))
    for index_tab in range(time_matrix.shape[0]) :
      if time_matrix[index_tab]<End_of_delay :
        Time_integer = int(floor(time_matrix[index_tab]*1/timestep))
        Matrix[spike_matrix[index_tab],Time_integer]+=1
    return Matrix

  def compute_psth_for_rcn(self,time_matrix,spike_matrix,End_of_delay,timestep) : #random connected network
    time_length = int(round(End_of_delay/timestep))
    Matrix = numpy.zeros((self.specF['N_random'],time_length))
    for index_tab in range(time_matrix.shape[0]) :
      if time_matrix[index_tab]<End_of_delay :
        Time_integer = int(floor(time_matrix[index_tab]*1/timestep))
        Matrix[spike_matrix[index_tab],Time_integer]+=1
    return Matrix
  

  def import_weights(self):
      weight_data = numpy.load(self.specF['path_for_same_network'])
      Intermed_matrix_rn_rn2 = weight_data['Intermed_matrix_rn_rn2']
      Intermed_matrix_rn_to_rcn2 = weight_data['Intermed_matrix_rn_to_rcn2']
      Intermed_matrix_rcn_to_rn2 = weight_data['Intermed_matrix_rcn_to_rn2']
      weight_data.close()
      return Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2
  

  def initialize_weights(self, index_simulation=None) :
    # if index_simulation == None:
    #   print(f'Initializing the weights of the network that will be used for all trials')
    # else:
    #   print(f'Initializing the weights of the network that will be used for trial {index_simulation}')
    
    numpy.random.seed()

    RecToRndW_EIBalance = self.specF['eibalance_ff'];
    RecToRndW_Baseline = 0; #baseline weight from recurrent network to random network, regardless of connection existing
    RndToRecW_EIBalance = self.specF['eibalance_fb'];
    RndToRecW_Baseline = 0;  # baseline weight from random network to recurrent network, regardless of connection existing
    RndWBaseline = 0;  # Baseline inhibition between neurons in random network
    RndWSelf = 0; # Self-excitation in random network
    PoolWBaseline = 0; # baseline weight between recurrent pools (scaled for # of neurons)
    PoolWRandom = 0; # +/- range of random weights between recurrent pools (scaled for # of neurons)
    # Connection matrix for RN to RN
    Angle = 2.*math.pi*numpy.arange(1,self.specF['N_sensory']+1)/float(self.specF['N_sensory'])
    def weight_intrapool(i) :
      return self.specF['RecWBaseline'] + self.specF['RecWAmp_exc']*exp(self.specF['RecWPositiveWidth']*(cos(i)-1)) - self.specF['RecWAmp_inh']*exp(self.specF['RecWNegativeWidth']*(cos(i)-1)) 

    Matrix_weight_intrapool = numpy.zeros((self.specF['N_sensory'],self.specF['N_sensory']))
    for index1 in range(self.specF['N_sensory']) :
      for index2 in range(self.specF['N_sensory']) :
        if index1 == index2 and self.specF['self_excitation']==False :
          Matrix_weight_intrapool[index1,index2] = 0
        else :
          Matrix_weight_intrapool[index1,index2] = weight_intrapool(Angle[index1]-Angle[index2])

    Intermed_matrix_rn_rn = (PoolWBaseline + PoolWRandom*2*(numpy.random.rand(self.specF['N_sensory_pools']*self.specF['N_sensory'], self.specF['N_sensory_pools']*self.specF['N_sensory']) - 0.5))/(self.specF['N_sensory']*(self.specF['N_sensory_pools']-1))
    for index_pool in range(self.specF['N_sensory_pools']) :
      Intermed_matrix_rn_rn[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory'],index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']]=Matrix_weight_intrapool[:,:]

    Intermed_matrix_rn_rn2 = Intermed_matrix_rn_rn.flatten()  #http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html#weight-matrices

    if self.specF['with_random_network'] :
      Matrix_Sym = numpy.random.rand(self.specF['N_sensory']*self.specF['N_sensory_pools'],self.specF['N_random'])
      Matrix_Sym = Matrix_Sym < self.specF['RndRec_f']

      Matrix_co_rn_to_rcn = RecToRndW_Baseline*numpy.ones((self.specF['N_sensory_pools']*self.specF['N_sensory'],self.specF['N_random']))
      Intermed_matrix_rn_to_rcn = self.apply_matFuncMask(Matrix_co_rn_to_rcn, self.specF['RecToRndW_TargetFR'] , Matrix_Sym,True)
      

      Matrix_co_rcn_to_rn = RndToRecW_Baseline*numpy.ones((self.specF['N_random'],self.specF['N_sensory_pools']*self.specF['N_sensory']))
      Intermed_matrix_rcn_to_rn = self.apply_matFuncMask(Matrix_co_rcn_to_rn.T, self.specF['RndToRecW_TargetFR'] , Matrix_Sym, False).T

      # Balance and then flatten
      for index_control_neuron in range(self.specF['N_random']) :
        Sum_of_excitation = numpy.sum(Intermed_matrix_rn_to_rcn[:,index_control_neuron])
        Intermed_matrix_rn_to_rcn[:,index_control_neuron]+=self.specF['eibalance_ff']*Sum_of_excitation/float(self.specF['N_sensory']*self.specF['N_sensory_pools'])
      Intermed_matrix_rn_to_rcn2 = Intermed_matrix_rn_to_rcn.flatten()   

      for index_neuron in range(self.specF['N_sensory']*self.specF['N_sensory_pools']) :
        Sum_of_excitation = numpy.sum(Intermed_matrix_rcn_to_rn[:,index_neuron])   # somme sur les 1024 control neurons
        Intermed_matrix_rcn_to_rn[:,index_neuron]+=self.specF['eibalance_fb']*Sum_of_excitation/float(self.specF['N_random'])
      Intermed_matrix_rcn_to_rn2 = Intermed_matrix_rcn_to_rn.flatten()

    else :
      Intermed_matrix_rn_to_rcn = numpy.zeros((self.specF['N_sensory']*self.specF['N_sensory_pools'],self.specF['N_random']))
      Intermed_matrix_rn_to_rcn2 = Intermed_matrix_rn_to_rcn.flatten()  
      Intermed_matrix_rcn_to_rn = numpy.zeros((self.specF['N_random'],self.specF['N_sensory_pools']*self.specF['N_sensory'])) 
      Intermed_matrix_rcn_to_rn2 = Intermed_matrix_rcn_to_rn.flatten()

    # creating a specific network, saving, and plotting the weights
    numpy.savez_compressed(self.specF['path_for_same_network'], Intermed_matrix_rn_rn2=Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2=Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2=Intermed_matrix_rcn_to_rn2)
    print("Network is saved in the folder, next time you can include 'same_network_to_use':True to reuse it")
    return 

  def run_a_trial(self, index_simulation, stimuli_grid):

    numpy.random.seed()
    # gcPython.enable()
    # --------------------------------- Network parameters ---------------------------------------------------------
    # Setting the simulation timestep
    defaultclock.dt = self.specF['clock']*ms     # this will be used by all objects that do not explicitly specify a clock or dt value during construction

    # Setting the simulation time
    Simtime = self.specF['simtime']*second
    fI_slope = self.specF['fI_slope']

    # Parameters of the neurons
    bias = 0  # bias in the firing response (cf page 1 right column of Burak, Fiete 2012)

    # Parameters of the synapses
    RecSynapseTau = self.specF['RecSynapseTau']*second
    RndSynapseTau = self.specF['RecSynapseTau']*second
    InitSynapseRange = 0.01    # Range to randomly initialize synaptic variables

    # --------------------------------- Network setting and equations ---------------------------------------------------------
    Intermed_matrix_rn_rn2, Intermed_matrix_rn_to_rcn2, Intermed_matrix_rcn_to_rn2 = self.import_weights()

    # Equations
    eqs_rec = '''
    dS_rec/dt = -S_rec/RecSynapseTau      :1
    S_ext : 1
    G_rec = S_rec + bias + S_ext  :1
    rate_rec = 0.4*(1+tanh(fI_slope*G_rec-3))/RecSynapseTau     :Hz
    '''

    eqs_rcn = '''
    dS_rnd/dt = -S_rnd/RndSynapseTau      :1
    S_ext_rnd : 1
    G_rnd = S_rnd + bias + S_ext_rnd  :1 
    rate_rnd = 0.4*(1+tanh(fI_slope*G_rnd-3))/RndSynapseTau  :Hz
    '''

    # Creation of the network
    Recurrent_Pools = NeuronGroup(self.specF['N_sensory']*self.specF['N_sensory_pools'], eqs_rec, threshold='rand()<rate_rec*dt')
    Recurrent_Pools.S_rec = 'InitSynapseRange*rand()'   # What this does is initialise each neuron with a different uniform random value between 0 and InitSynapseRange

    RCN_pool = NeuronGroup(self.specF['N_random'], eqs_rcn, threshold='rand()<rate_rnd*dt')
    RCN_pool.S_rnd = 'InitSynapseRange*rand()'

    # Building the recurrent connections within pools
    Rec_RN_RN = Synapses(Recurrent_Pools, Recurrent_Pools, model='w : 1', on_pre='S_rec+=w')  # Defining the synaptic model, w is the synapse-specific weight
    Rec_RN_RN.connect()  # connect all to all

    Rec_RN_RN.w = Intermed_matrix_rn_rn2

    if self.specF['with_random_network'] :
      # Building the symmetric recurrent connections from RN to RCN and from RCN to RN 
      Rec_RCN_RN = Synapses(RCN_pool, Recurrent_Pools, model='w : 1', on_pre='S_rec+=w')
      Rec_RCN_RN.connect()  # connect all to all

      Rec_RCN_RN.w = Intermed_matrix_rcn_to_rn2

      Rec_RN_RCN = Synapses(Recurrent_Pools, RCN_pool, model='w : 1', on_pre='S_rnd+=w')
      Rec_RN_RCN.connect()   # connect all to all

      Rec_RN_RCN.w = Intermed_matrix_rn_to_rcn2

    # R_rn = StateMonitor(Recurrent_Pools, 'rate_rec', record=True, dt=self.specF['window_save_data']*second)

    if self.specF['plot_raster'] :
      S_rn = SpikeMonitor(Recurrent_Pools)
      if self.specF['with_random_network'] :
        S_rcn = SpikeMonitor(RCN_pool)

    
    # STORE THE NETWORK, in case we run a large number of simulations with the same network
    store(f'initialized_{index_simulation}')

    # We build the baseline, random input (set to 0 for now)
    InputBaseline = 0; # strength of random inputs
    inp_baseline = numpy.zeros((self.specF['N_sensory_pools'],self.specF['N_sensory']))
    for index_pool in range(self.specF['N_sensory_pools']) :
      inp_baseline[index_pool,:] = InputBaseline*numpy.random.rand(self.specF['N_sensory'])

    inp_baseline_rnd = numpy.zeros(self.specF['N_random'])

    # The final result of this trial
    if self.specF['number_of_specific_load'] == None :
      load = numpy.random.randint(low=1,high=self.specF['N_sensory_pools']+1)
    else:
      load = self.specF['number_of_specific_load']

    shape_rn_trial = (len(stimuli_grid),) * load + (load,) + (self.specF['N_sensory'],)
    shape_rcn_trial = (len(stimuli_grid),) * load + (self.specF['N_random'],)
    psth_rn_trial = numpy.zeros(shape_rn_trial) # spikes count in 100ms. (stimuli1,stimuli2,...,loads, neurons in each ring network)
    psth_rcn_trial = numpy.zeros(shape_rcn_trial) # spikes count in 100ms. (stimuli1,stimuli2, neurons in the random network)

    Matrix_pools_receiving_inputs = []

    Matrix_pools = numpy.arange(self.specF['N_sensory_pools'])
    numpy.random.shuffle(Matrix_pools)
    Matrix_pools_receiving_inputs = Matrix_pools[:load]
    
    for index_combination in tqdm(itertools.product(range(len(stimuli_grid)), repeat=load),total=len(stimuli_grid)**load, desc="Inner Loop (stimuli)"):
      # ipdb.set_trace()

      restore(f'initialized_{index_simulation}')  # restore the initial network state for each stimuli
      numpy.random.seed()  # Set random seed for each process
      gcPython.enable()
      # Inputs
      # index_stimuli = [stimuli_grid[idx] for idx in index_combination]

      IT1 = self.specF['start_stimulation'] * second
      IT2 = self.specF['end_stimulation'] * second

      InputCenter = numpy.zeros(self.specF['N_sensory_pools'])
      InputCenter[Matrix_pools_receiving_inputs] = [stimuli_grid[idx] for idx in index_combination]

      # inp_vect is a matrix which gives the stimulus input to each SN
      inp_vect = numpy.zeros((self.specF['N_sensory_pools'], self.specF['N_sensory']))
      for index_pool in Matrix_pools_receiving_inputs:
          inp_vect[index_pool, :] = self.give_input_stimulus(InputCenter[index_pool])
      
      # ipdb.set_trace()

      # Running
      for index_pool in range(self.specF['N_sensory_pools']):
          Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool]
      if self.specF['with_random_network']:
          RCN_pool.S_ext_rnd = inp_baseline_rnd
      run(IT1)
      
      for index_pool in range(self.specF['N_sensory_pools']):
          if index_pool in Matrix_pools_receiving_inputs:
              Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool] + inp_vect[index_pool]
          else:
              Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool]
      if self.specF['with_random_network']:
          RCN_pool.S_ext_rnd = inp_baseline_rnd
      run(IT2 - IT1)

      for index_pool in range(self.specF['N_sensory_pools']):
          Recurrent_Pools[self.specF['N_sensory']*index_pool:self.specF['N_sensory']*(index_pool+1)].S_ext = inp_baseline[index_pool]
      if self.specF['with_random_network']:
          RCN_pool.S_ext_rnd = inp_baseline_rnd
      run(Simtime - IT2)

      # Results calculation
      End_of_delay = self.specF['simtime'] - self.specF['window_save_data']
      time_matrix_rn = numpy.zeros(S_rn.t.shape[0])
      for index in range(S_rn.t.shape[0]):
          time_matrix_rn[index] = S_rn.t[index]

      time_matrix_rcn = numpy.zeros(S_rcn.t.shape[0])
      for index in range(S_rcn.t.shape[0]):
          time_matrix_rcn[index] = S_rcn.t[index]

      # ipdb.set_trace()
      psth_rn = self.compute_psth_for_mldecoding(time_matrix_rn, S_rn.i, End_of_delay, self.specF['decode_spikes_timestep'])[:, int(round(End_of_delay / self.specF['decode_spikes_timestep'])) - 1]
      psth_rcn = self.compute_psth_for_rcn(time_matrix_rcn, S_rcn.i, End_of_delay, self.specF['decode_spikes_timestep'])[:, int(round(End_of_delay / self.specF['decode_spikes_timestep'])) - 1]

      for idx in range(len(Matrix_pools_receiving_inputs)):
        index_pool = Matrix_pools_receiving_inputs[idx]
        # ipdb.set_trace()
        psth_rn_trial[index_combination][idx] = psth_rn[index_pool*self.specF['N_sensory']:(index_pool+1)*self.specF['N_sensory']]

      psth_rcn_trial[index_combination] = psth_rcn
      gcPython.collect()

    return psth_rn_trial, psth_rcn_trial, index_simulation


  def find_tuning_curve(self) : 
    num_stimuli = self.specF['num_stimuli_gird']
    num_trials = self.specF['Number_of_trials']
    stimuli_grid = np.linspace(0, self.specF['N_sensory'], num_stimuli, dtype=int)
    psth_rn = numpy.zeros((num_trials, len(stimuli_grid), self.specF['N_sensory']))
    psth_rcn = numpy.zeros((num_trials, len(stimuli_grid), self.specF['N_random']))

    # ipdb.set_trace()
    load = self.specF['number_of_specific_load']
    shape_rn = (num_trials,) + (len(stimuli_grid),) * load + (load,) + (self.specF['N_sensory'],)
    shape_rcn = (num_trials,) + (len(stimuli_grid),) * load + (self.specF['N_random'],)
    psth_rn = numpy.zeros(shape_rn) # spikes count in 100ms. (num_trials, stimuli1,stimuli2,...,loads, neurons in each ring network)
    psth_rcn = numpy.zeros(shape_rcn) # spikes count in 100ms. (num_trials, stimuli1,stimuli2, neurons in the random network)

    if not (self.specF['same_network_to_use'] and os.path.exists(self.specF['path_for_same_network'])):
      self.initialize_weights()

    if self.specF['num_cores'] == 1:
      for index_simulation in tqdm(range(num_trials), desc="Outer Loop (trials)"):
        psth_rn_trial,psth_rcn_trial,index_simulation = self.run_a_trial(index_simulation, stimuli_grid)
        psth_rn[index_simulation, :, :] = psth_rn_trial
        psth_rcn[index_simulation, :, :] = psth_rcn_trial
        gcPython.collect()

    elif self.specF['num_cores'] > 1:
      with ProcessPoolExecutor(max_workers=self.specF['num_cores']) as executor:
        futures = [executor.submit(self.run_a_trial, index_simulation, stimuli_grid) for index_simulation in range(num_trials)]
        for future in tqdm(as_completed(futures), total=num_trials, desc="Outer Loop (trials)"):
        # for future in as_completed(futures):
          result = future.result() 
          psth_rn[result[-1]] = result[0]
          psth_rcn[result[-1]] = result[1]
          gcPython.collect()
    else:
      error

    # saving  
    numpy.savez_compressed(self.specF['path_psth'],psth_rn=psth_rn,psth_rcn=psth_rcn,stimuli_grid=stimuli_grid)
    print("All results saved in the folder")
    gcPython.collect()
    
    return
  




