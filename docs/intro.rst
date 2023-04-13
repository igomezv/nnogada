==================
First steps
==================

.. toctree::
   :maxdepth: 1
   
   requirements
   
   quickstart 
   
   inifile
   
After proper installation of the `requirements <requirements.html>`_, generally speaking, the steps to run ``SimpleMC``  are as follows: 1) configure an :ref:`ini file`, 2) read this in a :ref:`Python script`, 3) :ref:`run in terminal`, and 4) :ref:`analyze outputs`.

The output chains of Bayesian inference algoritms (MCMC and nested sampling) are text files that have the same structure as the `CosmoMC <https://cosmologist.info/cosmomc/>`_ chains consisting in:

	- First column.- weights of the sampling method. 
	- Second column.- log-likelihood function.
	- Third to N+2 columns.- N free parameters of the cosmological model under analysis. 
	- Last columns.- log-likelihood function per each type of data and derived parameters.

In addition, two other output files are generated: a summary and a ``.paramnames`` file with the names of the free parameters of the cosmological model. 





