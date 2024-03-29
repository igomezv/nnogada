===========
nnogada
===========

This is the ``nnogada`` documentation, a Python package for the hyperparameter tuning for feedforward neural networks implemented by ``keras`` or ``torch``. 

There are some options to install, or use, ``nnogada``: 

1) We recommend to install ``nnogada`` without clonning in the following way: 

   .. code-block:: bash
      
        pip3 install -e git+https://github.com/igomezv/Nnogada#egg=nnogada

2) On the other hand, you can visit the `GitHub code repository <https://github.com/igomezv/Nnogada.git>`_, download the `source code here <https://github.com/igomezv/Nnogada/archive/refs/heads/master.zip>`_ or clone it as follows:

   .. code-block:: bash

      git clone https://github.com/igomezv/Nnogada.git

Then, you can install it:

   .. code-block:: bash

      cd Nnogada
      
      pip3 install -e .

3) You can use ``nnogada`` in Google Colab, please see the Section `Using Google Colab in Requirements <requirements.html#using-google-colab>`_.

4) ``pip3 install nnogada`` sometimes fails.

Please read the `introduction <intro.html>`_ section where you can see the `requirements <intro.html#requirements>`_  and a simple `quick start <intro.html#quick-start>`_. 


For ``nnogada`` citation, please go to the `citation  section <Citation.html#cite-external-codes>`_.

Flow:

.. figure:: img/geneticalg.png

Output:

.. figure:: img/nnogada_output.png


Extended index
---------------

.. toctree::
   :maxdepth: 1
   
   requirements
   
   intro
   
   Citation
   
   API


Changelog
----------

- **0.9.0.0 (13/04/2023)** First release.


TO DO
------
- Add classification for torch models. 

