==================
Requirements
==================

This code runs both in 3x. In the github repository, the requirements.txt file contains the libraries to run ``nnogada``. You can manually install these dependencies with:

.. code-block:: bash
   
   pip3 install -r requirements.txt


..  _using_google_colab:

Using Google Colab
-------------------

For easy and immediate use, we recommend using Google Colab. Within it, it is necessary to install some libraries and simplemc in the first cells.


.. code-block:: bash
   
   !pip install -e git+https://github.com/igomezv/Nnogada#egg=nnogada
   %cd /content/src/nnogada/

If you have an error, restart the kernel, rerun and that's it, the problem could be solved. Then, you can follow any example of the  `Examples section <examples.html>`_ .

