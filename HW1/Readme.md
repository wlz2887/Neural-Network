The hand-written backpopagation algorithm for spiral data classification.  
==========================================================================
Train
------
By changing ```rate``` and ```Epoch``` in ```hw_bp.py``` , you can train the network by different learning rate and repeat different times.  

Validation
-----------
The ```validation``` function chooses 100 * 100 points in the 12 * 12 figure and pass the coordinates through the network. By comparing the output with ```0.5```, we can judge the color of it and thus using different color to plot the point.
