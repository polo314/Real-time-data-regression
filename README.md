# Real-time-data-regression

A machine learning experiment for bulding a real time regressor of financial data downloaded from the internet:

-supervised learning
-an experiment in terms of time series elaboration: features condensate informations of previous one data (some of the features are moving averages, some others try to measure the local variance in a time window range). As a result the dataset can be shuffled as each row of the dataset is standing alone with all the informations and correlations with its past. The advantage of this approach is easyness as the data can be freely divided into smaller dataset, shuffled in order to contain the overall spectrum of oscillations, without having to preserve time line cryteria.

-a blender of ada, gradient boosting and random forest regressor algorithms is used for a first layer prediction. A final layer uses gradient boosting algorithm to obtain a real time regression of the index.

After the training session, the third part of the program consists of a real time bot that within a given interval of time downloads fresh data and build a dataset. It then uses machine learning models to perform regression for evaluating the next value. Showing the updating real data and regression ones on a graph is not immediate due to time needed for computation of the first part of the dataset (this time depends on the efficiency of the code and on computation power of the pc used). During this time the program needs also to collect and process new incoming data at regular time intervals in order to not lose them. Once this synchronization between incoming new data and time needed to build an up to date dataset is obtained it is then able to update it and perform regressions for every new downloaded data, thus giving a new result within a certain time interval (which can range from seconds to 1-2 minutes or more depending on the user preferences).

The program is intended to be for educational purposes, as it doesn't aim to make predictions of financial data: the author actually thinks financial data aren't predictable.

Steps:

1 Loading data with "tools_data_download" file:
	-insert a folder address where to save downloaded data 
2 Building the training datasets. Creating the blender and final Gradient Boosting model:
	-insert the folder address previously indicated: there are several steps where data are saved, this was due to development needs, I will 	  deactivate some of them later. 
	-machine learning environement models are saved in the working space folder, but you can manually add the folder address with the model's name 		 when jobilib.dump is used.  
3 Real time regressor:
	-if you saved the machine learning models in a different folder, remember to specify it when using joblib.load. 
