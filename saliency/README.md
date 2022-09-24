In this section the code is for creating saliency heatmaps and comparing them with the human gaze heatmaps created by using the Reader class. 
To execute the files, you might need additional code from other folders of this repository. 

* [integrated_gradients.py](integrated_gradients.py) contains all needed functions to compute the integrated gradients of a model ([Source](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/integrated_gradients/integrated_gradients.ipynb)).
* [perturbation_for_sarfa.py](perturbation_for_sarfa.py) contains all the functions for perturbing an image and calculating the SARFA-saliency. For the calculation process it uses a function from the file [sarfa_saliency.py](sarfa_saliency.py) which was made publically availble by the original authors (Puri et al. 2020) on the [Github](https://github.com/nikaashpuri/sarfa-saliency) accompanying their Paper. 
* [heatmap_comparison.py](heatmap_comparison.py) contains the functions for the comparison of the gaze data (e.g. gaze heatmap) and the saliency heatmaps.
<br />
Main Files: <br />

* [create_saliency_database_main.py](create_saliency_database_main.py) calculates the SARFA-saliency heatmaps for the given area and stepsize (as variables) and saves them as png. (The created dataset we used for the comparison can be found [here](https://osf.io/eyskv/?view_only=9dab6ccb848d471a8f0e46dfbf8ee195)). 
* [all_saliency_comparison_main.py](all_saliency_comparison_main.py) compares the saliency maps we put in the dataset using one of the methodes presented in the report. Which one is chosen by a variable. 
* [average_pixel_comaprison_main.py](average_pixel_comaprison_main.py) compares the average pixels being one when converting the human gaze heatmap and saliency heatmaps to binary maps and the value for the fixation maps (already binary) and also their average pixel value originally.
* [comparison_finding_max_min_main.py](comparison_finding_max_min_main.py) this main finds the max and min value of a comparison method and returns the value and index. If the method is correlation is also visualizes both heatmaps that are compared as well as their correlation heatmap.
<br />

[saliency_results](saliency_results) :<br />
In this folder there are textfiles containing the output of [all_saliency_comparison_main.py](all_saliency_comparison_main.py) for different methods as well as the output of [average_pixel_comaprison_main.py](average_pixel_comaprison_main.py) and [comparison_finding_max_min_main.py](comparison_finding_max_min_main.py) using correlation. All averaged values are also in the [all_results.csv](saliency_results/all_results.csv) file and in the file [list_all_result_values.csv](saliency_results/list_all_result_values.csv) are the calculated values for each single data pair. 

<br />
[Paper_Image_Creation_Code](Paper_Image_Creation_Code) :<br />
In this folder are several main files who create visualizations for our report. They need other files to work, but are stuck in this folder, because they where only for creating these visualizations. 