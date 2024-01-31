# Landmark-Agnostic-FIQA
DTU master thesis

This project is from the master thesis "Landmark-agnostic Face Image Quality Assessment". The primary focus of the research is a comparative analysis between the landmark-agnostic approach, represented by the face-parsing method, and the landmark-dependent approach, represented by the Dlib method. The evaluation involves measuring various quality components, such as head length, head size, inter-eye distance, eye open, mouth closed, and crop of the face image. In addition, a novel heuristic function for computing head length is proposed and tested under various resolutions. The results of the comparative analysis, heuristic function testing, and their addressing of the research questions are presented through illustrative diagrams.

**Required libraries:**

To run the face-parsing scripts, please install: 
1. Face-parsing.PyTorch (https://github.com/zllrunning/face-parsing.PyTorch.git)
2. The pre-trained model from face-parsing. You can save them in your project for further use.

Three requirements files are provided in the main branch, since different estimators used in this project may require different working environments. As a suggestion, please build various virtual environments to run the estimators (for example, if you want to test the dlib, please build the virtual environment based on the given requirement file named "requirements MTCNN Dlib MediaPipe.txt". To create the virtual environment on the given requirements file, you can follow: https://stackoverflow.com/questions/41427500/creating-a-virtualenv-with-preinstalled-packages-as-in-requirements-txt). 

**Used Datasets for quality components measurements:**
1. FRLL dataset: (https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666)
2. LFW dataset: (https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download)

## Comparative Analysis
All of the employment and implementation of the comparative analysis are included in the "code" folder:

### Face-parsing:
To get the results of the quality component measurements on head size, inter-eye distance, eye open, mouth closed, and crop of the face: "QC measurement" > "FP__QC_measurements.py".
Please do not forget to change the path of the pre-trained model (cp='' in the evaluate() method).

### Dlib
To get the results of the quality component measurements on head size, inter-eye distance, eye open, mouth closed, and crop of the face: "QC measurement" > "DL_main.py".

Please change the input folder path of the dataset (e.g., LFW dataset) and the Excel file path that will save the results for both estimators.

## Head length measurement with different resolutions:
All of the employment and implementation of this topic are included in the "code" folder:
### Face-parsing: 
Please find it here: "QC measurement" >"FP__QC_measurements_resolutions.py".
### Dlib:
Please find it here: "QC measurement" > "dlib_headlength.py".

Please change the folder path of the dataset and CSV file path for results saving.

To test the quality component measurement on a single image and for visualization of the results, please use the Python files in the folder:
1. For Dlib: "code" > "dlib" > "DLIB_QualityComponents.py"; to see all landmarks:  "code" > "dlib" > "dlib.landmarks.py".
2. For Face-parsing: "code" > "dlib" > "QC_single image.py".

## Landmark-dependent estimators
This is for the selection of a landmark-dependent estimator, the files for the tested five estimators can be found in the folder: "code" > "landmark-dependent estimators selection".


## Comparison results
In "excel_outputs" folder, contains Excel worksheets with results from both Dlib and face-parsing estimators, and the comparison results and diagrams.
