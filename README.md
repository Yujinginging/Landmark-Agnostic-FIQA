# Landmark-Agnostic-FIQA
DTU master thesis

This project is from the master thesis "Landmark-agnostic Face Image Quality Assessment". The primary focus of the research is a comparative analysis between the landmark-agnostic approach, represented by the face-parsing method, and the landmark-dependent approach, represented by the Dlib method. The evaluation involves measuring various quality components, such as head length, head size, inter-eye distance, eye open, mouth closed, and crop of the face image. In addition, a novel heuristic function for computing head length is proposed and tested under various resolutions. The results of the comparative analysis, heuristic function testing, and their addressing of the research questions are presented through illustrative diagrams. The thesis concludes with suggestions for addressing limitations and outlines potential future work.

**Required libraries:**

To run the face-parsing scripts, please install: 
1. Face-parsing.PyTorch (https://github.com/zllrunning/face-parsing.PyTorch.git)
2. The pre-trained model from face-parsing. Save them in your project for further usage.

Three requirements files are provided in the main branch, since different estimators used in this project may require different working environments. As a suggestion, please build various virtual environments to run the estimators (for example, if you want to test the dlib, please build the virtual environment based on the given requirement file named "requirements MTCNN Dlib MediaPipe.txt"). 

**Used Datasets for quality components measurements:**
1. FRLL dataset: (https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666)
2. LFW dataset: (https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download)

All of the employment and implementation of the project are included in the "code" folder:
1. To get the quality components measurements with multiple images (dataset), please go to the "QC measurements" folder, "DL_main" is the script for measures using Dlib, and "FP__QC_measurements.py" is for measures using face-parsing. Dlib has a separate file for the head length measurement: "dlib_measurements.py". Please do not forget to change the input folder and output file path in the script to produce the results.
2. Dlib and face-parsing have two separate folders, each consisting of the script used to run and test on a single image.
3. The "landmark-dependent estimators selection" folder is for scripts running for five evaluated estimators.

In "excel_outputs" folder, contains Excel worksheets with results from both Dlib and face-parsing estimators, and the comparison results and diagrams.
