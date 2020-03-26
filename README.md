# covid_classification
This is a shared repository for the project to build an image classifier distinguishing x-rays of healthy lungs, covid-19 infected lungs and pneumonia infected lungs.

Ideas/Goals:

- write clean code and lay down ground work for future analysis
- compose a useful dataset with an appropriate datastructure
- provide some insight for affected areas of the covid-19 cases (i.e. shap values, GRAD-CAM)

## Data

Covid X-Ray data
Collected from different sources by Joseph Paul Cohen. Postdoctoral Fellow, Mila, University of Montreal
https://github.com/ieee8023/covid-chestxray-dataset

Pneumonia dataset 1
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Pneumonia dataset 2
National Institutes of Health Clinical Center
https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview

Other existing datasets, most of them restricted reseach only and not on the public domain
MIT Chest X-Ray database https://archive.physionet.org/physiobank/database/mimiccxr/
Stanford Chest X-Ray database https://stanfordmlgroup.github.io/projects/chexnet/
MIMIC Chest X-ray performed at Beth Israel Deaconess Medical Center in Boston, MA https://physionet.org/content/mimic-cxr-jpg/2.0.0/
PadChest Hospital San Juan de Alicante – University of Alicante  http://bimcv.cipf.es/bimcv-projects/padchest/