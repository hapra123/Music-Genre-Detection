Initially, we chose KNN based on our analysis, but after using LazyPredict, we found that SVC performed better. Therefore, we switched to SVC as the base model. We achieved an accuracy of 88% with the SVC base model, and after adding CalibratedClassifierCV, we improved the accuracy to 90%. The following files are included:

train_knn.py: The KNN model training script.

test_for_accuracy_lazy.py: The script for running LazyPredict and evaluating different models.

svc.py: The final script using SVC with CalibratedClassifierCV, which resulted in the improved accuracy. The model has been pickled as asvc_model.