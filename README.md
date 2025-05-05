Time Series Based Supply Chain Demand Forecasting Using LSTM

- Project Overview

This project explores demand forecasting using Long Short-Term Memory (LSTM)  to predict future order demand based on historical data. The objective is to enhance inventory management and supply chain efficiency by leveraging deep learning models.

- Project Status

This is an ongoing project with active improvements and refinements in model training, evaluation, and deployment. While initial model performance metrics suggest challenges in generalization, further optimizations are in progress to improve predictive accuracy.

- Objectives

    - Develop a time-series forecasting model using LSTM networks.

    - Optimize model training on TPU for efficiency.

    - Implement robust preprocessing and data transformation techniques.

    - Evaluate model performance and refine hyperparameters.

    - Deploy a scalable prediction model for real-world supply chain applications.

- Dataset

The dataset used for this project is Historical Product Demand.csv from Kaggle

- Data Preprocessing

    - Log Transformation: Applied to stabilize variance in demand values.

    - MinMax Scaling: Normalized data between 0 and 1 for improved LSTM performance.

    - Sequence Generation: Converted time series into supervised learning format.

- Model Architecture

The LSTM model consists of:

    - LSTM Layer 1: 64 units, return sequences enabled.

    - Dropout: 20%.

    - LSTM Layer 2: 32 units, return sequences disabled.

    - Dropout: 20%.

    - Dense Layer : 16 units with ReLU activation.

    - Output Layer: 1 neuron for demand prediction.

- Current Challenges and Ongoing Improvements

    - Model Evaluation and Performance Issues

    - Initial evaluation metrics indicate that the model predictions are not well-aligned with actual demand trends.

    - The predicted values show minimal variation, requiring further investigation into training dynamics, loss convergence, and data   preprocessing.

- Next Steps

    - Refine Preprocessing Pipelines: Ensure inverse transformations correctly map back to original demand values.

    - Hyperparameter Tuning: Adjust learning rate, batch size, and sequence length to optimize model learning.

    - Additional Feature Engineering: Incorporate external variables such as holidays, weather patterns, or seasonal trends.

    - Model Training Enhancements: Increase training epochs and monitor overfitting with early stopping mechanisms.

- Deployment Plan

    - Once the model achieves satisfactory performance, the next steps will include:

    - Saving and Versioning Models: Maintain different versions of trained models for comparison.

    - API Development: Deploy a REST API using Flask or FastAPI to serve predictions.

- Conclusion

While this project is still in development, it highlights the complexities of time-series forecasting using deep learning. The initial findings underscore the importance of feature engineering, hyperparameter tuning, and systematic evaluation in achieving a high-performing demand forecasting model. Future iterations will focus on refining model accuracy and deploying a production-ready solution.

