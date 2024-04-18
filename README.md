# AI-Based Face Mask Detection System

This project is an AI-based face mask detection system developed using TensorFlow and Keras. The system uses a trained VGG16 model to predict whether a person in an uploaded image is wearing a mask or not. The application is integrated with a Streamlit app for user interaction and is deployed on AWS EC2.

## Project Structure

The project is divided into two main parts:

1. **Model Training:** The model is trained using a face mask dataset downloaded from Kaggle. The dataset is split into training and validation subsets. The VGG16 model is used as the base model, and additional dense layers are added for the binary classification task. The model is trained for 5 epochs.

2. **Streamlit App:** The trained model is loaded into a Streamlit app. The app allows users to upload an image, and it uses the trained model to predict whether the person in the image is wearing a mask or not.

## Usage

1. Clone the repository.
2. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Results

The model achieves an accuracy of 95% on the validation set.

## Future Work

Future improvements could include training the model on a larger dataset using data augmentation technique, fine-tuning the model, and improving the user interface of the Streamlit app.

## License

This project is licensed under the terms of the MIT license.
