import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from keras.applications.vgg16 import VGG16
from module_16_models import get_data_for_cnn, get_data_for_vgg16, CNN_model, VGG16_model, add_padding

# Initialize session state variables
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'test_images' not in st.session_state:
    st.session_state.test_images = None
if 'test_labels' not in st.session_state:
    st.session_state.test_labels = None
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False

if st.button('Stop Execution'):  # Stop Execution Button
    st.write("Execution Stopped")
    st.stop()  # Stops the execution of the Streamlit app


st.title('Image Classification with CNN and VGG16')
model_option = st.selectbox('Select Model', ('CNN', 'VGG16'))
epochs = st.slider('Number of Epochs', 1, 30, 5)
learning_rate = st.slider('Learning Rate', 0.001, 0.05, 0.005)
n_hidden_1 = st.number_input('Number of Hidden Units', 32, 512, 128)
batch_size = st.number_input('Batch size', 32, 512, 128)

# Radio buttons for activation function and output function
activation_function_l1 = st.radio('Activation Function', ['relu', 'tanh', 'sigmoid'])
activation_output = st.radio('Output Function', ['softmax', 'sigmoid'])

if st.button('Start Training'):   # Button to start training
    # Reset stop signal
    st.session_state.stop_training = False

    progress_bar = st.progress(0)
    status_text = st.empty()
    combined_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    if st.button('Stop Training'):    # Add a "Stop Training" button
        st.session_state.stop_training = True

    if model_option == 'CNN':
        (train_images, train_labels), (test_images, test_labels) = get_data_for_cnn()  # Load data

        model = CNN_model(
            activation_function_l1=activation_function_l1,
            n_hidden_1=n_hidden_1,
            activation_output=activation_output,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        def train_with_progress(model, train_images, train_labels, batch_size, epochs, validation_data):
            for epoch in range(epochs):
                if st.session_state.stop_training:
                    st.warning("Training stopped.")
                    break

                history = model.fit(
                    train_images, train_labels,
                    batch_size=batch_size,
                    epochs=1,
                    validation_data=validation_data,
                    verbose=0)

                for key in combined_history:
                    if key in history.history:
                        combined_history[key].extend(history.history[key])

                progress = (epoch + 1) / epochs                  # Update progress bar
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch + 1}/{epochs} completed')
            return combined_history

        history = train_with_progress(model, train_images, train_labels, batch_size, epochs, validation_data=(test_images, test_labels))

    else:
        (train_data, test_data), (test_images, test_labels) = get_data_for_vgg16(batch_size)
        model = VGG16_model(
            activation_function_l1=activation_function_l1,
            n_hidden_1=n_hidden_1,
            activation_output=activation_output,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        def train_with_progress(model, train_data, epochs, validation_data): # Training function with progress bar
            for epoch in range(epochs):
                if st.session_state.stop_training:
                    st.warning("Training stopped.")
                    break

                history = model.fit(
                    train_data,
                    steps_per_epoch=200,
                    epochs=1,
                    validation_data=validation_data,
                    validation_steps=50,
                    verbose=0)

                for key in combined_history:
                    if key in history.history:
                        combined_history[key].extend(history.history[key])

                progress = (epoch + 1) / epochs          # Train the model and show progress
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch + 1}/{epochs} completed')
            return combined_history

        history = train_with_progress(model, train_data, epochs, validation_data=test_data)         # Train the model and show progress

    if not st.session_state.stop_training:
        # Plot the training loss and accuracy
        st.write("Training Loss and Accuracy")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history['loss'], label='Training Loss')
        ax[0].plot(history['val_loss'], label='Validation Loss')
        ax[0].legend()

        ax[1].plot(history['accuracy'], label='Training Accuracy')
        ax[1].plot(history['val_accuracy'], label='Validation Accuracy')
        ax[1].legend()

        st.pyplot(fig)

        # Save trained model and test data to session state
        st.session_state.trained = True
        st.session_state.model = model
        st.session_state.test_images = test_images
        st.session_state.test_labels = test_labels


if st.session_state.trained:
    image_index = st.slider('Select an image from the test dataset', 0, st.session_state.test_images.shape[0] - 1, 0) # Select an image from the test dataset using a slider
    image = st.session_state.test_images[image_index]
    st.image(image, caption='Selected Image', use_column_width=True)

    if st.button('Classify Image'):
        if model_option == 'CNN':
            image_resized = np.expand_dims(image, axis=0)
            predictions = st.session_state.model.predict(image_resized)
        else:
            image_resized = np.expand_dims(image, axis=-1).astype('float32') / 255.0
            image_resized_padded = add_padding(image_resized)
            image_resized_padded_tf = tf.convert_to_tensor(image_resized_padded, dtype=tf.float32) # Convert NumPy array to TensorFlow tensor
            image_resized_padded_tf_rgb = tf.image.grayscale_to_rgb(image_resized_padded_tf)             # Convert grayscale image to RGB
            predictions = st.session_state.model.predict(image_resized_padded_tf_rgb)

        predicted_class = np.argmax(predictions, axis=-1)[0]
        st.write(f'Predicted Class: {predicted_class}, True Class: {np.argmax(st.session_state.test_labels[image_index])}')
        st.bar_chart(predictions[0])