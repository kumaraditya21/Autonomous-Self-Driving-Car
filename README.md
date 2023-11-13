# Autonomous-Self-Driving-Car
This project focuses on achieving autonomous navigation for a car using computer vision and Convolutional Neural Networks (CNN). The entire pipeline includes data collection, training on a CNN model, and deployment on the car for real-time navigation through curved paths.
## Prerequisite

Ensure you have the Sunfounder PiCar-X kit. You can find more information and purchase the kit [here](https://docs.sunfounder.com/projects/picar-x/en/latest/).

## Setting up the Sunfounder PiCar-X

Follow the steps in the [YouTube tutorial](<setup_youtube_link>) to set up your PiCar-X for autonomous driving and install all the necessary libraries to run the car.

## Data Collection

Data collection is performed using the Sunfounder App-based controller. Refer to the [YouTube tutorial](<data_collection_youtube_link>) for a detailed guide on data collection.

## Training the Data

For training the data, the Nvidia Network Architecture is employed. The weights of the network are trained to minimize the mean-squared error between the steering command output by the network. The network architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The input image is split into YUV planes and passed to the network. The trained model file is converted to a TFLite file using int8 quantization and saved for running it in the car. For more details, refer to the [YouTube tutorial](<training_youtube_link>).

## Running the Car

The TFLite model is deployed on a Raspberry Pi to run the car on a curved path autonomously. Refer to the above demonstration video for the car in action.


