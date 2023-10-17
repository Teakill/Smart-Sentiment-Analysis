  # Smart-Sentiment-Analysis
Smart-Sentiment-Analysis is an innovative approach towards sentiment analysis, originally inspired by the data from Hackathon 2022.

[https://vinuni.edu.vn/wp-content/uploads/2022/12/34790f8754378d69d426.jpg
](https://www.google.com/url?sa=i&url=https%3A%2F%2Fvinuni.edu.vn%2Fevent%2Fvinuni-hackathon-2022%2F&psig=AOvVaw2ttYQucqBOYt6wnUvjMCrz&ust=1697188984110000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCOj5-eSX8IEDFQAAAAAdAAAAABAE)

🧠 Model Architecture
The model is structured around 6 unique classifiers. Each classifier is dedicated to a specific feature, making for a total of 36 neurons. This architecture allows for fine-grained sentiment analysis and captures nuances that traditional models might miss.

There are 3 different version: Concat 4 last layers , Concat 4 last layers + mean - max pooling (highest final score), Concat 4 last layers + LSTM pooling (all trained on 30 epochs)

🎯 Results
Our model achieved a commendable score of 0.73! To put that into perspective, the winning model's score was just a nudge higher at 0.75. We're pretty proud of how close we came to the top spot!

⚡ Fun Fact
Did you know? The human brain consists of approximately 86 billion neurons! Our model might have 36, but we're catching up! 😉

Technology
-Torch Framework
-VnCoreNlp
-KFoldStratified

Preprocessing
-Remove special char
-Remove stopword (without doing this, the model actually performs better)
-Segmentation

How to train: https://colab.research.google.com/drive/1Hvm9CRNRchITdAJcYqKK3ozxl-nge-sD?usp=sharing
