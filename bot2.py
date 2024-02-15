# for speech-to-text
import speech_recognition as sr
# for text-to-speech
from gtts import gTTS
# for language model
import transformers
import os
import time
# for data
import os
import datetime
import numpy as np
import yfinance as yf
from googlesearch import search
import matplotlib.pyplot as plt

# Building the AI
class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def get_user_input(self):
        print("Choose input method:")
        print("1. Text")
        print("2. Speech")
        choice = input("Enter your choice (1/2): ")
        if choice == '1':
            self.text_input()
        elif choice == '2':
            self.speech_input()
        else:
            print("Invalid choice. Please select 1 or 2.")
            self.get_user_input()

    def text_input(self):
        self.text = input("You --> ")
        self.process_input()

    def speech_input(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio)
            print("You --> ", self.text)
            self.process_input()
        except:
            print("You --> ERROR")
            self.process_input()

    @staticmethod
    def text_to_speech(text):
        print("Dev --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        os.system('start res.mp3')  #if you are using mac->afplay or else for windows->start
        time.sleep(int(50*duration))
        os.remove("res.mp3")

    def process_input(self):
        ## wake up
        if self.wake_up(self.text) is True:
            res = "Hello I am Pauldern the AI, what can I do for you?"
        ## action time
        elif "time" in self.text:
            res = self.action_time()
        ## respond politely
        elif any(i in self.text for i in ["thank", "thanks"]):
            res = np.random.choice(["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "mention not"])
        elif any(i in self.text for i in ["exit", "close", "tata", "bye"]):
            res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon"])
            self.exit()
        elif "stock" in self.text:
            # Extract the stock symbol from the user's query (you may need to refine this logic)
            symbol = self.text.split()[-1]
            stock_data = self.collect_stock_data(symbol)
            if not stock_data.empty:
                predicted_price = self.predict_stock_price(stock_data)
                if predicted_price is not None:
                    res = f"Here is the stock data for {symbol}:\n{stock_data}\n\nPredicted stock price for the next day: ${predicted_price:.2f}"
                else:
                    res = "Sorry, I couldn't make a prediction for that stock."
            else:
                res = "Sorry, I couldn't fetch data for that stock symbol."
        elif "search" in self.text:
            # Extract the search query from the user's input (you may need to refine this logic)
            query = self.text.split("search", 1)[1].strip()
            search_result = self.search_stock_info(query)
            res = "Here is what I found:\n{}".format(search_result)
        ## conversation
        else:
            if self.text == "ERROR":
                res = "Sorry, come again?"
            else:
                chat = nlp(transformers.Conversation(self.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ")+6:].strip()
        self.text_to_speech(res)
        self.get_user_input()

    def exit(self):
        print("----- Closing down Pauldern -----")
        exit()

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

    @staticmethod
    def collect_stock_data(symbol):
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        return data

    @staticmethod
    def search_stock_info(query):
        try:
            search_results = list(search(query, num=1, stop=1))
            return search_results[0]
        except Exception as e:
            print("Error searching for stock information:", str(e))
            return "Sorry, I couldn't find information for that stock."

    @staticmethod
    def predict_stock_price(data):
        if not data.empty:
            # Create a new column 'Day' representing the number of days
            data['Day'] = range(1, len(data) + 1)
            # Define the features (X) and target (y)
            X = data[['Day']]
            y = data['Close']
            # Implement your prediction model here (e.g., linear regression, ARIMA, etc.)
            # For demonstration, let's use a simple linear regression model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            # Predict the stock price for the next day
            next_day = len(data) + 1
            predicted_price = model.predict([[next_day]])[0]
            # Create a graph to visualize the prediction
            plt.figure(figsize=(10, 6))
            plt.scatter(data['Day'], data['Close'], color='blue', label='Actual Price')
            plt.plot(next_day, predicted_price, marker='o', markersize=8, color='red', label='Predicted Price')
            plt.xlabel('Day')
            plt.ylabel('Stock Price')
            plt.title('Stock Price Prediction')
            plt.legend()
            plt.grid(True)
            # Save the graph as an image
            plt.savefig('stock_prediction.png')
            return predicted_price
        else:
            return None

# Running the AI
if __name__ == "__main__":
    ai = ChatBot(name="Pauldern")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ai.get_user_input()
