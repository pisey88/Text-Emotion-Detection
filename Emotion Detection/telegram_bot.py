from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
from text_cleaner import clean_text
import os
import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline_logistic_regression.pkl")
pipe_rf = joblib.load(open(MODEL_PATH, "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

def predict_emotion(text):
    pred = pipe_rf.predict([text])[0]
    proba = pipe_rf.predict_proba([text])
    confidence = np.max(proba) * 100
    emoji = emotions_emoji_dict.get(pred, "❓")
    return pred, emoji, confidence


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Send me a sentence and I’ll predict the emotion behind it.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    pred, emoji, confidence = predict_emotion(user_text)
    response = f"**Prediction**: {pred} {emoji}\n**Confidence**: {confidence:.2f}%"
    await update.message.reply_text(response, parse_mode="Markdown")


def main():
    BOT_TOKEN = "7384843114:AAF1pwJVkiFKVsZ4Pr3uzObPTMDMF5qHpqo"

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
