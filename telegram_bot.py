import logging
import json
import random
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🔥 Telegram Token (замени на свой)
TOKEN = "7588087338:AAGesdDnSAWFW4zHsioSEckDovg92-PXmeA"

# 🏎️ Используем GPU (если доступно)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🚀 Загружаем новую русскую GPT-модель
model_name = "ai-forever/mGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# ⚡ Ускоряем модель (если есть видеокарта)
if device == "cuda":
    model.half()

# 🔍 Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📂 Функции для работы с памятью (сохранение истории чата)
def save_chat_history(user_id, chat_history):
    with open(f"history_{user_id}.json", "w") as file:
        json.dump(chat_history, file)

def load_chat_history(user_id):
    try:
        with open(f"history_{user_id}.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# 🏁 Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я твой AI-бот. Чем могу помочь? 😊")

# ℹ️ Команда /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я могу ответить на твои вопросы! Просто напиши мне 😊")

# 🔄 Команда /reset (очистка памяти)
async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    context.user_data["chat_history"] = []
    save_chat_history(user_id, [])
    await update.message.reply_text("История чата очищена! Начнем с чистого листа. 🧹")

# 🤣 Команда /joke (шутки)
async def joke_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jokes = [
        "Почему программисты любят тёмную тему? Потому что свет притягивает баги! 🤣",
        "Что сказал Python коду? 'Ты слишком многозадачен!' 🐍"
    ]
    await update.message.reply_text(random.choice(jokes))

# 🤖 Генерация ответа GPT
def generate_response(chat_history):
    input_text = " ".join(chat_history)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=150,  # Длина ответа
        temperature=0.8,  # Баланс между логикой и креативностью
        top_p=0.9,  # Умное семплирование
        repetition_penalty=1.2,  # Убираем повторения
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    bot_response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Если ответ бессмысленный - даём заранее заготовленный
    if bot_response.strip() == "" or len(bot_response) < 10:
        bot_response = random.choice([
            "Хм... интересный вопрос! 🤔",
            "Давай обсудим! 😉",
            "Не совсем понял, можешь уточнить? 😊",
            "Хороший вопрос! Сейчас подумаю... 🤓"
        ])

    return bot_response

# 📩 Обработка сообщений
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    user_id = update.message.chat_id
    logger.info(f"Пользователь ({user_id}): {user_message}")

    # Загружаем историю чата пользователя
    chat_history = load_chat_history(user_id)
    chat_history.append(user_message)

    # Оставляем только последние 10 сообщений
    chat_history = chat_history[-10:]
    context.user_data["chat_history"] = chat_history

    # Генерация ответа от GPT
    bot_response = generate_response(chat_history)
    logger.info(f"Бот: {bot_response}")

    # Сохраняем обновлённую историю чата
    save_chat_history(user_id, chat_history)

    await update.message.reply_text(bot_response)

# 🚀 Запуск бота
def main():
    application = Application.builder().token(TOKEN).build()

    # Добавляем команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("joke", joke_command))

    # Обрабатываем обычные текстовые сообщения
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    # Запуск бота
    application.run_polling()

if __name__ == "__main__":
    main()
