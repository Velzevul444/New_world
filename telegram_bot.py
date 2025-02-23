import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Токен бота (замени на свой)
TOKEN = "7588087338:AAGesdDnSAWFW4zHsioSEckDovg92-PXmeA"

# Инициализация логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Используем русскоязычную модель
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Обработка команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("сосите хуй")

# Обработка сообщений
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text.strip()
    logger.debug(f"Received message: {user_message}")

    # Храним историю диалога в user_data
    chat_history = context.user_data.get("chat_history", [])
    chat_history.append(user_message)
    if len(chat_history) > 5:  # Ограничиваем длину истории
        chat_history.pop(0)
    context.user_data["chat_history"] = chat_history

    # Генерация ответа
    bot_response = generate_response(" ".join(chat_history))
    logger.debug(f"Bot response: {bot_response}")

    await update.message.reply_text(bot_response)

def generate_response(input_text):
    """Генерация развёрнутого ответа на русском языке."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=200,  # Делаем ответы длиннее
        temperature=0.9,  # Более креативные ответы
        top_p=0.95,  # Используем семплинг
        pad_token_id=tokenizer.eos_token_id
    )

    bot_response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response

# Основная функция для запуска бота
def main() -> None:
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))
    application.run_polling()

if __name__ == "__main__":
    main()