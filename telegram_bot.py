import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Твой токен бота (замени на свой)
TOKEN = "7588087338:AAGesdDnSAWFW4zHsioSEckDovg92-PXmeA"

# Инициализация логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Инициализация модели и токенизатора
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False)

# Обработка команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при старте."""
    await update.message.reply_text("Привет! Я чат-бот. Напиши мне что-нибудь!")

# Обработка текстовых сообщений
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения от пользователя."""
    user_message = update.message.text.strip()

    if not user_message:  # Игнорируем пустые сообщения
        return

    logger.debug(f"Получено сообщение: {user_message}")

    # Генерация ответа
    bot_response = generate_response(user_message, context)

    logger.debug(f"Ответ бота: {bot_response}")

    await update.message.reply_text(bot_response)  # Отправляем ответ пользователю

def generate_response(input_text, context):
    """Генерирует ответ на основе истории чата."""
    chat_history = context.user_data.get("chat_history", None)

    logger.debug(f"Генерация ответа для: {input_text}")

    # Токенизация входного текста
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # Объединение с историей (если есть)
    bot_input_ids = new_input_ids if chat_history is None else torch.cat([chat_history, new_input_ids], dim=-1)

    # Генерация ответа
    output = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(bot_input_ids)
    )

    # Обновляем историю чата
    context.user_data["chat_history"] = output

    # Декодируем ответ
    bot_response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    logger.debug(f"Сгенерированный ответ: {bot_response}")

    return bot_response

# Основная функция для запуска бота
def main() -> None:
    """Запускает Telegram-бота."""
    # Создание приложения для бота
    application = Application.builder().token(TOKEN).build()

    # Обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    # Запуск бота в режиме polling
    application.run_polling()

if __name__ == '__main__':
    main()