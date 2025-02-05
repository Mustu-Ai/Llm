import telebot
from transformers import AutoModel, AutoTokenizer
import torch
import os

# تحميل النموذج من Hugging Face
model_name = "distilbert/distilbert-base-cased-distilled-squad"  # استبدل باسم النموذج الفعلي على Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# تهيئة بوت Telegram
TELEGRAM_BOT_TOKEN = '8098049763:AAH-20VbteLPQX5pIf4Z-lYP1N5gk9b-eWY'  # الحصول على التوكن من متغيرات البيئة
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# أمر البدء
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "مرحبًا! أنا بوت مدعوم بنموذج DeepSeek-V3. أرسل لي رسالة وسأرد عليك.")

# معالجة الرسائل النصية
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text

    # توليد رد باستخدام النموذج
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # إرسال الرد إلى المستخدم
    bot.reply_to(message, response)

# تشغيل البوت
bot.polling()
