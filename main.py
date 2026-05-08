from email.mime import image
import telebot
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps  # Installing pillow instead of PIL
import numpy as np
import PIL

bot = telebot.TeleBot("8449185864:AAFFhvzCRMaXsiHD9DHRHV1yEglssNs0mWY")
model = load_model("keras_model.h5", compile=False)

def detect_trash(img, model_path, image_path):
   np.set_printoptions(suppress=True)
   class_names = open("labels.txt", "r").readlines()
   data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   image = Image.open("img").convert("RGB")
   size = (224, 224)
   image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
   image_array = np.asarray(image)
   normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
   data[0] = normalized_image_array
   prediction = model.predict(data)
   index = np.argmax(prediction)
   class_name = class_names[index]
   confidence_score = prediction[0][index]
   return class_name[2:], confidence_score



@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь картинку мусора, и я подскажу, в какой контейнер её выбросить.")

@bot.message_handler(commands=['carrot'])
def send_hello(message):
    bot.reply_to(message, "Не выбрасывай морковь!")

@bot.message_handler(commands=['colors'])
def send_bye(message):
    bot.reply_to(message, "Синий - бумага, зелёный - стекло, жёлтый - пластик, коричневый - органика, серый - остальное.")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):

    file_info = bot.get_file(message.photo[-1].file_id)

    downloaded_file = bot.download_file(file_info.file_path)

    with open("trash.jpg", "wb") as new_file:
        new_file.write(downloaded_file)

    trash_type, confidence = detect_trash("trash.jpg")

    bot.reply_to(
        message,
        f"Это: {trash_type}\n"
        f"Точность: {confidence:.2f}"
    )

bot.polling()

