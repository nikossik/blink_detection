import telebot
import os
import dlib

import blink_detection
import config


# announce important variables
bot = telebot.TeleBot(config.TOKEN)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# reply on start command
@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, 'Привет!')


# reply on help command
@bot.message_handler(commands=['help'])
def help_command(message):
    bot.send_message(message.chat.id, 'Скинь мне видео, я скажу сколько раз там человек моргнул.')


# reply on text messages
@bot.message_handler(content_types=['text'])
def text_handler(message):
    bot.send_message(message.chat.id, 'Используй /help')


# work with user's video
@bot.message_handler(content_types=['video'])
def handle_docs_photo(message):
    try:
        # download video
        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        src = f"{dir_path}/video/{file_info.file_unique_id}.{file_info.file_path.split('.')[-1]}"

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Пожалуй, я сохраню это")

        # calculate blinks and reply
        ears = blink_detection.calculate_ears(src, detector, predictor)
        left_blinks, right_blinks = blink_detection.calculate_blinks(*ears)
        bot.send_message(message.chat.id, f'Левым глазом ты моргнул {left_blinks} раз, а правым {right_blinks} раз')


    except Exception as e:
        bot.reply_to(message, e)


if __name__ == "__main__":
    print('bot started!')
    bot.polling()
