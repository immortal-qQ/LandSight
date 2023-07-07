import asyncio
import aiogram

from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile

from app.config_reader import load_config

from aiogram.contrib.fsm_storage.memory import MemoryStorage

from aiogram.dispatcher.filters import Text

import logging
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.utils import PersonalPhotoDataset, plot_img_bbox, apply_nms, torch_to_pil

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {device}")

num_classes = 4

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

model.load_state_dict(torch.load('model/LandSight_model.pth'))

model.eval()

# Парсинг файла конфигурации
config = load_config("config/bot.ini")

# Объявление и инициализация объектов бота и диспетчера
# https://docs.aiogram.dev/en/latest/dispatcher/index.html
bot = Bot(token=config.tg_bot.token)
dp = Dispatcher(bot, storage=MemoryStorage())


async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="/start", description="Главное меню.")
    ]
    await bot.set_my_commands(commands)


@dp.message_handler(commands="start")
@dp.message_handler(Text(equals=["Назад в Главное меню..."]))
async def start(message: types.Message):
    uid = message.from_user.id  # unique id
    tag = message.from_user.username  # tag
    fname = message.from_user.first_name
    if fname is None:
        fname = ' '
    if tag is None:
        tag = f'empty_tag_{uid}'

    # print(f"[{uid}] [{tag}] [{fname}] [{lname}]")
    buttons = ['Распознать🎲', 'Назад в Главное меню...']
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(*buttons)

    out = f"Добро пожаловать в LandSight, {tag if not 'empty_tag' in tag else fname}! \n\n" \
          f"Воспользуйтесь кнопками ⬇️ экрана"

    await message.answer(out, reply_markup=keyboard)


@dp.message_handler(Text(equals=["Распознать🎲"]))
async def confirm(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['Назад в Главное меню...']
    keyboard.add(*buttons)
    await message.answer("Отправьте изображение, на котором необходимо распознать достопримечательность 🖼",
                         reply_markup=keyboard)


@dp.message_handler(content_types=['photo'])
async def get_photo(message: types.Message):
    path = os.path.join("data", "raw_photo", f"id_{message.from_user.id}", f'{message.from_user.id}.jpg')
    await message.photo[-1].download(destination_file=path)
    await detect(dir_path=os.path.join("data", "raw_photo", f"id_{message.from_user.id}"), message=message)


async def detect(dir_path, message):
    test_dataset = PersonalPhotoDataset(dir_path, 480, 480)
    img = test_dataset[0] # TODO добавить поддержку распознавания нескольких изображений
    prediction = model(torch.unsqueeze(img, 0).to(device))[0]
    # print(prediction)
    prediction = {k: v.detach().cpu() for k, v in prediction.items()}

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,10), squeeze=False)

    nms_prediction = apply_nms(prediction, iou_thresh=0.01)
    plot_img_bbox(torch_to_pil(img), nms_prediction, ax[0][0], classes=test_dataset.classes)
    # plt.show();
    plt.axis('off')
    plt.savefig(f'{dir_path}/out.png')

    photo = InputFile(f'{dir_path}/out.png')
    await bot.send_photo(chat_id=message.chat.id, photo=photo)


async def main():
    # Включаем логирование, чтобы не пропустить важные сообщения
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Установка команд бота
    await set_commands(bot)
    await dp.start_polling()


def run_bot():
    asyncio.run(main())


if __name__ == '__main__':
    run_bot()
